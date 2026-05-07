# AIC Tier 2 Scoring Gap Closure — Design Spec

**Date**: 2026-05-07  
**Status**: Draft  
**Scope**: Add path-length, jerk, and off-limit contact rewards to close Tier 2 scoring gaps  

## Problem

The reward system covers Tier 3 insertion (+75 pts) with ~7600 weight budget but has three unaddressed Tier 2 scoring components:

| Gap | Official Score Impact | Current Coverage |
|-----|----------------------|-----------------|
| Path-length efficiency | 0–6 pts/trial | None |
| Trajectory smoothness (jerk) | 0–6 pts/trial | Indirect (action_rate only) |
| Off-limit contact | −24 pts/trial | Reward exists but disabled |

Maximum scoring risk: 36 pts/trial × 3 trials = 108 pts unaddressed.

## Design Decisions

- **Curriculum-gated**: All three activate at `get_current_stage() >= 5` (agent must first learn insertion)
- **Conservative weights**: New penalties are 100–1000× smaller than insertion rewards during useful motion
- **Off-limit = terminal**: Episode ends on contact (mirrors competition: any contact invalidates quality)
- **TensorBoard diagnostics**: All three log metrics for monitoring

## Architecture

All three are standard `RewTerm` entries in `RewardsCfg`, gated internally by the existing curriculum stage system. No changes to env infrastructure, observation space, or termination logic (except re-enabling the off-limit DoneTerm with modifications).

```
rewards.py
├── ee_step_distance()        ← NEW
├── ee_jerk_penalty()         ← NEW  
├── off_limit_contact_penalty_v2()  ← NEW (replaces disabled v1)
└── [existing 26 terms unchanged]

aic_task_env_cfg.py
├── RewardsCfg
│   ├── path_efficiency = RewTerm(func=mdp.ee_step_distance, weight=-1.5, ...)
│   ├── jerk_penalty = RewTerm(func=mdp.ee_jerk_penalty, weight=-0.001, ...)
│   └── off_limit_penalty = RewTerm(func=mdp.off_limit_contact_penalty_v2, weight=-50.0, ...)
└── TerminationsCfg
    └── off_limit_contact = DoneTerm(func=mdp.off_limit_contact_v2, ...)  ← RE-ENABLED
```

## Component 1: `ee_step_distance` (Path-Length Efficiency)

### Purpose
Penalize cumulative EE path length. Each step's penalty = distance traveled that step. Sum over episode = total path length.

### Official Scoring Reference
- Score = 6 × (1 − (path − initial_dist) / 1.0)
- Path ≤ initial_dist → 6 pts; Path ≥ initial_dist + 1m → 0 pts
- Only awarded when Tier 3 > 0

### Function Signature
```python
_path_prev_pos: torch.Tensor | None = None

def ee_step_distance(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    min_stage: int = 5,
    frame_idx: int = 0,
) -> torch.Tensor:
```

### Logic
1. Get current EE position (cable tip)
2. If first step or just reset: initialize prev_pos, return zeros
3. Compute `delta = ‖pos_t − prev_pos‖` per env
4. Update prev_pos
5. Gate: return `delta` if `get_current_stage() >= min_stage`, else zeros

### State Management
- Module-level `_path_prev_pos: Tensor | None` — shape (N, 3)
- Reset detection: `env.episode_length_buf <= 1` → overwrite prev_pos with current pos

### Weight & Magnitude
| Scenario | EE speed | Step delta | Weighted penalty |
|----------|----------|-----------|-----------------|
| Fast approach (0.1 m/s) | 0.1 | 3.3 mm | −0.005 |
| Near port (0.03 m/s) | 0.03 | 1.0 mm | −0.0015 |
| Inserting (0.01 m/s) | 0.01 | 0.33 mm | −0.0005 |
| Oscillating | 0.03 | 1.0 mm | −0.0015 |

**Weight: −1.5**

### Diagnostic
Log to TensorBoard every print interval:
- `tier2/path_length_mean` — mean cumulative path (m) across envs this window

## Component 2: `ee_jerk_penalty` (Trajectory Smoothness)

### Purpose
Penalize EE linear jerk (rate of change of acceleration). Directly targets the official smoothness metric.

### Official Scoring Reference
- Jerk computed via Savitzky-Golay filter (15-sample window) on velocity
- Only accumulated when speed > 0.01 m/s
- Score = 6 × (1 − jerk/50); jerk = 0 → 6 pts; jerk ≥ 50 m/s³ → 0 pts
- Only awarded when Tier 3 > 0

### Function Signature
```python
_jerk_prev_vel: torch.Tensor | None = None
_jerk_prev_acc: torch.Tensor | None = None

def ee_jerk_penalty(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    min_speed: float = 0.01,
    min_stage: int = 5,
    frame_idx: int = 0,
) -> torch.Tensor:
```

### Logic
1. Get EE linear velocity (from Articulation body_lin_vel_w)
2. Compute `acc_t = (vel_t − prev_vel) / dt`
3. Compute `jerk_t = (acc_t − prev_acc) / dt`
4. Compute `jerk_mag = ‖jerk_t‖`
5. Apply speed gate: `gate = sigmoid(200 × (speed − min_speed))`
6. Return `gate × jerk_mag` if `get_current_stage() >= min_stage`, else zeros
7. Update state: prev_vel = vel_t, prev_acc = acc_t

### State Management
- Module-level `_jerk_prev_vel`, `_jerk_prev_acc` — shape (N, 3)
- Reset detection: `env.episode_length_buf <= 1` → init from current vel, zero acc
- Need 2 steps of history before jerk is meaningful — return 0 for steps 0-1

### Weight & Magnitude
| Motion type | Typical jerk (m/s³) | Weighted penalty |
|-------------|--------------------|-----------------| 
| Smooth constant velocity | 0–2 | −0.000002 |
| Starting/stopping | ~9 | −0.009 |
| Jerky oscillation | ~36 | −0.036 |
| Violent reversal | ~90 | −0.09 |

**Weight: −0.001**

### Why Not Savitzky-Golay
- SG needs 15-sample window state per env (added complexity)
- Sim velocities are cleaner than hardware (less noise to filter)
- Simple finite differences provide correct directional gradient
- Can add EMA smoothing later if noise is problematic

### Diagnostic
Log to TensorBoard every print interval:
- `tier2/jerk_mean` — mean jerk magnitude (m/s³) across envs
- `tier2/jerk_max` — max jerk observed

## Component 3: `off_limit_contact_penalty_v2` + DoneTerm

### Purpose
Terminate episode and apply large penalty when any robot link contacts an off-limit entity (enclosure, walls, task board).

### Official Scoring Reference
- Binary: ANY geometric contact between robot model and off-limit models → −24 pts
- No force threshold in official scoring
- Off-limit models: `enclosure`, `enclosure walls`, `task_board` (includes NIC card, SFP port)
- Entire robot model monitored (including gripper fingers)
- Cable model is exempt

### Why Previously Disabled
The original implementation used `net_forces_w` which reports **ALL** contact forces on each body (including floor, self-collision, gravity). This produces non-zero values constantly, triggering false positives regardless of actual enclosure contact. Additionally, PhysX depenetration at reset generates transient forces.

### Root Cause Fix: `force_matrix_w`
The ContactSensor's `force_matrix_w` field (shape: `(N, B, M, 3)`) reports forces **only** between sensor bodies and the `filter_prim_paths_expr` targets (enclosure, task_board). This eliminates floor/self-collision/gravity noise entirely.

| Field | Reports | Use |
|-------|---------|-----|
| `net_forces_w` | ALL contacts (floor, self, everything) | ❌ Wrong for off-limit detection |
| `force_matrix_w` | Only robot ↔ filter targets | ✅ Correct: enclosure/board only |

### Function Signature (Reward)
```python
def off_limit_contact_penalty_v2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold_n: float = 1.0,
    grace_steps: int = 3,
    min_stage: int = 5,
    excluded_bodies: tuple[str, ...] = (
        "sfp_module_link", "sfp_tip_link", "cable_link",
        "gripper_left_finger_link", "gripper_right_finger_link",
    ),
) -> torch.Tensor:
```

### Function Signature (Termination)
```python
def off_limit_contact_v2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold_n: float = 1.0,
    grace_steps: int = 3,
    min_stage: int = 5,
    excluded_bodies: tuple[str, ...] = (
        "sfp_module_link", "sfp_tip_link", "cable_link",
        "gripper_left_finger_link", "gripper_right_finger_link",
    ),
) -> torch.Tensor:
```

### Logic (shared between reward and DoneTerm)
1. Read `contact_sensor.data.force_matrix_w` — shape `(N, B, M, 3)`, filtered pairwise forces
2. Exclude EE-related bodies (they legitimately contact the port)
3. Compute `max_force_per_env = max over included bodies × all filter prims`
4. Check `triggered = max_force > force_threshold_n`
5. Apply grace: `past_grace = env.episode_length_buf > grace_steps`
6. Apply stage gate: `active = get_current_stage() >= min_stage`
7. Return `triggered & past_grace & active`

### Design Choices
| Choice | Value | Rationale |
|--------|-------|-----------|
| Data field | `force_matrix_w` | Only reports robot↔filter_prim contacts. Eliminates floor/self/gravity noise. |
| Force threshold | 1.0 N | Filters PhysX solver noise. Standard in IsaacLab (`illegal_contact` uses 1.0N). |
| Grace period | 3 steps (0.1s) | Depenetration settles in 1-2 steps; 3 gives margin. Reduced from 5 since `force_matrix_w` eliminates floor contact noise. |
| Terminal | Yes (DoneTerm) | Mirrors competition: contact invalidates trial quality |
| Reward weight | −50.0 | One-shot terminal. Comparable to timeout_penalty (−10) but more severe. |
| Stage gate | ≥ 5 | Don't punish early exploration |
| Excluded bodies | gripper, cable, sfp links | These legitimately contact the port during insertion |

### Sim-to-Real Gap
- Official: ANY geometric contact = penalty (no force check)
- Sim: 1.0 N threshold required to filter numerical artifacts
- Implication: policy trained in sim may still occasionally produce sub-threshold brushes on hardware. Acceptable trade-off for training stability.

### Diagnostic
Log to TensorBoard every print interval:
- `tier2/off_limit_triggers` — number of episodes terminated by off-limit contact this window
- `tier2/off_limit_max_force` — max force observed on monitored bodies (before threshold)

## Curriculum Integration

All three rewards check `get_current_stage()` internally:
```python
from .curriculum import get_current_stage

# Inside each function:
if get_current_stage() < min_stage:
    return torch.zeros(N, device=env.device)
```

This means:
- **Stages 1–4**: Agent learns approach + insertion with full existing reward budget
- **Stages 5–7**: Tier 2 quality penalties activate, refining motion quality
- No weight changes needed — binary activation via stage gate

## Scoring Print Update

Add to the existing scoring table:
```
│        Smoothness  (jerk, 0→6)         jerk={mean_jerk:.1f} m/s³         ~{jerk_score:.1f} /  6
│        Efficiency  (path length, 0→6)  path={mean_path:.3f}m             ~{eff_score:.1f} /  6
│        Off-limit contact (→ -24)       {trigger_count} triggers           {contact_pts} /  0
```

Replace "not tracked in sim" with computed values for jerk and efficiency. Off-limit now shows actual trigger count.

## File Changes Summary

| File | Change |
|------|--------|
| `mdp/rewards.py` | Add `ee_step_distance`, `ee_jerk_penalty`, `off_limit_contact_penalty_v2` |
| `mdp/__init__.py` | Export new functions |
| `aic_task_env_cfg.py` | Add 3 RewTerm entries + modify DoneTerm |
| `mdp/rewards.py` (print block) | Update scoring table with computed values |

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Path penalty slows approach | Low (60× weaker than approach pull) | Monitor approach time in TB |
| Jerk penalty prevents acceleration | Low (−0.009 at startup vs +4 approach reward) | Speed gate filters stationary |
| Off-limit false positives | Medium | Grace period + 1.0N threshold + stage gate |
| Off-limit during insertion (wrist near board) | Medium | Policy must learn clearance — this IS the desired behavior |
| Stage 5 activation destabilizes | Low | Weights are very conservative | Monitor insertion rate after stage transition |

## Success Criteria

1. **Path efficiency**: Mean path length < 0.3m for successful insertions (straight-line ≈ 0.15m)
2. **Jerk**: Mean jerk < 25 m/s³ (would score ~3/6 pts on hardware)
3. **Off-limit**: < 1% of episodes terminated by off-limit after 1000 post-stage-5 iterations
4. **No regression**: Insertion success rate doesn't drop > 5% after stage 5 activation
