# Tier 2 Scoring Gap Closure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three new reward components (path-length, jerk, off-limit contact) to close Tier 2 AIC scoring gaps worth up to 36 pts/trial.

**Architecture:** Three standard `RewTerm` entries in `RewardsCfg`, each curriculum-gated at stage ≥ 5. The off-limit contact also adds a `DoneTerm` for terminal episodes. All use module-level state tensors reset on episode boundary. The key fix for off-limit contact is switching from `net_forces_w` to `force_matrix_w`.

**Tech Stack:** PyTorch, IsaacLab manager-based env, RSL-RL, ContactSensor API

---

## File Map

| File | Responsibility |
|------|---------------|
| `aic/.../mdp/rewards.py` | Add 3 new reward functions + module-level state vars |
| `aic/.../mdp/terminations.py` | Add `off_limit_contact_v2` termination function |
| `aic/.../mdp/__init__.py` | Export 4 new symbols |
| `aic/.../aic_task_env_cfg.py` | Wire 3 RewTerm + 1 DoneTerm entries |

**Base path:** `aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task`

---

### Task 1: `ee_step_distance` — Path-Length Efficiency Penalty

**Files:**
- Modify: `mdp/rewards.py` (add function + module-level state at top)
- Modify: `mdp/__init__.py` (export)
- Modify: `aic_task_env_cfg.py` (add RewTerm)

- [ ] **Step 1: Add module-level state variable**

In `mdp/rewards.py`, after the existing `_ep_force_triggered` declaration (line ~92), add:

```python
# ---------------------------------------------------------------------------
# Path-length efficiency state (cumulative per-step EE distance)
# ---------------------------------------------------------------------------
_path_prev_pos: "torch.Tensor | None" = None
```

- [ ] **Step 2: Implement `ee_step_distance` function**

In `mdp/rewards.py`, add after the `completion_time_bonus` function (around line 1020):

```python
# ---------------------------------------------------------------------------
# Tier 2 gap closure — path-length efficiency
# ---------------------------------------------------------------------------
_path_prev_pos: "torch.Tensor | None"  # declared at module top


def ee_step_distance(
    env: "ManagerBasedRLEnv",
    ee_cfg: SceneEntityCfg,
    min_stage: int = 5,
    frame_idx: int = 0,
) -> torch.Tensor:
    """Per-step EE travel distance. Cumulative sum = total path length.

    Penalizes inefficient paths. Weight × sum-over-episode approximates the
    AIC efficiency metric: 6 × (1 − (path − init_dist) / 1.0).

    Curriculum-gated: returns zeros when stage < min_stage.
    """
    global _path_prev_pos

    ee_pos, _ = _ee_world_pose(env, ee_cfg, frame_idx)
    N = ee_pos.shape[0]
    dev = ee_pos.device

    # Initialize or resize
    if _path_prev_pos is None or _path_prev_pos.shape[0] != N:
        _path_prev_pos = ee_pos.clone()
        return torch.zeros(N, device=dev)

    # Reset detection: new episodes get prev_pos overwritten
    just_reset = env.episode_length_buf <= 1
    _path_prev_pos[just_reset] = ee_pos[just_reset]

    # Compute step distance
    delta = torch.norm(ee_pos - _path_prev_pos, dim=1)  # (N,)

    # Update state
    _path_prev_pos = ee_pos.clone()

    # Stage gate
    if get_current_stage() < min_stage:
        return torch.zeros(N, device=dev)

    # Zero out freshly-reset envs (no meaningful delta yet)
    delta[just_reset] = 0.0
    return delta
```

- [ ] **Step 3: Export in `__init__.py`**

In `mdp/__init__.py`, add `ee_step_distance` to the rewards import list (alphabetical within the existing block):

```python
from .rewards import completion_time_bonus, ee_step_distance  # noqa: F401
```

(Modify the existing `from .rewards import completion_time_bonus` line to include both.)

- [ ] **Step 4: Wire RewTerm in env cfg**

In `aic_task_env_cfg.py`, inside `class RewardsCfg`, add after the `timeout_penalty` entry (around line 1178):

```python
    # Tier 2 — path-length efficiency penalty (curriculum-gated at stage ≥ 5)
    # Each step's penalty = EE distance traveled. Sum ≈ total path length.
    path_efficiency = RewTerm(
        func=mdp.ee_step_distance,
        weight=-1.5,
        params={
            "ee_cfg": SceneEntityCfg(CABLE_TIP_FRAME),
            "min_stage": 5,
        },
    )
```

- [ ] **Step 5: Verify syntax**

Run:
```bash
cd /home/patrickn/IsaacLab/aic && python -c "
import ast
with open('aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py') as f:
    ast.parse(f.read())
with open('aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py') as f:
    ast.parse(f.read())
print('OK')
"
```
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
cd /home/patrickn/IsaacLab/aic && git add -A && git commit -m "feat: add ee_step_distance reward (Tier 2 path-length efficiency)

Penalizes per-step EE travel distance. Cumulative sum approximates
total path length for AIC efficiency scoring (0-6 pts/trial).
Curriculum-gated at stage >= 5.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: `ee_jerk_penalty` — Trajectory Smoothness

**Files:**
- Modify: `mdp/rewards.py` (add function + module-level state)
- Modify: `mdp/__init__.py` (export)
- Modify: `aic_task_env_cfg.py` (add RewTerm)

- [ ] **Step 1: Add module-level state variables**

In `mdp/rewards.py`, after the `_path_prev_pos` declaration:

```python
# ---------------------------------------------------------------------------
# Jerk penalty state (finite-difference acceleration & jerk)
# ---------------------------------------------------------------------------
_jerk_prev_vel: "torch.Tensor | None" = None
_jerk_prev_acc: "torch.Tensor | None" = None
```

- [ ] **Step 2: Implement `ee_jerk_penalty` function**

In `mdp/rewards.py`, add after `ee_step_distance`:

```python
def ee_jerk_penalty(
    env: "ManagerBasedRLEnv",
    ee_cfg: SceneEntityCfg,
    vel_cfg: SceneEntityCfg | None = None,
    min_speed: float = 0.01,
    min_stage: int = 5,
    frame_idx: int = 0,
) -> torch.Tensor:
    """EE linear jerk magnitude (m/s³), speed-gated.

    Targets the AIC smoothness metric: jerk computed via finite differences,
    only counted when speed > min_speed (matches official Savitzky-Golay gate).

    Returns jerk_mag × speed_gate. Multiply by a small negative weight (e.g. -0.001).
    Curriculum-gated: returns zeros when stage < min_stage.

    Args:
        ee_cfg: FrameTransformer or Articulation body for EE position.
        vel_cfg: Articulation body config for direct velocity access.
                 Required when ee_cfg is a FrameTransformer (no velocity data).
        min_speed: Speed gate threshold (m/s). Below this, jerk is not penalized.
        min_stage: Curriculum gate — only active at stage >= this.
        frame_idx: Frame index for FrameTransformer.
    """
    global _jerk_prev_vel, _jerk_prev_acc

    ee_pos, _ = _ee_world_pose(env, ee_cfg, frame_idx)
    N = ee_pos.shape[0]
    dev = ee_pos.device
    dt = env.step_dt

    # Get EE velocity from vel_cfg (Articulation body) or ee_cfg directly
    if vel_cfg is not None:
        v_asset: Articulation = env.scene[vel_cfg.name]
        vel_w = v_asset.data.body_lin_vel_w[:, vel_cfg.body_ids[0]]  # (N, 3)
    else:
        entity = env.scene[ee_cfg.name]
        if isinstance(entity, FrameTransformer):
            # Cannot get velocity from FrameTransformer — return zeros with warning
            return torch.zeros(N, device=dev)
        vel_w = entity.data.body_lin_vel_w[:, ee_cfg.body_ids[0]]

    # Initialize or resize
    if _jerk_prev_vel is None or _jerk_prev_vel.shape[0] != N:
        _jerk_prev_vel = vel_w.clone()
        _jerk_prev_acc = torch.zeros(N, 3, device=dev)
        return torch.zeros(N, device=dev)

    # Reset detection
    just_reset = env.episode_length_buf <= 1
    _jerk_prev_vel[just_reset] = vel_w[just_reset]
    _jerk_prev_acc[just_reset] = 0.0

    # Need 2+ steps of history for meaningful jerk
    too_early = env.episode_length_buf <= 2

    # Finite differences
    acc_t = (vel_w - _jerk_prev_vel) / dt          # (N, 3)
    jerk_t = (acc_t - _jerk_prev_acc) / dt         # (N, 3)
    jerk_mag = torch.norm(jerk_t, dim=1)           # (N,)

    # Speed gate: sigmoid centered at min_speed
    speed = torch.norm(vel_w, dim=1)               # (N,)
    gate = torch.sigmoid(200.0 * (speed - min_speed))  # ~0 below, ~1 above

    # Update state
    _jerk_prev_vel = vel_w.clone()
    _jerk_prev_acc = acc_t.clone()

    # Stage gate
    if get_current_stage() < min_stage:
        return torch.zeros(N, device=dev)

    result = gate * jerk_mag
    result[just_reset | too_early] = 0.0
    return result
```

- [ ] **Step 3: Export in `__init__.py`**

Update the line to:
```python
from .rewards import completion_time_bonus, ee_jerk_penalty, ee_step_distance  # noqa: F401
```

- [ ] **Step 4: Wire RewTerm in env cfg**

In `aic_task_env_cfg.py`, inside `class RewardsCfg`, add after `path_efficiency`:

```python
    # Tier 2 — jerk penalty (curriculum-gated at stage ≥ 5)
    # Penalizes rate of change of acceleration. Speed-gated at 0.01 m/s.
    jerk_penalty = RewTerm(
        func=mdp.ee_jerk_penalty,
        weight=-0.001,
        params={
            "ee_cfg": SceneEntityCfg(CABLE_TIP_FRAME),
            "vel_cfg": SceneEntityCfg("aic_unified_robot", body_names=["sfp_tip_link"]),
            "min_speed": 0.01,
            "min_stage": 5,
        },
    )
```

- [ ] **Step 5: Verify syntax**

Run same syntax check as Task 1.
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
cd /home/patrickn/IsaacLab/aic && git add -A && git commit -m "feat: add ee_jerk_penalty reward (Tier 2 smoothness)

Penalizes EE linear jerk (m/s³) via finite differences. Speed-gated
at 0.01 m/s to match official AIC scorer. Uses vel_cfg for direct
body velocity access (FrameTransformer has no velocity data).
Targets smoothness metric (0-6 pts/trial). Curriculum-gated at stage >= 5.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: `off_limit_contact_penalty_v2` + `off_limit_contact_v2` DoneTerm

**Files:**
- Modify: `mdp/rewards.py` (add v2 reward function)
- Modify: `mdp/terminations.py` (add v2 termination function)
- Modify: `mdp/__init__.py` (export both)
- Modify: `aic_task_env_cfg.py` (replace disabled entries)

- [ ] **Step 1: Implement `off_limit_contact_v2` termination**

In `mdp/terminations.py`, add after the existing `off_limit_contact` function:

```python
def off_limit_contact_v2(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("arm_contact_sensor"),
    force_threshold_n: float = 1.0,
    grace_steps: int = 3,
    min_stage: int = 5,
    excluded_bodies: tuple[str, ...] = (
        "sfp_module_link",
        "sfp_tip_link",
        "cable_link",
        "gripper_left_finger_link",
        "gripper_right_finger_link",
    ),
) -> torch.Tensor:
    """Terminate if non-EE robot body contacts off-limit entities (enclosure/board).

    Uses force_matrix_w (filtered pairwise forces) to only detect robot↔off-limit
    contacts, ignoring floor, self-collision, and gravity.

    Key fix over v1: net_forces_w includes ALL contacts (floor, self-collision),
    causing constant false positives. force_matrix_w reports ONLY contacts between
    sensor bodies and filter_prim_paths_expr targets.

    Args:
        sensor_cfg: ContactSensor with filter_prim_paths_expr for off-limit prims.
        force_threshold_n: Min force to trigger (1.0N filters PhysX solver noise).
        grace_steps: Ignore contacts for this many steps after reset (depenetration).
        min_stage: Curriculum stage gate (only active at stage >= this value).
        excluded_bodies: Body names excluded from check (EE-related, legitimately contact port).
    """
    from .curriculum import get_current_stage

    # Stage gate
    if get_current_stage() < min_stage:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    force_matrix = contact_sensor.data.force_matrix_w  # (N, B, M, 3)

    if force_matrix is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Build body exclusion mask
    body_names = contact_sensor.body_names
    include_mask = torch.ones(len(body_names), dtype=torch.bool, device=env.device)
    for i, name in enumerate(body_names):
        if any(excl in name for excl in excluded_bodies):
            include_mask[i] = False

    # Force magnitude per (body, filter_prim) pair, only included bodies
    filtered = force_matrix[:, include_mask, :, :]  # (N, B_incl, M, 3)
    force_mag = torch.norm(filtered, dim=-1)         # (N, B_incl, M)
    max_force = force_mag.reshape(env.num_envs, -1).max(dim=1).values  # (N,)

    contact_detected = max_force > force_threshold_n
    in_grace = env.episode_length_buf <= grace_steps

    return contact_detected & ~in_grace
```

- [ ] **Step 2: Implement `off_limit_contact_penalty_v2` reward**

In `mdp/rewards.py`, add after `ee_jerk_penalty`:

```python
def off_limit_contact_penalty_v2(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("arm_contact_sensor"),
    force_threshold_n: float = 1.0,
    grace_steps: int = 3,
    min_stage: int = 5,
    excluded_bodies: tuple[str, ...] = (
        "sfp_module_link",
        "sfp_tip_link",
        "cable_link",
        "gripper_left_finger_link",
        "gripper_right_finger_link",
    ),
) -> torch.Tensor:
    """Penalty for off-limit contact. Returns 1.0 on violation step, 0.0 otherwise.

    Uses force_matrix_w (filtered pairwise forces) — only detects robot↔off-limit
    contacts. Pair with a large negative weight (e.g. -50.0).

    Mirrors AIC scoring: -24 pts for any robot-enclosure/board contact.
    In sim, uses 1.0N threshold to filter PhysX solver noise.
    """
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    force_matrix = contact_sensor.data.force_matrix_w  # (N, B, M, 3)

    N = env.num_envs
    dev = env.device

    if force_matrix is None:
        return torch.zeros(N, device=dev)

    # Stage gate
    if get_current_stage() < min_stage:
        return torch.zeros(N, device=dev)

    # Build body exclusion mask
    body_names = contact_sensor.body_names
    include_mask = torch.ones(len(body_names), dtype=torch.bool, device=dev)
    for i, name in enumerate(body_names):
        if any(excl in name for excl in excluded_bodies):
            include_mask[i] = False

    # Force magnitude per (body, filter_prim) pair, only included bodies
    filtered = force_matrix[:, include_mask, :, :]  # (N, B_incl, M, 3)
    force_mag = torch.norm(filtered, dim=-1)         # (N, B_incl, M)
    max_force = force_mag.reshape(N, -1).max(dim=1).values  # (N,)

    contact_detected = max_force > force_threshold_n
    in_grace = env.episode_length_buf <= grace_steps

    return (contact_detected & ~in_grace).float()
```

- [ ] **Step 3: Export in `__init__.py`**

Add to the rewards import:
```python
from .rewards import completion_time_bonus, ee_jerk_penalty, ee_step_distance, off_limit_contact_penalty_v2  # noqa: F401
```

Add to the terminations import:
```python
from .terminations import handoff_gate, handoff_reached, off_limit_contact, off_limit_contact_v2  # noqa: F401
```

- [ ] **Step 4: Wire in env cfg — replace disabled entries**

In `aic_task_env_cfg.py`, **replace** the commented-out DoneTerm (lines 522-529) with:

```python
    # Off-limit contact v2: terminate if robot arm hits enclosure/board.
    # Uses force_matrix_w for accurate robot↔off-limit detection.
    # Curriculum-gated at stage >= 5.
    off_limit_contact = DoneTerm(
        func=mdp.off_limit_contact_v2,
        params={
            "sensor_cfg": SceneEntityCfg("arm_contact_sensor"),
            "force_threshold_n": 1.0,
            "grace_steps": 3,
            "min_stage": 5,
        },
    )
```

**Replace** the existing `off_limit_penalty` RewTerm (lines 1182-1189) with:

```python
    # Terminal penalty for off-limit contact v2 (uses force_matrix_w).
    # Fires on the same step as off_limit_contact_v2 termination.
    off_limit_penalty = RewTerm(
        func=mdp.off_limit_contact_penalty_v2,
        weight=-50.0,
        params={
            "sensor_cfg": SceneEntityCfg("arm_contact_sensor"),
            "force_threshold_n": 1.0,
            "grace_steps": 3,
            "min_stage": 5,
        },
    )
```

- [ ] **Step 5: Verify syntax**

Run same syntax check as Task 1.
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
cd /home/patrickn/IsaacLab/aic && git add -A && git commit -m "feat: add off_limit_contact_v2 reward + DoneTerm (Tier 2 contact penalty)

Replaces disabled v1 that used net_forces_w (fired constantly from
floor/self-collision noise). V2 uses force_matrix_w for filtered
pairwise detection of robot↔enclosure/board contacts only.

- 1.0N threshold (filters PhysX solver noise)
- 3-step grace period for post-reset depenetration settling
- Curriculum-gated at stage >= 5
- Terminal: episode ends on contact (mirrors AIC -24 pts rule)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: Update Scoring Print with Computed Values

**Files:**
- Modify: `mdp/rewards.py` (update the scoring table in the print block)

- [ ] **Step 1: Add episode-end path-length and jerk tracking to the capture logic**

In `mdp/rewards.py`, add module-level buffers after `_ep_force_triggered`:

```python
_ep_end_path_length: "torch.Tensor | None" = None   # (N,) cumulative path at ep end
_ep_end_jerk_mean: "torch.Tensor | None" = None     # (N,) mean jerk at ep end
_ep_cumulative_path: "torch.Tensor | None" = None   # (N,) running path total this episode
_ep_jerk_sum: "torch.Tensor | None" = None          # (N,) running jerk sum this episode
_ep_jerk_count: "torch.Tensor | None" = None        # (N,) steps with jerk computed
_ep_off_limit_triggers: int = 0                      # count of off-limit terminations this window
```

- [ ] **Step 2: Accumulate path + jerk every step in `aic_insertion_score`**

Inside `aic_insertion_score`, after the episode-end capture block (where `ending = env.reset_buf.bool()`), add accumulation:

```python
    # Accumulate path-length and jerk for episode-end capture
    global _ep_end_path_length, _ep_end_jerk_mean, _ep_cumulative_path
    global _ep_jerk_sum, _ep_jerk_count, _ep_off_limit_triggers

    if _ep_cumulative_path is None or _ep_cumulative_path.shape[0] != N:
        _ep_cumulative_path = torch.zeros(N, device=dev)
        _ep_jerk_sum = torch.zeros(N, device=dev)
        _ep_jerk_count = torch.zeros(N, device=dev)
        _ep_end_path_length = torch.zeros(N, device=dev)
        _ep_end_jerk_mean = torch.zeros(N, device=dev)

    # Accumulate path from _path_prev_pos (shared state with ee_step_distance)
    if _path_prev_pos is not None and _path_prev_pos.shape[0] == N:
        step_dist = torch.norm(ee_pos - _path_prev_pos, dim=1)
        step_dist[just_reset] = 0.0
        _ep_cumulative_path += step_dist

    # Reset accumulators for new episodes
    _ep_cumulative_path[just_reset] = 0.0
    _ep_jerk_sum[just_reset] = 0.0
    _ep_jerk_count[just_reset] = 0.0

    # Capture at episode end
    if ending.any():
        _ep_end_path_length[ending] = _ep_cumulative_path[ending]
        _ep_end_jerk_mean[ending] = torch.where(
            _ep_jerk_count[ending] > 0,
            _ep_jerk_sum[ending] / _ep_jerk_count[ending],
            torch.zeros_like(_ep_jerk_sum[ending]),
        )
```

Note: Jerk accumulation should happen inside `ee_jerk_penalty` itself — add at the end of that function:

```python
    # Accumulate for scoring print (only when active)
    if _ep_jerk_sum is not None and _ep_jerk_sum.shape[0] == N:
        active = ~just_reset & ~too_early
        _ep_jerk_sum[active] += result[active]
        _ep_jerk_count[active] += 1.0
```

(Import `_ep_jerk_sum` and `_ep_jerk_count` as globals in `ee_jerk_penalty`.)

- [ ] **Step 3: Update scoring table in print block**

Replace the "not tracked" lines for Smoothness and Efficiency with computed values:

```python
        # Compute episode-end jerk and efficiency scores
        if n_valid > 0:
            ep_mean_path = _ep_end_path_length[valid].mean().item()
            ep_mean_jerk = _ep_end_jerk_mean[valid].mean().item()
            # Efficiency: 6 × (1 - (path - 0.15) / 1.0), clamped [0, 6]
            ep_eff_score = max(0.0, min(6.0, 6.0 * (1.0 - (ep_mean_path - 0.15) / 1.0)))
            # Smoothness: 6 × (1 - jerk/50), clamped [0, 6]
            ep_smooth_score = max(0.0, min(6.0, 6.0 * (1.0 - ep_mean_jerk / 50.0)))
            # Apply T2 gating
            ep_eff_gated = ep_eff_score if ep_mean_t3 > 0 else 0.0
            ep_smooth_gated = ep_smooth_score if ep_mean_t3 > 0 else 0.0
        else:
            ep_mean_path = 0.0
            ep_mean_jerk = 0.0
            ep_eff_gated = 0.0
            ep_smooth_gated = 0.0
```

Then in the print f-string, update the Smoothness and Efficiency lines:

```python
            f"  │  {'':6}{'Smoothness (jerk)':<26}{f'jerk={ep_mean_jerk:.1f} m/s³':<22}{f'~{ep_smooth_gated:.1f}':>6}{'+6':>6}\n"
            f"  │  {'':6}{'Duration (≤5s→12,≥60s→0)':<26}{f'med {med_dur_s:.0f}s [{dur_bar}]':<22}{f'~{ep_mean_dur_gated:.1f}':>6}{'+12':>6}\n"
            f"  │  {'':6}{'Efficiency (path length)':<26}{f'path={ep_mean_path:.3f}m':<22}{f'~{ep_eff_gated:.1f}':>6}{'+6':>6}\n"
```

And update the Off-limit contact line:

```python
            f"  │  {'':6}{'Off-limit contact (→ −24)':<26}{f'{_ep_off_limit_triggers} triggers':<22}{'--':>6}{'0':>6}\n"
```

Update `est_total` to include the new scores:

```python
            ep_est_total = 1.0 + ep_mean_dur_gated + ep_smooth_gated + ep_eff_gated + ep_mean_force + ep_mean_t3
```

Update the "Not tracked" disclaimer:

```python
            f"  │  {'':6}{'Not tracked: wrong-port':<48}{'':<12}\n"
```

- [ ] **Step 4: Track off-limit trigger count**

In `off_limit_contact_penalty_v2`, at the end, increment the counter:

```python
    # Track for scoring print
    global _ep_off_limit_triggers
    if _ep_off_limit_triggers is None:
        _ep_off_limit_triggers = 0
    _ep_off_limit_triggers += int((contact_detected & ~in_grace).sum().item())
```

Reset it in the print block alongside `_aic_scored_count = 0`:

```python
        _ep_off_limit_triggers = 0
```

- [ ] **Step 5: Verify syntax**

Run same syntax check.
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
cd /home/patrickn/IsaacLab/aic && git add -A && git commit -m "feat: update scoring print with computed jerk, efficiency, off-limit

Replace 'not tracked' placeholders with episode-end computed values:
- Smoothness: mean jerk (m/s³) → estimated score
- Efficiency: mean path length → estimated score  
- Off-limit: trigger count per print window
- TOTAL now includes all tracked Tier 2 components

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 5: Final Validation

- [ ] **Step 1: Full syntax check of all modified files**

```bash
cd /home/patrickn/IsaacLab/aic && python -c "
import ast, pathlib
base = pathlib.Path('aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task')
for f in ['mdp/rewards.py', 'mdp/terminations.py', 'mdp/__init__.py', 'aic_task_env_cfg.py']:
    path = base / f
    ast.parse(path.read_text())
    print(f'✅ {f}')
print('All files OK')
"
```

- [ ] **Step 2: Import check**

```bash
cd /home/patrickn/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task && python -c "
from aic_task.tasks.manager_based.aic_task.mdp import (
    ee_step_distance,
    ee_jerk_penalty,
    off_limit_contact_penalty_v2,
    off_limit_contact_v2,
)
print('✅ All imports resolve')
"
```

If this fails due to missing IsaacLab dependencies (not in PYTHONPATH), confirm at least the AST parse passes — the full import test requires the sim environment.

- [ ] **Step 3: Review git log**

```bash
cd /home/patrickn/IsaacLab/aic && git log --oneline -6
```

Expected: 4 clean commits (Tasks 1–4: path-length, jerk, off-limit, scoring print).
