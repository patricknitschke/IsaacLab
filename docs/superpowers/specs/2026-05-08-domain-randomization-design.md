# Domain Randomization Design Spec

**Date**: 2026-05-08
**Status**: Draft — awaiting approval
**Prerequisite**: None (DR-8 uniform scaling can be implemented now; DR-1–7 wait for stage 20)

---

## 1. Problem Statement

The AIC evaluation randomizes task board pose, NIC card rail/translation/yaw, SC port
positions, grasp perturbations, and target port selection. Current training has minimal
DR (±5mm board XY, 4 discrete NIC card Y positions, no yaw, no board rotation). The
policy will fail to generalize under eval conditions.

Additionally, the scaling curriculum only scales the NIC card mesh while leaving the
board and other components at 1.0×, creating visual disproportion that will break
camera-based observations.

## 2. Goals

1. **Uniform board scaling**: All task board components scale proportionally by the
   curriculum scale factor at every stage.
2. **Eval-matching DR**: After curriculum completion, randomize all parameters to match
   AIC evaluation conditions.
3. **Progressive introduction**: DR ranges widen through curriculum stages to avoid
   catastrophic policy collapse.
4. **Camera readiness**: Scene looks visually realistic at every curriculum stage.

---

## 3. Architecture Overview

### 3.1 Current Flow (on episode reset)

```
EventTerm execution order:
  1. reset_robot_joints          → random joint positions
  2. randomize_robot_pose        → random robot base XY
  3. randomize_dome_light        → random lighting
  4. randomize_board_and_parts   → board XY, part offsets (FIXED orientations)
  5. apply_scale_curriculum_event → position scaled NIC card + port marker
  6. near_port_curriculum         → 50% resets use mined near-port states
```

### 3.2 Proposed Flow

```
EventTerm execution order:
  1. reset_robot_joints
  2. randomize_robot_pose
  3. randomize_dome_light
  4. randomize_board_and_parts   → MODIFIED: accepts scale, yaw ranges,
                                    scales ALL offsets, rotates parts
  5. apply_scale_curriculum_event → MODIFIED: also scales board + SC ports
                                    via USD Xform scale
  6. near_port_curriculum
```

### 3.3 Key Invariant

The **target port world position** must remain approximately constant across scales
so the robot can always reach it. This is achieved by anchoring the board position
relative to the port:

```
board_pos = desired_port_pos − scale × offset_board_to_port
```

As scale decreases (2.0 → 1.0), the board center moves closer to the port position.
At scale=1.0, the board center is at its nominal position.

---

## 4. Detailed Design

### 4.1 DR-8: Uniform Board Scaling

**Files**: `events.py`, `aic_task_env_cfg.py`

#### 4.1.1 `randomize_board_and_parts` Changes

Add parameters:
```python
def randomize_board_and_parts(
    env, env_ids,
    board_scene_name="task_board",
    board_default_pos=(0.2837, 0.229, 0.0),     # nominal board center (1.0×)
    board_range={"x": (-0.005, 0.005), "y": (-0.005, 0.005)},
    board_yaw_range=(0.0, 0.0),                  # NEW: (min_rad, max_rad)
    parts=[...],
    port_anchor_offset=(-0.03235, 0.02329, 0.0743),  # NEW: board→port offset at 1.0×
    sync_usd_xforms=True,
):
```

Core logic changes:
1. **Read scale**: `from .curriculum import get_current_stage, SCALE_STAGES`
   `scale = SCALE_STAGES[get_current_stage()]`

2. **Compute board position** (port-anchored):
   ```python
   # Port should stay at a fixed reachable world position
   port_nominal = board_default_pos + port_anchor_offset  # world pos at 1.0×
   board_pos = port_nominal - scale * port_anchor_offset  # shift board so port stays put
   board_pos += random_xy(board_range)                     # add small XY jitter
   ```

3. **Board yaw**:
   ```python
   yaw = uniform(board_yaw_range)
   board_rot = compose_yaw(_cached_orientations[board_name], yaw)
   ```

4. **Scale part offsets**:
   ```python
   for part in parts:
       scaled_offset = (ox * scale, oy * scale, oz * scale)
       scaled_pose_range = {k: (lo*scale, hi*scale) for k,(lo,hi) in pose_range}
       scaled_snap_step = {k: v*scale for k,v in snap_step}
   ```

5. **Rotate part offsets by board yaw**:
   ```python
   rotated_offset = quat_apply(board_rot, scaled_offset)
   part_pos = board_world_pos + rotated_offset + random_delta(scaled_pose_range)
   ```

6. **Part yaw** (for NIC card ±10°):
   ```python
   if "yaw_range" in part_cfg:
       part_yaw = uniform(part_cfg["yaw_range"])  # scaled by DR curriculum
       part_rot = compose_yaw(board_rot, part_yaw)  # relative to board
   ```

#### 4.1.2 `apply_scale_curriculum_event` Changes

After positioning the NIC card + port marker (existing logic), also scale visual
geometry for board and SC ports:

```python
# Scale board mesh
board = env.scene["task_board"]
_set_usd_xform_scale(stage, board.cfg.prim_path, env_ids, (scale, scale, scale))

# Scale SC port meshes
for sc_name in ["sc_port", "sc_port_2"]:
    sc = env.scene[sc_name]
    _set_usd_xform_scale(stage, sc.cfg.prim_path, env_ids, (scale, scale, scale))
```

New helper:
```python
def _set_usd_xform_scale(stage, prim_path_template, env_ids, scale_xyz):
    """Set USD Xform scale for visual mesh scaling."""
    for env_id in env_ids.tolist():
        prim_path = _ENV_REGEX_RE.sub(f"env_{env_id}", prim_path_template)
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        xf = UsdGeom.Xformable(prim)
        for op in xf.GetOrderedXformOps():
            if "scale" in op.GetOpName():
                op.Set(Gf.Vec3f(*scale_xyz))
                break
        else:
            xf.AddScaleOp().Set(Gf.Vec3f(*scale_xyz))
```

**Note**: This scales the visual mesh only. Collision geometry for SC ports is irrelevant
(robot doesn't insert into them). Board collision is kinematic. Only the NIC card's
collision geometry matters, and that's already handled by the pre-scaled USD assets.

#### 4.1.3 Config Changes (`aic_task_env_cfg.py`)

```python
randomize_board_and_parts = EventTerm(
    func=randomize_board_and_parts,
    mode="reset",
    params={
        "board_scene_name": "task_board",
        "board_default_pos": (0.2837, 0.229, 0.0),
        "board_range": {"x": (-0.005, 0.005), "y": (-0.005, 0.005)},
        "board_yaw_range": (0.0, 0.0),        # DR-2 widens this later
        "port_anchor_offset": (-0.03235, 0.02329, 0.0743),
        "parts": [
            {
                "scene_name": "sc_port",
                "offset": (0.0067, -0.0362, 0.005),
                "pose_range": {"x": (-0.005, 0.02)},
            },
            {
                "scene_name": "sc_port_2",
                "offset": (0.0076, -0.0783, 0.005),
                "pose_range": {"x": (-0.005, 0.02)},
            },
            {
                "scene_name": "nic_card",
                "offset": (-0.03235, 0.02329, 0.0743),
                "pose_range": {"y": (0.0, 0.12)},
                "snap_step": {"y": 0.04},
                "yaw_range": (0.0, 0.0),      # DR-1 sets to (-0.1745, 0.1745)
            },
        ],
    },
)
```

---

### 4.2 DR-1: NIC Card Pose Randomization

**Files**: `aic_task_env_cfg.py` (config params only, after DR-8 infrastructure)

Changes to `randomize_board_and_parts` params:
```python
# NIC card part config:
{
    "scene_name": "nic_card",
    "offset": (-0.03235, 0.02329, 0.0743),
    "pose_range": {"y": (0.0, 0.16)},    # 5 rails: 0/40/80/120/160mm (was 0-120)
    "snap_step": {"y": 0.04},             # unchanged: 40mm between rails
    "yaw_range": (-0.1745, 0.1745),       # ±10° (NEW)
    "translation_range": {"x": (-0.023, 0.023)},  # ±23mm along rail (NEW)
}
```

**Note**: `pose_range` and `snap_step` get auto-scaled by `scale` in the function body.
At 2.0× scale, snap_step becomes 80mm, pose_range becomes 0–320mm, etc.

---

### 4.3 DR-2: Task Board Pose Randomization

**Files**: `aic_task_env_cfg.py` (config params only)

```python
"board_range": {"x": (-0.10, 0.10), "y": (-0.10, 0.10)},   # ±100mm (was ±5mm)
"board_yaw_range": (-0.087, 0.087),                          # ±5° (eval undisclosed)
```

**Important**: Board XY range is NOT scaled — it's world-space jitter. Board yaw applies
to the board orientation, and all parts inherit it via rotated offsets.

---

### 4.4 DR-3: Target Port Selection

**Files**: `events.py`, `aic_task_env_cfg.py`

Each NIC card has 2 SFP ports. The port_local_offset in `apply_scale_curriculum_event`
currently hardcodes port 0's position.

Changes:
```python
def apply_scale_curriculum_event(
    env, env_ids,
    nic_card_name="nic_card",
    marker_name="port_frame_marker",
    port_offsets={                            # NEW: both ports
        0: (0.01295, -0.0751, 0.00530),       # SFP_PORT_0
        1: (0.01295, -0.0751 + Δy, 0.00530),  # SFP_PORT_1 (offset TBD from USD)
    },
    port_local_rot=(0.707, 0.0, 0.0, 0.707),
):
```

Per-env port selection:
```python
# Randomize target port per environment
target_port = torch.randint(0, 2, (num_envs,), device=device)
# Select offset per env
port_offset = torch.where(target_port.unsqueeze(-1) == 0,
                          port_offsets[0], port_offsets[1])
```

**Prerequisite**: Measure SFP_PORT_1 offset from USD asset.

---

### 4.5 DR-4: Grasp Perturbation

**Files**: `events.py`, `aic_task_env_cfg.py`

New EventTerm (runs after robot joint reset, before board randomization):
```python
def randomize_grasp_perturbation(
    env, env_ids,
    cable_cfg: SceneEntityCfg,
    translation_range: float = 0.002,    # ±2mm
    rotation_range: float = 0.04,        # ±0.04 rad (~2.3°)
):
    """Apply small perturbation to grasped cable relative to gripper."""
```

This offsets the cable-in-hand pose slightly, simulating imperfect grasps.

---

### 4.6 DR-5: Enable Observation Corruption

**Files**: `aic_task_env_cfg.py`

Single-line change:
```python
class PolicyCfg:
    enable_corruption = True   # was False
```

Existing noise levels are already configured and reasonable.

---

### 4.7 DR-6: Progressive DR Curriculum

**Files**: `curriculum.py`

Extend SCALE_STAGES and PARAM_STAGES with DR stages 21–25:

| Stage | Scale | DR Level |
|-------|-------|----------|
| 0–6   | 2.0×–1.14× | No extra DR (scaling provides difficulty) |
| 7–20  | 1.10×–1.0× | No extra DR (param tightening provides difficulty) |
| 21    | 1.0× | Obs noise ON, NIC yaw ±2° |
| 22    | 1.0× | NIC yaw ±5°, translation ±10mm |
| 23    | 1.0× | Full NIC DR (±10° yaw, ±23mm, all 5 rails) |
| 24    | 1.0× | Board pose DR (±100mm XY, ±5° yaw) + grasp perturbation |
| 25    | 1.0× | Full eval-matching DR |

Implementation: Add `dr_level` field to PARAM_STAGES. `randomize_board_and_parts`
reads `dr_level` and selects appropriate ranges:

```python
DR_RANGES = {
    0: {"nic_yaw": 0.0, "nic_trans": 0.0, "board_range": 0.005, "board_yaw": 0.0},
    1: {"nic_yaw": 0.035, "nic_trans": 0.0, "board_range": 0.005, "board_yaw": 0.0},
    2: {"nic_yaw": 0.087, "nic_trans": 0.010, "board_range": 0.005, "board_yaw": 0.0},
    3: {"nic_yaw": 0.1745, "nic_trans": 0.023, "board_range": 0.005, "board_yaw": 0.0},
    4: {"nic_yaw": 0.1745, "nic_trans": 0.023, "board_range": 0.10, "board_yaw": 0.087},
    5: {"nic_yaw": 0.1745, "nic_trans": 0.023, "board_range": 0.10, "board_yaw": 0.087},
}
```

---

### 4.8 DR-7: Force Clamping

**Files**: `rewards.py`

In `aic_insertion_score` and `contact_force_penalty`, clamp force magnitude:
```python
force_mag = force_mag.clamp(max=200.0)  # 200N cap, real insertion < 50N
```

This prevents PhysX solver spike artifacts (observed: max 13.5kN) from corrupting
reward signals and triggering false T2 force penalties.

Does NOT affect physics simulation — only the reward/scoring computation.

---

## 5. File Change Summary

| File | Changes |
|------|---------|
| `events.py` | Modify `randomize_board_and_parts` (scale, yaw, rotated offsets). Modify `apply_scale_curriculum_event` (board/SC USD scale). Add `_set_usd_xform_scale` helper. Add `randomize_grasp_perturbation`. |
| `aic_task_env_cfg.py` | Update EventTerm params (board_yaw_range, port_anchor_offset, part yaw_range). Add grasp perturbation EventTerm. Set `enable_corruption=True`. |
| `curriculum.py` | Add DR stages 21–25. Add `dr_level` to PARAM_STAGES. Export DR_RANGES. |
| `rewards.py` | Add `force_mag.clamp(max=200.0)` in force-related rewards. |

---

## 6. Implementation Order

```
Phase 1: Infrastructure (can start now)
  DR-8  Uniform board scaling (events.py refactor)
  DR-7  Force clamping (rewards.py, 1-line change)

Phase 2: Eval-critical DR (after stage 20)
  DR-1  NIC card pose (config params)
  DR-2  Board pose (config params)
  DR-3  Port selection (events.py + config)
  DR-4  Grasp perturbation (new event function)
  DR-5  Obs noise (config flag)

Phase 3: Curriculum extension
  DR-6  Progressive DR curriculum (curriculum.py)
```

DR-8 is the foundation — all other DR tasks build on its infrastructure (yaw support,
scaled offsets, rotated part placement).

---

## 7. Validation Plan

1. **Visual check**: At each curriculum stage, verify board + parts look proportional
   in the viewport (livestream mode).
2. **Port reachability**: Verify the robot can reach the port at scale 2.0× and 1.0×
   by checking IK success rate in early episodes.
3. **Reward sanity**: After DR-7, verify force averages drop from ~387N to capped
   values. Check T2 force penalty rate in scoring print.
4. **DR rollout test**: At stage 25 (full DR), run 1000 episodes and verify >80%
   insertion success before declaring DR complete.
5. **Regression**: Compare insertion success at stage 20 (no DR) vs stage 25 (full DR).
   Acceptable regression: <15% success rate drop.

---

## 8. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| USD Xform scale doesn't affect collision | Board/SC visual-only scaling | Acceptable: only NIC card collision matters for insertion |
| Port-anchored board position at 2× pushes board edge out of env bounds | Overlap with adjacent envs | Increase env spacing if needed |
| Yaw rotation of part offsets introduces numerical error | Parts drift off board | Unit test: verify part position at multiple yaw values |
| DR curriculum stages are too many (25 stages) | Training takes too long | Can skip DR stages or loosen advancement threshold |
| Camera obs + scaled board at stage <10 has visual gap | Vision module confused | Accept: vision not critical before stage 10 (scale ≤1.05×) |
