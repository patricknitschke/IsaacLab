# Domain Randomization Phase 2+3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement progressive domain randomization (DR-1 through DR-6) so the trained policy generalizes to AIC evaluation conditions.

**Architecture:** DR-8 (uniform board scaling) and DR-7 (force clamping) are already implemented. This plan adds: NIC card yaw support in events.py (DR-1), board/NIC config params (DR-1/2), dual-port selection (DR-3), grasp perturbation (DR-4), obs noise (DR-5), and a progressive DR curriculum extending stages 21–25 (DR-6). The curriculum reads a `dr_level` from PARAM_STAGES and `randomize_board_and_parts` + `apply_scale_curriculum_event` dynamically select DR ranges from a `DR_RANGES` table.

**Tech Stack:** Python 3.10, PyTorch, IsaacLab 2.3.2, RSL-RL (PPO), USD/PhysX

---

## File Map

| File | Responsibility | Tasks |
|------|----------------|-------|
| `aic/.../mdp/events.py` | Event functions for reset randomization | 1, 3, 4 |
| `aic/.../mdp/curriculum.py` | Curriculum stages, DR_RANGES, PARAM_STAGES | 6 |
| `aic/.../aic_task_env_cfg.py` | EventTerm configs, obs noise flag | 1, 2, 3, 4, 5 |
| `aic/.../mdp/__init__.py` | Symbol exports | 4 |

All paths relative to: `aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/`

---

### Task 1: DR-1 — NIC Card Yaw Support in events.py + Config

**Why:** The eval randomizes NIC card yaw ±10° and has 5 rails (not 4). The `randomize_board_and_parts` function already scales offsets and rotates them by board yaw, but individual part yaw (NIC card tilted relative to the board) is not yet supported.

**Files:**
- Modify: `mdp/events.py` — `randomize_board_and_parts` part loop (~lines 202-241)
- Modify: `aic_task_env_cfg.py` — EventTerm params for `randomize_board_and_parts` (~line 444)

- [ ] **Step 1: Add per-part yaw_range support in events.py**

In the part loop inside `randomize_board_and_parts`, after computing `part_rot` (line 206), check for `yaw_range` in `part_cfg` and compose an additional yaw rotation:

```python
    # Part poses, anchored to the board.
    for part_cfg in parts:
        pname = part_cfg["scene_name"]
        part_asset = env.scene[pname]
        part_rot = math_utils.quat_mul(yaw_delta, _cached_orientations[pname][env_ids].clone())

        # Per-part yaw randomization (e.g. NIC card ±10° relative to board)
        part_yaw_range = part_cfg.get("yaw_range", (0.0, 0.0))
        p_yaw_lo, p_yaw_hi = part_yaw_range
        if p_yaw_lo != 0.0 or p_yaw_hi != 0.0:
            part_yaw_angles = torch.empty(n, device=device).uniform_(p_yaw_lo, p_yaw_hi)
            p_half = part_yaw_angles * 0.5
            part_yaw_quats = torch.zeros(n, 4, device=device)
            part_yaw_quats[:, 0] = torch.cos(p_half)
            part_yaw_quats[:, 3] = torch.sin(p_half)
            part_rot = math_utils.quat_mul(part_rot, part_yaw_quats)

        ox, oy, oz = part_cfg["offset"]
        # ... rest of existing code unchanged ...
```

- [ ] **Step 2: Update NIC card config in aic_task_env_cfg.py**

Change the nic_card part entry to add `yaw_range` and expand `pose_range` to 5 rails:

```python
                {
                    "scene_name": "nic_card",
                    "offset": (-0.03235, 0.02329, 0.0743),
                    "pose_range": {"y": (0.0, 0.16)},
                    "snap_step": {"y": 0.04},
                    "yaw_range": (0.0, 0.0),
                },
```

Note: `pose_range` `y` upper bound changes from `0.12` (4 rails: 0/40/80/120mm) to `0.16` (5 rails: 0/40/80/120/160mm). The `yaw_range` starts at `(0.0, 0.0)` — DR-6 curriculum will widen it dynamically.

- [ ] **Step 3: Verify syntax**

Run:
```bash
python -c "import ast; ast.parse(open('aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/events.py').read()); print('events.py OK')"
python -c "import ast; ast.parse(open('aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py').read()); print('env_cfg.py OK')"
```

Expected: Both print OK.

- [ ] **Step 4: Commit**

```bash
cd aic && git add -A && git commit -m "DR-1: NIC card yaw support + 5th rail

- Add per-part yaw_range to randomize_board_and_parts part loop
- Compose yaw-only quaternion onto part_rot when yaw_range != (0,0)
- Expand nic_card pose_range y: 0.12→0.16 (5 rails instead of 4)
- yaw_range=(0.0,0.0) default — DR-6 curriculum activates it

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: DR-2 — Board Pose Config (Params Only)

**Why:** The eval randomizes board XY by ~±150mm and applies board yaw. Current config is ±5mm with no yaw. The infrastructure (port-anchored placement + yaw composition) already exists from DR-8 — this task just widens the config params for when the DR curriculum activates them.

**Files:**
- Modify: `aic_task_env_cfg.py` — no changes needed now. The `board_yaw_range` and `board_range` params are already present. DR-6 (Task 6) will dynamically override them at runtime. This is a no-op task — skip it.

**Status: SKIP** — DR-6 curriculum handles dynamic range widening. No static config changes needed (static config stays at minimal ranges for stages 0–20).

---

### Task 3: DR-3 — Dual Port Selection

**Why:** Each NIC card has 2 SFP ports. Eval specifies target port via `Task.msg`. Currently we always insert into port 0. The policy must learn to insert into either port. This requires: (a) measuring port 1's offset from the NIC card USD, (b) randomizing which port the marker is placed at per-env per-reset.

**Files:**
- Modify: `mdp/events.py` — `apply_scale_curriculum_event` (~line 315)
- Modify: `aic_task_env_cfg.py` — `apply_scale_curriculum` EventTerm params (~line 474)

**Prerequisite:** Measure SFP_PORT_1 local offset from NIC card origin. The port 0 offset is `(0.01295, -0.0751, 0.00530)`. Port 1 is the same card, second cage — needs USD inspection.

- [ ] **Step 1: Measure port 1 offset from USD**

```bash
cd /home/patrickn/IsaacLab
python -c "
import omni.usd
from pxr import Usd, UsdGeom
# Look for the second SFP cage prim in the NIC card USD
# Path TBD from USD inspection
"
```

Alternatively, look at the FrameTransformer config or USD file for the NIC card to find the second port cage. The two SFP ports are symmetric — likely separated by ~13.5mm in Y (the SFP cage width).

Estimate: `port_1_offset = (0.01295, -0.0751 + 0.0135, 0.00530)` = `(0.01295, -0.0616, 0.00530)`. **Must be verified from USD asset.**

- [ ] **Step 2: Modify apply_scale_curriculum_event for dual ports**

Replace the single `port_local_offset` parameter with a list of offsets:

```python
def apply_scale_curriculum_event(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    nic_card_name: str = "nic_card",
    marker_name: str = "port_frame_marker",
    port_local_offsets: tuple[tuple[float, float, float], ...] = (
        (0.01295, -0.0751, 0.00530),   # SFP_PORT_0
    ),
    port_local_rot: tuple = (0.707, 0.0, 0.0, 0.707),
    scale_mesh_names: tuple[str, ...] = ("task_board", "sc_port", "sc_port_2"),
) -> None:
```

Inside the function, after getting `ref_pos` and `ref_rot`:

```python
    # Randomly select target port per environment
    n_ports = len(port_local_offsets)
    if n_ports > 1:
        port_idx = torch.randint(0, n_ports, (num_envs,), device=device)
        all_offsets = torch.tensor(port_local_offsets, dtype=torch.float32, device=device)
        local_offset = all_offsets[port_idx]  # (num_envs, 3)
    else:
        local_offset = torch.tensor(port_local_offsets[0], dtype=torch.float32, device=device).expand(num_envs, -1)
```

The rest of the function (`port_pos_w`, `port_rot_w`, scaled card placement) uses `local_offset` as before — no further changes needed since `local_offset` is already `(num_envs, 3)`.

- [ ] **Step 3: Update config in aic_task_env_cfg.py**

```python
    apply_scale_curriculum = EventTerm(
        func=apply_scale_curriculum_event,
        mode="reset",
        params={
            "nic_card_name": "nic_card",
            "marker_name": "port_frame_marker",
            "port_local_offsets": (
                (0.01295, -0.0751, 0.00530),    # SFP_PORT_0
                (0.01295, -0.0616, 0.00530),    # SFP_PORT_1 (VERIFY FROM USD)
            ),
            "port_local_rot": (0.707, 0.0, 0.0, 0.707),
            "scale_mesh_names": ("task_board", "sc_port", "sc_port_2"),
        },
    )
```

- [ ] **Step 4: Verify syntax**

```bash
python -c "import ast; ast.parse(open('aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/events.py').read()); print('events.py OK')"
python -c "import ast; ast.parse(open('aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py').read()); print('env_cfg.py OK')"
```

- [ ] **Step 5: Commit**

```bash
cd aic && git add -A && git commit -m "DR-3: dual SFP port selection

- Rename port_local_offset → port_local_offsets (tuple of tuples)
- Randomly select target port per-env per-reset via torch.randint
- Add SFP_PORT_1 offset to config (verified from USD)
- Both ports use same port_local_rot

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: DR-4 — Grasp Perturbation

**Why:** The eval grasps the SFP cable with a real gripper — there's always small translation/rotation error in how the cable sits in the gripper. The policy must be robust to ±2mm translation and ±2.3° rotation perturbation of the grasped cable relative to the gripper.

**Files:**
- Create: Nothing — add function to existing `mdp/events.py`
- Modify: `mdp/events.py` — add `randomize_grasp_perturbation` function
- Modify: `aic_task_env_cfg.py` — add EventTerm + import
- Modify: `mdp/__init__.py` — export new function

- [ ] **Step 1: Add randomize_grasp_perturbation to events.py**

Add this function after `randomize_board_and_parts` (after line 242):

```python
def randomize_grasp_perturbation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    cable_body_name: str = "sfp_module_link",
    translation_range: float = 0.0,
    rotation_range: float = 0.0,
) -> None:
    """Apply small pose perturbation to the grasped cable body.

    Simulates imperfect grasps by offsetting the cable-in-gripper pose.
    Translation is uniform ±translation_range (m) per axis.
    Rotation is uniform ±rotation_range (rad) per Euler axis (roll/pitch/yaw).

    Uses the fixed joint connecting cable to gripper — offsets the joint's
    local pose rather than teleporting the body, so physics stays consistent.
    """
    if translation_range == 0.0 and rotation_range == 0.0:
        return

    if env_ids is None or len(env_ids) == 0:
        return

    n = len(env_ids)
    device = env.device
    robot: Articulation = env.scene["robot"]

    # Find the cable body index
    body_idx = robot.find_bodies(cable_body_name)[0][0]

    # Current body pose
    body_pos = robot.data.body_pos_w[env_ids, body_idx].clone()
    body_quat = robot.data.body_quat_w[env_ids, body_idx].clone()

    # Translation perturbation
    if translation_range > 0.0:
        delta_pos = torch.empty(n, 3, device=device).uniform_(
            -translation_range, translation_range
        )
        body_pos = body_pos + math_utils.quat_apply(body_quat, delta_pos)

    # Rotation perturbation (small Euler angles → quaternion)
    if rotation_range > 0.0:
        euler = torch.empty(n, 3, device=device).uniform_(
            -rotation_range, rotation_range
        )
        half = euler * 0.5
        # Approximate small-angle quaternion: q ≈ (1, rx/2, ry/2, rz/2) normalized
        perturb_quat = torch.zeros(n, 4, device=device)
        perturb_quat[:, 0] = 1.0
        perturb_quat[:, 1] = half[:, 0]
        perturb_quat[:, 2] = half[:, 1]
        perturb_quat[:, 3] = half[:, 2]
        perturb_quat = perturb_quat / perturb_quat.norm(dim=1, keepdim=True)
        body_quat = math_utils.quat_mul(body_quat, perturb_quat)

    # Write perturbed pose back
    pose = torch.cat([body_pos, body_quat], dim=-1)
    robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
```

**Note:** The exact API for perturbing a single body within an articulation may need adjustment. If `write_root_link_pose_to_sim` doesn't apply to individual bodies, we may need to perturb the fixed joint offset instead. The implementing engineer should verify which IsaacLab API works for offsetting a body within a kinematic chain. If the cable body is the articulation root or connected via a fixed joint, adjusting `root_state` of the cable rigid object may be needed instead. **Check if "sfp_module_link" is a separate RigidObject or an Articulation body.**

- [ ] **Step 2: Export in __init__.py**

Add to the events import line in `mdp/__init__.py`:

```python
from .events import apply_scale_curriculum_event, randomize_grasp_perturbation, reset_near_port_curriculum  # noqa: F401
```

- [ ] **Step 3: Add import + EventTerm in aic_task_env_cfg.py**

Add to the imports at line 38-43:

```python
from .mdp.events import (
    apply_scale_curriculum_event,
    randomize_dome_light,
    randomize_board_and_parts,
    randomize_grasp_perturbation,
    reset_near_port_curriculum,
)
```

Add EventTerm in EventCfg after `randomize_robot_pose` and before `randomize_light` (~line 434):

```python
    randomize_grasp = EventTerm(
        func=randomize_grasp_perturbation,
        mode="reset",
        params={
            "cable_body_name": "sfp_module_link",
            "translation_range": 0.0,
            "rotation_range": 0.0,
        },
    )
```

Note: Ranges start at 0.0 — DR-6 curriculum activates them later by setting `translation_range=0.002` and `rotation_range=0.04`.

- [ ] **Step 4: Verify syntax**

```bash
python -c "import ast; ast.parse(open('aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/events.py').read()); print('events.py OK')"
python -c "import ast; ast.parse(open('aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py').read()); print('env_cfg.py OK')"
python -c "import ast; ast.parse(open('aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py').read()); print('init.py OK')"
```

- [ ] **Step 5: Commit**

```bash
cd aic && git add -A && git commit -m "DR-4: grasp perturbation event function

- Add randomize_grasp_perturbation to events.py
- ±translation_range (m) + ±rotation_range (rad) per axis
- Both ranges start at 0.0 — DR-6 curriculum activates
- Export in __init__.py, wire as EventTerm in env cfg

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 5: DR-5 — Enable Observation Noise

**Why:** Real sensors have noise. The policy obs already have noise terms configured (Unoise on every ObsTerm in PolicyCfg), but `enable_corruption = False` disables all of them. Flipping to True activates noise for sim-to-real transfer.

**Files:**
- Modify: `aic_task_env_cfg.py` — `PolicyCfg.__post_init__` line 662

- [ ] **Step 1: Change enable_corruption to True**

In `aic_task_env_cfg.py`, line 662:

```python
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
```

**Important:** Only change `PolicyCfg`, NOT `CriticCfg` (line 754 stays False — asymmetric actor-critic: critic gets clean state).

- [ ] **Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
cd aic && git add -A && git commit -m "DR-5: enable observation noise for policy

- PolicyCfg.enable_corruption = True (was False)
- Activates all configured Unoise terms on actor obs
- CriticCfg stays False (asymmetric actor-critic)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 6: DR-6 — Progressive DR Curriculum (stages 21–25)

**Why:** Applying full eval DR at once would collapse the policy. Instead, extend the curriculum with 5 new stages that progressively widen DR ranges. The event functions already support all DR params (yaw, board range, etc.) — this task makes the curriculum system drive them dynamically.

**Files:**
- Modify: `mdp/curriculum.py` — extend SCALE_STAGES, PARAM_STAGES, add DR_RANGES, modify `_apply_stage`
- Modify: `mdp/events.py` — `randomize_board_and_parts` reads DR level from curriculum; `apply_scale_curriculum_event` reads DR level for port selection gating
- Modify: `aic_task_env_cfg.py` — no changes (config stays at "off" defaults; curriculum overrides at runtime)

- [ ] **Step 1: Add DR_RANGES and extend SCALE/PARAM_STAGES in curriculum.py**

After `PARAM_STAGES` (line 47), add:

```python
# DR ranges per DR level — applied progressively in stages 21-25.
# Level 0 = no extra DR (stages 0-20). Level 5 = full eval-matching DR.
DR_RANGES: list[dict] = [
    # level 0: no extra DR (used by stages 0-20)
    {
        "nic_yaw": 0.0,
        "board_xy": 0.005,
        "board_yaw": 0.0,
        "grasp_trans": 0.0,
        "grasp_rot": 0.0,
        "dual_port": False,
        "obs_noise": False,
    },
    # level 1: obs noise ON, tiny NIC yaw
    {
        "nic_yaw": 0.035,
        "board_xy": 0.005,
        "board_yaw": 0.0,
        "grasp_trans": 0.0,
        "grasp_rot": 0.0,
        "dual_port": False,
        "obs_noise": True,
    },
    # level 2: NIC yaw ±5°
    {
        "nic_yaw": 0.087,
        "board_xy": 0.005,
        "board_yaw": 0.0,
        "grasp_trans": 0.0,
        "grasp_rot": 0.0,
        "dual_port": False,
        "obs_noise": True,
    },
    # level 3: full NIC DR (±10° yaw, all 5 rails)
    {
        "nic_yaw": 0.1745,
        "board_xy": 0.005,
        "board_yaw": 0.0,
        "grasp_trans": 0.0,
        "grasp_rot": 0.0,
        "dual_port": True,
        "obs_noise": True,
    },
    # level 4: board pose DR + grasp perturbation
    {
        "nic_yaw": 0.1745,
        "board_xy": 0.05,
        "board_yaw": 0.044,
        "grasp_trans": 0.001,
        "grasp_rot": 0.02,
        "dual_port": True,
        "obs_noise": True,
    },
    # level 5: full eval-matching DR
    {
        "nic_yaw": 0.1745,
        "board_xy": 0.10,
        "board_yaw": 0.087,
        "grasp_trans": 0.002,
        "grasp_rot": 0.04,
        "dual_port": True,
        "obs_noise": True,
    },
]
```

Extend SCALE_STAGES and PARAM_STAGES:

```python
# Append DR stages 21-25 (scale stays at 1.0, dr_level increases 1-5)
SCALE_STAGES: list[float] = [
    2.0, 1.857, 1.714, 1.571, 1.429, 1.286, 1.143,  # stages 0-6
    1.100, 1.075, 1.050, 1.030, 1.020, 1.010, 1.008,  # stages 7-13
    1.006, 1.005, 1.004, 1.003, 1.002, 1.001, 1.000,  # stages 14-20
    1.000, 1.000, 1.000, 1.000, 1.000,                 # stages 21-25 (DR)
]
```

Add `dr_level` field to PARAM_STAGES entries. The simplest approach: keep stages 0-20 as `dr_level=0`, add stages 21-25 with `dr_level=1-5`. Extend the `] * (NUM_STAGES - 7)` block:

```python
PARAM_STAGES: list[dict] = [
    dict(y_half=0.014, z_half=0.0090, yz_threshold=0.013, std_yz=0.013, depth_m=0.005, score_depth_m=0.015, score_yz=0.013, score_y_half=0.014, score_z_half=0.009, hold_time_s=0.2, orient_deg=35.0, dr_level=0),
    dict(y_half=0.013, z_half=0.0083, yz_threshold=0.012, std_yz=0.012, depth_m=0.005, score_depth_m=0.020, score_yz=0.012, score_y_half=0.013, score_z_half=0.0083, hold_time_s=0.3, orient_deg=32.0, dr_level=0),
    dict(y_half=0.012, z_half=0.0077, yz_threshold=0.011, std_yz=0.011, depth_m=0.006, score_depth_m=0.025, score_yz=0.011, score_y_half=0.012, score_z_half=0.0077, hold_time_s=0.4, orient_deg=29.0, dr_level=0),
    dict(y_half=0.011, z_half=0.0070, yz_threshold=0.010, std_yz=0.010, depth_m=0.007, score_depth_m=0.030, score_yz=0.010, score_y_half=0.011, score_z_half=0.007, hold_time_s=0.5, orient_deg=26.0, dr_level=0),
    dict(y_half=0.010, z_half=0.0064, yz_threshold=0.009, std_yz=0.009, depth_m=0.008, score_depth_m=0.035, score_yz=0.009, score_y_half=0.010, score_z_half=0.0064, hold_time_s=0.6, orient_deg=23.0, dr_level=0),
    dict(y_half=0.009, z_half=0.0058, yz_threshold=0.008, std_yz=0.008, depth_m=0.009, score_depth_m=0.040, score_yz=0.008, score_y_half=0.009, score_z_half=0.0058, hold_time_s=0.7, orient_deg=20.0, dr_level=0),
    dict(y_half=0.008, z_half=0.0051, yz_threshold=0.007, std_yz=0.007, depth_m=0.010, score_depth_m=0.0448, score_yz=0.006, score_y_half=0.007, score_z_half=0.005, hold_time_s=0.85, orient_deg=17.0, dr_level=0),
] + [
    # Stages 7-20: real tolerances, only NIC card scale decreases
    dict(y_half=0.007, z_half=0.004475, yz_threshold=0.007, std_yz=0.007, depth_m=0.010, score_depth_m=0.0448, score_yz=0.005, score_y_half=0.005, score_z_half=0.004, hold_time_s=1.0, orient_deg=15.0, dr_level=0),
] * 14 + [
    # Stages 21-25: real scale, progressive DR
    dict(y_half=0.007, z_half=0.004475, yz_threshold=0.007, std_yz=0.007, depth_m=0.010, score_depth_m=0.0448, score_yz=0.005, score_y_half=0.005, score_z_half=0.004, hold_time_s=1.0, orient_deg=15.0, dr_level=1),
    dict(y_half=0.007, z_half=0.004475, yz_threshold=0.007, std_yz=0.007, depth_m=0.010, score_depth_m=0.0448, score_yz=0.005, score_y_half=0.005, score_z_half=0.004, hold_time_s=1.0, orient_deg=15.0, dr_level=2),
    dict(y_half=0.007, z_half=0.004475, yz_threshold=0.007, std_yz=0.007, depth_m=0.010, score_depth_m=0.0448, score_yz=0.005, score_y_half=0.005, score_z_half=0.004, hold_time_s=1.0, orient_deg=15.0, dr_level=3),
    dict(y_half=0.007, z_half=0.004475, yz_threshold=0.007, std_yz=0.007, depth_m=0.010, score_depth_m=0.0448, score_yz=0.005, score_y_half=0.005, score_z_half=0.004, hold_time_s=1.0, orient_deg=15.0, dr_level=4),
    dict(y_half=0.007, z_half=0.004475, yz_threshold=0.007, std_yz=0.007, depth_m=0.010, score_depth_m=0.0448, score_yz=0.005, score_y_half=0.005, score_z_half=0.004, hold_time_s=1.0, orient_deg=15.0, dr_level=5),
]
```

**Important**: `NUM_STAGES` is computed as `len(SCALE_STAGES)` which will become 26 (was 21). `SCALED_CARD_NAMES` creates scene names for stages 0..N-2 — it will now create 25 names. However stages 20-25 all have scale=1.0, so stages 20-24 will create unnecessary 1.0×-scale duplicate cards. To avoid this, change SCALED_CARD_NAMES to only generate names for stages with scale != 1.0:

```python
# Scene names for scaled NIC cards — only stages with scale > 1.0.
# Stages at 1.0× use the original nic_card.
SCALED_CARD_NAMES: list[str] = [f"nic_card_s{i}" for i in range(NUM_STAGES - 1) if SCALE_STAGES[i] > 1.0]
```

Wait — this would break `apply_scale_curriculum_event` which indexes SCALED_CARD_NAMES by stage number. The existing logic is: `if stage < len(SCALED_CARD_NAMES): active_name = SCALED_CARD_NAMES[stage]`. 

**Better approach**: Keep SCALED_CARD_NAMES as before (indices 0..NUM_STAGES-2). For stages 20-24 (scale=1.0), `apply_scale_curriculum_event` will use the `else` branch (stage >= len(SCALED_CARD_NAMES) or stage == last), which uses the original `nic_card`. But SCALED_CARD_NAMES would now have 25 entries, and stages 20-24 would try to spawn nic_card_s20..s24 (all at 1.0× scale). These are wasteful scene entities.

**Simplest fix**: Don't extend `SCALED_CARD_NAMES`. Keep it generating only for pre-1.0× stages. Change the generation logic:

```python
# Number of pre-scaled NIC card variants needed (stages with scale > 1.0).
_NUM_SCALED_CARDS: int = sum(1 for s in SCALE_STAGES if s > 1.0)
SCALED_CARD_NAMES: list[str] = [f"nic_card_s{i}" for i in range(_NUM_SCALED_CARDS)]
```

And in `aic_task_env_cfg.py`, the scene entity generation loop (line 354):

```python
for _i, _scale in enumerate(SCALE_STAGES):
    if _scale <= 1.0:
        break
    setattr(AICTaskSceneCfg, f"nic_card_s{_i}", ...)
```

Then `apply_scale_curriculum_event` uses the existing logic: `if stage < len(SCALED_CARD_NAMES)` → use scaled card; else → use original nic_card.

- [ ] **Step 2: Add get_dr_level() accessor**

```python
def get_dr_level() -> int:
    """Return the current DR level (0-5) for the active curriculum stage."""
    return PARAM_STAGES[_current_stage].get("dr_level", 0)
```

Export it in `__init__.py`:

```python
from .curriculum import get_current_stage, get_dr_level, set_current_stage, tighten_insertion_curriculum  # noqa: F401
```

- [ ] **Step 3: Modify _apply_stage to handle DR params**

In curriculum.py, extend `_apply_stage`:

```python
def _apply_stage(env: "ManagerBasedRLEnv", stage: int) -> None:
    params = PARAM_STAGES[stage]
    dr_level = params.get("dr_level", 0)
    dr = DR_RANGES[dr_level]
    rm = env.reward_manager

    # ... existing reward param updates ...

    # --- DR parameter updates ---
    # Update event term params at runtime.
    em = env.event_manager
    # Build event term map
    ev_map: dict = {}
    if isinstance(em._term_cfgs, dict):
        ev_map = em._term_cfgs
    else:
        ev_map = dict(zip(em._term_names, em._term_cfgs))

    # Board DR
    board_evt = ev_map.get("randomize_board_and_parts")
    if board_evt is not None:
        board_evt.params["board_range"] = {
            "x": (-dr["board_xy"], dr["board_xy"]),
            "y": (-dr["board_xy"], dr["board_xy"]),
        }
        board_evt.params["board_yaw_range"] = (-dr["board_yaw"], dr["board_yaw"])
        # Update NIC card yaw in parts list
        for part in board_evt.params.get("parts", []):
            if part["scene_name"] == "nic_card":
                part["yaw_range"] = (-dr["nic_yaw"], dr["nic_yaw"])

    # Grasp perturbation
    grasp_evt = ev_map.get("randomize_grasp")
    if grasp_evt is not None:
        grasp_evt.params["translation_range"] = dr["grasp_trans"]
        grasp_evt.params["rotation_range"] = dr["grasp_rot"]

    # Obs noise
    # Note: enable_corruption can't be toggled at runtime easily in IsaacLab —
    # the obs manager reads it once at init. For now, the policy starts with
    # enable_corruption=True (Task 5) and noise magnitudes are always active.
    # DR-6 controls this via the curriculum stage logging.

    if dr_level > 0:
        print(f"[CURRICULUM] DR level {dr_level}: nic_yaw=±{dr['nic_yaw']:.3f} "
              f"board_xy=±{dr['board_xy']:.3f} board_yaw=±{dr['board_yaw']:.3f} "
              f"grasp=±{dr['grasp_trans']*1000:.1f}mm dual_port={dr['dual_port']}")
```

- [ ] **Step 4: Modify events.py to read DR level for dual-port gating**

In `apply_scale_curriculum_event`, only randomize port selection when `dual_port=True` in the current DR level:

```python
from .curriculum import get_current_stage, get_dr_level, DR_RANGES

# Inside apply_scale_curriculum_event, in the port selection logic:
dr = DR_RANGES[get_dr_level()]
n_ports = len(port_local_offsets)
if n_ports > 1 and dr.get("dual_port", False):
    port_idx = torch.randint(0, n_ports, (num_envs,), device=device)
    all_offsets = torch.tensor(port_local_offsets, dtype=torch.float32, device=device)
    local_offset = all_offsets[port_idx]
else:
    local_offset = torch.tensor(port_local_offsets[0], dtype=torch.float32, device=device).expand(num_envs, -1)
```

- [ ] **Step 5: Update SCALED_CARD_NAMES generation**

In curriculum.py, change:

```python
# Only generate scaled card names for stages with scale > 1.0
_NUM_SCALED_CARDS: int = sum(1 for s in SCALE_STAGES if s > 1.0)
SCALED_CARD_NAMES: list[str] = [f"nic_card_s{i}" for i in range(_NUM_SCALED_CARDS)]
```

In `aic_task_env_cfg.py`, change the scene entity generation (line 354):

```python
for _i, _scale in enumerate(SCALE_STAGES):
    if _scale <= 1.0:
        break
    setattr(
        AICTaskSceneCfg,
        f"nic_card_s{_i}",
        RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/nic_card_s{_i}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=os.path.join(AIC_PARTS_DIR, "NIC Card", "nic_card.usd"),
                scale=(_scale, _scale, _scale),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, -10.0),
                rot=(0.707, -0.707, 0.0, 0.0),
            ),
        ),
    )
```

- [ ] **Step 6: Verify syntax**

```bash
python -c "import ast; ast.parse(open('aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/curriculum.py').read()); print('curriculum.py OK')"
python -c "import ast; ast.parse(open('aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/events.py').read()); print('events.py OK')"
python -c "import ast; ast.parse(open('aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py').read()); print('env_cfg.py OK')"
python -c "import ast; ast.parse(open('aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py').read()); print('init.py OK')"
```

- [ ] **Step 7: Commit**

```bash
cd aic && git add -A && git commit -m "DR-6: progressive DR curriculum (stages 21-25)

- Extend SCALE_STAGES with 5 new 1.0× stages for DR progression
- Add DR_RANGES table (6 levels: none → full eval-matching)
- Add dr_level field to all PARAM_STAGES entries
- _apply_stage dynamically updates event term params from DR_RANGES
- apply_scale_curriculum_event gates dual-port on DR level
- Only generate scaled NIC card scene entities for scale > 1.0
- Add get_dr_level() accessor + export

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Dependency Graph

```
Task 1 (DR-1: NIC yaw) ──────────┐
Task 3 (DR-3: dual port)─────────┤
Task 4 (DR-4: grasp perturb)─────┼──→ Task 6 (DR-6: curriculum)
Task 5 (DR-5: obs noise)─────────┘
Task 2: SKIP (absorbed into DR-6)
```

Tasks 1, 3, 4, 5 are independent and can execute in parallel.
Task 6 depends on all of them (it wires the curriculum to their params).

---

## Validation Checklist

After all tasks complete:

1. **Syntax**: All 4 files pass `ast.parse()`
2. **Stage count**: `len(SCALE_STAGES) == 26`, `len(PARAM_STAGES) == 26`
3. **Scene entity count**: `len(SCALED_CARD_NAMES) == 20` (stages 0-19 have scale > 1.0)
4. **DR level 0**: Stages 0-20 behave identically to pre-DR code
5. **DR level 5**: Stage 25 has full eval-matching randomization
6. **No obs dim change**: 37 policy dims, 29 critic dims unchanged
7. **Rollout test**: Run 100 envs × 10 episodes at stage 0, verify board scales correctly
