---
description: "Use when working on the AIC (AI for Industry Challenge) SFP cable insertion task with IsaacLab, UR5e robot, RL training, MDP terms, reward shaping, or simulation configs under aic/."
applyTo: "aic/**"
---
# AIC Task — IsaacLab RL Environment

## Project Overview

The AIC task trains a UR5e robot arm to insert an SFP (Small Form-factor Pluggable) fiber optical cable into an LC port on a NIC card. The environment is built on **IsaacLab 2.3.0** (Isaac Sim) and uses **RSL-RL** (PPO) for training.

**Official insertion criterion**: cable must exert ≥ 20 N on the port for ≥ 1 consecutive second while geometrically aligned.

## Key Paths

| What | Path (relative to repo root) |
|------|-----|
| Task env config | `aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py` |
| Custom MDP terms | `aic/…/aic_task/mdp/` (observations.py, rewards.py, actions.py, events.py) |
| RSL-RL PPO config | `aic/…/aic_task/agents/rsl_rl_ppo_cfg.py` |
| Training script | `aic/…/aic_isaaclab/scripts/rsl_rl/train.py` |
| USD assets | `aic/…/aic_task/Intrinsic_assets/` |

## Commands

```bash
# Train
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-Task-v2 --num_envs 100 --livestream 2

# Evaluate
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/play.py --task AIC-Task-v2 --num_envs 1 --livestream 2

# TensorBoard
tensorboard --logdir ~/IsaacLab/logs/rsl_rl/aic_task/
```

## Coding Conventions

- Use `@configclass` (from `isaaclab.utils`) for all config dataclasses.
- Follow IsaacLab manager-based env pattern: `SceneCfg`, `ObservationsCfg`, `RewardsCfg`, `ActionsCfg`, `EventCfg`, `TerminationsCfg` as nested `@configclass` definitions.
- MDP term functions go in `mdp/` as standalone functions; wire them via `ObsTerm`, `RewTerm`, `EventTerm`, `DoneTerm` in the env cfg.
- Reference scene entities by name using `SceneEntityCfg("robot", body_names=[...])` or `SceneEntityCfg("sfp_port_frame")`.
- Resolve USD asset paths relative to the Python file: `os.path.join(_THIS_DIR, "Intrinsic_assets", ...)`.
- Use `CABLE_TIP_BODY` / `SFP_PORT_FRAME` constants for the cable tip body and port frame sensor.
- Formatting: **Ruff** (line-length 120), **Black**, **isort** with IsaacLab custom sections.

## Coordinate Frame Convention

The SFP port entrance frame (exposed by `FrameTransformerCfg`):
- **+X** = insertion direction (into port)
- **+Y** = card-plane direction (keying axis)
- **+Z** = card normal

Identity quaternion in this frame = perfectly aligned for insertion.

## Reward Design Patterns

Rewards follow a staged curriculum (see papers: FORGE, IndustReal):

1. **Approach**: L2 distance penalty + dual-logistic proximity (coarse @ 10 cm, fine @ 1 cm)
2. **Alignment**: 3-tier orientation (always-on coarse → proximity-gated fine → roll)
3. **Insertion**: Dense depth-progress + sparse insertion bonus (geometric gates)
4. **Scoring**: AIC official criterion (force + hold time + alignment gates)
5. **Regularisation**: action rate, wrist deviation, contact force penalty, joint torques

When adding or modifying rewards:
- Use `frame_cfg: SceneEntityCfg(SFP_PORT_FRAME)` to reference the port frame — never hardcode world-space targets.
- Gate fine rewards by proximity to avoid rewarding random lucky configurations.
- Include paper references in docstrings/comments for non-trivial reward functions.

## Action Space

6-DOF differential IK in **port entrance frame** via `PortFrameDiffIKAction`:
- Policy outputs `(dx, dy, dz, dax, day, daz)` in port-local coordinates.
- Custom action class rotates deltas into robot base frame each step.
- Scale: 0.01 m/step, 0.01 rad/step.

## Observation Space

| Group | Dims | Noise | Usage |
|-------|------|-------|-------|
| **Policy** | 25 | Mild additive uniform | Actor (deployed) |
| **Critic** | 25+ | None (privileged) | Asymmetric critic (sim only) |

Components: joint_pos_rel (6), cable_pos_in_port (3), cable_quat_in_port (4), ee_wrench (6), last_action (6).

## Domain Randomisation

Applied on `mode="reset"`:
- Robot joint offsets (±0.3 rad)
- Robot base pose (±0.2 m XY)
- Dome light intensity (1500–3500) and color
- Task board position (±5 mm) and parts (SC ports ±20 mm, NIC card snap-step)
