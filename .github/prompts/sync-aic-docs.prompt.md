---
description: "Regenerate the baked-in knowledge tables in AIC agent files and instructions from the current codebase. Run after significant reward, observation, or env config changes."
---

Read the current state of these files:
- `aic/.../aic_task/aic_task_env_cfg.py` (RewardsCfg, ObservationsCfg)
- `aic/.../aic_task/mdp/rewards.py` (all reward function signatures)
- `aic/.../aic_task/mdp/observations.py` (all observation function signatures)
- `aic/.../aic_task/mdp/__init__.py` (exports)

Then update the baked-in knowledge sections in these files to match reality:
1. `.github/agents/reward-engineer.agent.md` — "Current Reward Architecture" table
2. `.github/agents/observation-engineer.agent.md` — "Current Observation Architecture" table
3. `.github/agents/geometry-engineer.agent.md` — "AIC Task Geometry" and "Key Geometric Relationships" sections
4. `.github/agents/aic-research-manager.agent.md` — task context if needed
5. `.github/instructions/aic-task.instructions.md` — Reward Design Patterns, Observation Space table

For each file, diff the current baked-in tables against the actual code. Only update what's changed. Preserve all non-table content (approach guidelines, constraints, output format).

Report a summary of what changed.
