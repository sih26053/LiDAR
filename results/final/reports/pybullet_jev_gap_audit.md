# PHASE 0 — PyBullet + Jev gap audit

Date: 2026-09-21 (UTC). Inspected before edits.

## Reusable unchanged

- Stages 1–6: `lidar_loader`, `preprocessing`, `semantic_model` (MLP),
  `semantic_adapter`, `scene_analysis`, `importance_engine`,
  `resolution_engine`, `mapper_2_5d`, `rl_state` (13-D: idx 0–7 sector
  ranges/30 m, 8 obstacle density, 9 moving share, 10 static share,
  11 mean importance, 12 mean uncertainty; terrain_share reported, not in vector).
- `LiDARFrame` dataclass (`data_types.py`): frame_id/timestamp/points Nx4/scene_id/source.
- Safety: `safety_controller.evaluate()` (SAFE_TO_EXECUTE/OVERRIDE_TO_STOP);
  `action_executor` sim/hardware (hardware disabled).
- Decision baseline: `rl_agent` DQN (untrained, save/load works); Action enum
  lowercase (`forward/turn_left/turn_right/stop`) — canonical, kept.
- Backend routes: replay/stream/tracking/model/flow/simulation; frontend panels.
- CARLA code (`carla_lidar`, `carla_env`, `carla_manager`, `carla_action_executor`,
  `run_carla_closed_loop.py`, `carla_scenarios.json`): legacy, stays optional.

## Missing (to build)

- `src/simulation/simulator.py` abstraction; `pybullet_env.py`;
  `pybullet_lidar.py` (raycast); LiDARFrame adapter; `pybullet_action_executor.py`.
- `src/decision/` package: `policy_interface.py`, `jev_state_adapter.py`,
  `jev_decision.py`; `config/jev_config.json`, `config/simulation_config.json`,
  `config/pybullet_scenarios.json`.
- `scripts/check_pybullet.py`, `scripts/check_jev.py`,
  `scripts/run_pybullet_closed_loop.py`; `results/{pybullet,decisions,metrics}/`.
- Dashboard decision/sim panel + `/decision/*`, `/metrics/current` endpoints.

## Blockers (verified, not assumed)

- PyBullet: `pip install pybullet` FAILS on Python 3.14.4 (source build fails;
  `--only-binary` → "No matching distribution"). No alternate Python installed.
- Jev: `TYPESAFE_API_KEY` NOT SET in environment; no reachable service configured.
- GPU: no `nvidia-smi` interface on this host.
- CARLA: still absent (legacy path unchanged).

## Consequence

All PyBullet/Jev code is written import-safe (lazy imports) and fully tested
where testable without the live deps; live stages stay honestly BLOCKED until
the human actions in `system_status` are completed. Stages 1–6 frozen.
