# Pre-RL/CARLA Baseline — Stages 1–6 frozen status

Date: RL/CARLA work session. Backup: `Paradox_Protocol_pre_RL_CARLA.zip`
(401 files, 11.1 MB, code+configs+results, no data/env).
Git checkpoint: `2fc2b17` (pathspec-limited to the project dir; home repo root).

## Stages 1–6 status (all PASS, verified by execution where stated)

| Stage | Module(s) | Status | Evidence |
|---|---|---|---|
| 1 Data Acquisition | `src/data_types.LiDARFrame`, `lidar_loader`, `replay_service` | PASS (frozen) | `stage_validation/stage1_validation.csv`: 3/3 frames, Nx4 float64, all finite |
| 2 Perception | `src/semantic_model` (MLP, real weights) + annotation fallback | PASS (frozen) | `stage2_validation.csv`: annotation+fallback sources per cell; model-mode softmax finite, 136 ms |
| 3 Scene Analysis | `src/scene_analysis.analyze_frame` (6 factors + provenance) | PASS (frozen) | `stage3_validation.csv`: region fields present; single importance algorithm |
| 4 Adaptive Resolution | `src/resolution_engine` (THRESH_C) | PASS (frozen) | `results/final/config/final_config.json` W_BASE+THRESH_C |
| 5 2.5D Map | `src/mapper_2_5d` (2 m cells, x/y/elev/occ/sem/conf/imp/res) | PASS (frozen) | 7/7 live runs byte-identical to 16 Sept |
| 6 RL State | `src/rl_state.build_state` → 13-dim vector | PASS (frozen) | `stage6_validation.csv`: dim 13, finite, [0,1], build ~ms |

## Interfaces (frozen)

- `LiDARFrame(frame_id, timestamp, points[Nx4])` (+ optional `scene_id`, `source`, default `"nuscenes_replay"`) → `PerceptionResult` → `RegionFeatures` → Importance → Resolution → map cells → `build_state(cells)` → 13-vector → DQN `act()` → safety `evaluate()` → executor.
- DQN input dim (13) matches `config/rl_state_config.json` `state_dim` (verified in stage6 CSV + `describe()`).
- Semantic sources observed in real outputs: `annotation`, `fallback`, `unknown`, `model` (MLP softmax). Literal strings `lidarseg_annotation`/`object_annotation` do not occur in code; mapping: `annotation` covers object-annotation references, `lidarseg` vocabulary value covers LiDAR-seg references (see `semantic_mapping.VALID_SEMANTIC_SOURCES`). Separation is preserved; nothing is relabelled as model output.

## Replay pipeline

`data/processed/*.npy` → `replay_service` → `robustness.run_pipeline_condition` (seed 42) → backend `PipelineResult` → React dashboard. No notebook state required.

## Known limitations (carried forward)

- DQN: implemented (NumPy 13→64→64→4, 1156 params), random-init, NOT TRAINED; `train_episode()` refuses without simulator. No checkpoint exists (`models/rl/` absent — correct).
- CARLA: package not installed; all CARLA entry points raise `CarlaUnavailable` with setup pointer. Stage 9 BLOCKED.
- Physical testing: NOT EXECUTED.
- Checkpoint commit includes stray `__pycache__/*.pyc` files (reset missed nested dirs) — harmless, noted.
