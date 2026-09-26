METHODOLOGY IMPLEMENTATION COMPLIANCE (LiDAR.JPG reference)
============================================================
Date: 2026-09-20. Format: REFERENCE STAGE / IMPLEMENTED MODULE / INPUT /
OUTPUT / STATUS / EVIDENCE. Measured values only; blocked stages name
the missing dependency.

1. Data Acquisition
   Module: src/lidar_loader.py + backend replay_service (Mode A);
     src/simulation/carla_lidar.py (Mode B interface, blocked).
   Input: nuScenes Mini LIDAR_TOP .bin (Mode A).
   Output: N x 4 float64 [x, y, z, intensity] (verified 34,359 pts/frame).
   Status: PARTIAL (replay LIVE; CARLA BLOCKED -- package not installed).
   Evidence: GET /frames; docs/carla_setup.md.

2. Perception
   Module: src/semantic_model.py (mlp-point-segmenter-v1, point-wise
     shared MLP; 5 project classes; display taxonomy Road/Vehicle/
     Pedestrian/Static Obstacle/Others).
   Input: N x 4 point cloud. Output: labels + softmax confidence +
     source "model" (533/533 cells, all finite confidence, live frame).
   Status: PASS (PointNet++/sparse-CNN explicitly NOT claimed: no
     torch/GPU runtime on this CPU-only offline host).
   Evidence: GET /model/info; POST /model/predict/{frame_id};
     results/segmentation/metrics.json (acc 0.9114, mIoU 0.4104).

3. Scene Analysis
   Module: src/scene_analysis.py (config/scene_analysis_config.json).
   Input: region detail rows. Output: distance, point density, object
     density proxy, object importance, elevation variation, prediction
     uncertainty (+ provenance each).
   Status: PASS. Evidence: backend/tests/test_flow.py.

4. Adaptive Resolution Engine
   Module: src/resolution_engine.py (frozen 0.05/0.10/0.20/0.50 m) +
     src/quadtree.py (importance-driven subdivision; HIGH/MEDIUM/LOW bands).
   Input: importance + frozen thresholds. Output: per-region resolution;
     live quadtree 228 nodes / 158 leaves / depth 4 (b26e7915).
   Status: PASS. Evidence: GET /map/quadtree/{frame_id}.

5. 2.5D Semantic Map
   Module: src/mapper_2_5d.py + src/map_validator.py (RL gate).
   Input: current LiDAR + semantics + scene factors + resolutions.
   Output: cells with Z/class/occupancy/confidence/resolution
     (test-verified); corrupt maps raise InvalidMapError before RL use.
   Status: PASS. Evidence: test_map_cells_carry_diagram_fields.

6. RL State Generation
   Module: src/rl_state.py (config/rl_state_config.json; frontend cannot
     modify the definition).
   Input: adaptive map cells. Output: documented 13-dim vector
     (8 sectors + obstacle/moving/static shares + importance +
     uncertainty) + safety block.
   Status: PASS. Evidence: GET /rl/decide/{frame_id}.

7. RL Decision Making / DQN
   Module: src/rl_agent.py (NumPy DQN 13->64->4, 1,156 params;
     config/dqn_config.json); env interface src/rl/carla_env.py.
   Input: 13-dim state. Output: genuine Q-values + argmax action from
     {forward, turn_left, turn_right, stop}; trained=False always.
   Status: PARTIAL (implemented, untrained; training BLOCKED, no episodes).
   Evidence: GET /rl/status; notebooks/11 (refusal path verified).

8. Action Execution
   Module: src/safety_controller.py (SAFE_TO_EXECUTE/OVERRIDE_TO_STOP) +
     src/action_executor.py (Simulation default; Hardware disabled).
   Input: DQN action + obstacle distance + map validity.
   Output: validated final action; close-obstacle FORWARD becomes STOP;
     no hardware command ever issued.
   Status: PARTIAL (safety LIVE; vehicle absent).
   Evidence: tests/robotics/test_safety_scenarios.py (12 passed).

9. Real-World Testing
   Module: tests/robotics/ framework (scenarios over replay/synthetic).
   Input: replay frames + synthetic layouts. Output: scenario verdicts.
   Status: BLOCKED -- REAL_WORLD_EXECUTION = NOT_AVAILABLE (no hardware);
     simulation/replay validation substituted, never presented as physical.
   Evidence: tests/robotics/__init__.py.

CARLA Training Environment
   Module: src/simulation/carla_lidar.py + src/rl/carla_env.py +
     notebooks/12 (blocked path executed as status report).
   Status: BLOCKED (not installed; setup in docs/carla_setup.md).

Reward System
   Module: config/reward_config.json (progress +1.0, clearance +0.5,
     smoothness +0.2, collision -10, unnecessary stop -1.0).
   Status: SPECIFIED, NEVER EXECUTED (no episodes; weights configurable,
     not tuned optima).

Safety Layer
   Module: src/safety_controller.py, DQN-independent, always overrides.
   Status: PASS (rule live; override tested incl. end-to-end on replay).
