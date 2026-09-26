METHODOLOGY GAP ANALYSIS (LiDAR.JPG reference)
================================================
Date: 2026-09-20. Audited against the actual repository before changes.

| REFERENCE STAGE | CURRENT IMPLEMENTATION | MISSING COMPONENT | REQUIRED CHANGE | STATUS |
|---|---|---|---|---|
| 1. Data Acquisition (LiDAR / CARLA, N x 4 XYZI) | nuScenes Mini replay via src/lidar_loader.py + backend replay_service; N x 4 float64 verified | CARLA input adapter | new src/simulation/carla_lidar.py (interface + blocked execution) | PARTIAL |
| 2. Perception (PointNet++/Sparse CNN; Road/Vehicle/Pedestrian/Static/Others) | Trained MLP point segmenter src/semantic_model.py (5 project classes, softmax confidence); display taxonomy mapped | PointNet++/sparse-CNN runtime (no torch/GPU); high-level category mapping doc | config mapping + honesty note (no fake PointNet++ claim) | PARTIAL |
| 3. Scene Analysis (6 factors) | src/scene_analysis.py: distance, point density, object density proxy, object importance, elevation variation, prediction uncertainty | per-point object counts (member labels not retained) | proxy documented; no new formula | PASS |
| 4. Adaptive Resolution (High 5cm / Med 20cm / Low 80cm + quadtree) | Frozen 4-tier engine (0.05/0.10/0.20/0.50 m); no quadtree module | src/quadtree.py; HIGH/MED/LOW category exposure | add quadtree subdivision driven by importance; keep validated tiers | PARTIAL |
| 5. 2.5D Semantic Map (Z/class/occupancy/confidence/resolution) | PipelineResult.map_cells carries all five (verified by test) | formal gate before RL use | src/map_validator.py | PASS (gate missing) |
| 6. RL State Generation | src/rl_state.py: genuine 13-dim vector from map cells | RLState structured groups; rl_state_config.json | document + externalize config | PARTIAL |
| 7. DQN Decision (Forward/Left/Right/Stop) | src/rl_agent.py: NumPy DQN, random init, zero episodes | dqn_config.json; training notebooks; results/rl/ | configs + notebooks 11/12 (blocked paths honest) | PARTIAL |
| 8. Action Execution (safety check, motor, toy car) | Safety rule inside rl_state only; no executor modules | src/safety_controller.py; src/action_executor.py (sim/hw, hw disabled) | implement rule layer + abstraction | MISSING |
| 9. Real-World Testing | None (no hardware); replay evaluation only | tests/robotics/ framework; NOT_EXECUTED marking | scenario tests over replay/synthetic inputs | MISSING |
| Training Environment (CARLA) | Not installed (verified: import carla fails; no gymnasium) | src/simulation/carla_lidar.py; src/rl/carla_env.py; reward config; notebooks 11/12 | interfaces + blocked execution + setup docs | MISSING |
| Reward System (+/-) | Specified only in config/rl_reward.json | canonical config/reward_config.json with weights | consolidate + document unexecuted | PARTIAL |
| System outputs | Mapping/memory panels measured; decision/safety only via /rl/decide | dashboard decision row; simulation controller + APIs | SimulationPanel + /simulation/* + /rl/status | PARTIAL |

BLOCKED (environment, not code): CARLA install/connect/LiDAR sim/closed loop;
DQN training (weights, episodes, rewards optimized); real-world execution;
gymnasium interface (implemented without the dependency instead).
