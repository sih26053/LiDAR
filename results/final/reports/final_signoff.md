# FINAL MODEL SIGN-OFF

Project:
Adaptive Variable-Resolution 2.5D LiDAR Mapping

Final Version:
prototype-final-v1 (W_BASE+THRESH_C)

Dataset:
nuScenes Mini, 7-frame final subset

Frames Tested:
7 (all PASS)

Final Configuration:
results/final/config/final_config.json

Model Artifact:
No separately trained weights (CASE B: deterministic/annotation-driven prototype)

Core Modules (frozen, unchanged):
src/importance_engine.py, src/resolution_engine.py, src/mapper_2_5d.py,
src/semantic_adapter.py, src/semantic_mapping.py, evaluation via src/robustness.py

Reproducibility:
PASS (10/10 field checks incl. canonical SHA-256, results/final/reproducibility_checks.csv)

End-to-End Validation:
PASS (7/7, results/final/final_reproducibility_results.csv)

Critical-Object Validation:
PASS (496/496 critical regions represented in map;
representation check, NOT detection accuracy)

Final Robustness Regression:
PASS (28/28, results/final/metrics/robustness_regression.csv)

14-Sept vs 15-Sept Regression:
PASS (map-cell counts identical on all 7 frames; results/final/metrics/comparison_14sep.csv)

Independent Handoff Validation:
see results/final/handoff_validation.json

Mean measured performance (CPU compute-only):
latency 1152.5 ms | FPS 0.881 | cells 900.6

Known Limitations:
see results/final/reports/limitations.md
