# Dataset and semantic limitations (15 September final validation)

## Dataset
- nuScenes Mini (v1.0-mini), 7-frame final evaluation subset
  (see `results/final/final_test_manifest.csv`).
- Frames, scenes, preprocessing, configuration, seed (42) and evaluation
  procedure are identical to the 14 September validation.

## Semantic source
- Actual source: `annotation` (nuScenes object boxes projected to the
  LiDAR sensor frame) + `fallback` (points outside any box -> `unknown`).
- No point-level lidarseg used in this path. No trained perception model.

## Model predictions
- UNAVAILABLE. There is no trained neural network in this prototype, so
  there are no model predictions. The `source_separation` record shows
  `n_model_regions == 0` for every run.

## Ground truth
- Object-level annotation boxes used as reference only
  (representation-preservation check, NOT detection-accuracy scoring).

## Confidence
- UNAVAILABLE for annotation/fallback sources: NaN at point level
  (documented "not applicable"), numeric 0.0 interface placeholder at
  region level (documented, never presented as a measurement).

## Classes
- Project taxonomy: vehicle, pedestrian_vru, static_manmade, vegetation,
  road_driveable, unknown.

## Known dataset limitations
- 7-frame subset of v1.0-mini; not a full-dataset evaluation.
- Dropout / sparsity / XYZ-noise rows are controlled, seeded test
  conditions — not claims about true sensor error statistics.

## Controlled robustness tests
- dropout {5,10,20,30}%, sparsity keep {75,50,25}%, XYZ Gaussian noise
  sigma {0.01,0.03,0.05} m, seed 42. Final regression subset re-run on the
  frozen pipeline: 28/28 PASS (`results/final/metrics/robustness_regression.csv`).

## Hardware / runtime
- Local CPU, compute-only `time.perf_counter` timing; visualization excluded.
- FPS values are measured throughput, not real-time claims.
- torch / open3d are not installed and are not required by the pipeline.
