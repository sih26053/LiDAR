# Pipeline Documentation — Simulation Integration (16 September 2026)

Frozen selection: **W_BASE+THRESH_C** (`results/final/config/final_config.json`, read-only).
Integration only — no weight/threshold retuning, no mapper behavior change.

## Stage table

| Stage | Input | Processing | Output | Source module |
|---|---|---|---|---|
| Load | nuScenes Mini `LIDAR_TOP` `.bin` (5-col float32) | `load_sample_lidar`: keep `[x, y, z, intensity]`, drop `ring_index` | `points` N×4 float64 + `frame_id, sample_token, sample_data_token, timestamp, source_path` | `src/lidar_loader.py` |
| Preprocess | raw N×4 | `preprocess_points`: drop non-finite, drop zero-range, ROI X/Y ±80 m, Z −10…15 m; raises on empty | clean N×4 + `n_raw, n_processed, retained_ratio` | `src/preprocessing.py` |
| Perception | clean points + nuScenes annotations | `annotations_to_sensor_frame` (global→ego→sensor, single implementation) → `assign_annotation_semantics` → `build_perception_from_semantics` | `PerceptionResult` (point labels, NaN confidence for non-ML) + source array | `src/semantic_adapter.py`, `src/semantic_mapping.py` |
| Feature extraction | `PerceptionResult` | `aggregate_to_regions` (2.0 m integration cell) | `RegionFeatures` (`region_id, x, y, distance, elevation, roughness→terrain, point_density, semantic_label, semantic_importance, confidence, dynamic_relevance, uncertainty, point_count`) | `src/semantic_adapter.py`, `src/feature_adapter.py`, `src/data_types.py` |
| Importance | `RegionFeatures` | `ImportanceEngine.calculate` (weights distance 0.30, semantic 0.30, terrain 0.15, dynamic 0.15, uncertainty 0.10; max 100 m; λ 0.15) | importance ∈ [0, 1] | `src/importance_engine.py` |
| Resolution | importance | `ResolutionEngine.assign_resolution` (≥0.70→0.05 m, ≥0.45→0.10 m, ≥0.20→0.20 m, else 0.50 m) | resolution ∈ {0.05, 0.10, 0.20, 0.50} | `src/resolution_engine.py` |
| Mapper | regions + engines | `build_adaptive_map` (one region → one cell) + `validate_adaptive_map_df` | `AdaptiveMapCell` rows (`x, y, elevation, occupancy, semantic_class, confidence, importance, resolution, point_count, region_id, semantic_source`) | `src/mapper_2_5d.py` |
| Visualization | points + map DataFrame | `plot_combined_demo` (LiDAR / semantic / importance / resolution) | PNGs in `results/final/simulation_outputs/` | `src/visualization.py` |
| Validation | regions / frames | `validate_region_features`, `check_frame_health`, `safe_fps` | PASS/FAIL gates, never silent | `src/interface_validator.py`, `src/robustness.py` |

## Coordinate convention

nuScenes **LIDAR_TOP sensor frame** throughout: **+X forward, +Y left, +Z up**, origin at the
sensor. Points are never reprojected (preprocessing only filters). Annotation boxes are converted
global→ego→sensor exactly once in `annotations_to_sensor_frame`; model-development and simulation
share that code path, so conventions are identical by construction. Orientation check figure:
`results/final/simulation_outputs/coordinate_frame_check.png`.

## Data representation

`[x, y, z, intensity]` float64 N×4, all finite. `ring_index` is dropped at load and never invented.

## Semantic source

No trained perception model (CASE B). Actual point-level values: `annotation` (inside a projected
3D box) else `fallback`; region/map `semantic_source` ∈ {`annotation`, `fallback`} (see
`VALID_SEMANTIC_SOURCES`). Region label for unseen classes: `unknown`. Annotations are references,
never `model_prediction`. Confidence: point-level NaN (not applicable) → region/map `0.0`
placeholder; uncertainty `0.5` documented fallback proxy; `dynamic_relevance` heuristic prior, not
measured velocity.

## Final configuration

`results/final/config/final_config.json` (W_BASE+THRESH_C): weights {0.30, 0.30, 0.15, 0.15, 0.10},
max distance 100 m, λ 0.15, thresholds fine 0.70 / medium 0.45 / coarse 0.20, levels
{0.05, 0.10, 0.20, 0.50} m, integration cell 2.0 m, seed 42, ROI as above.

## Output structure

Per frame: `results/final/simulation_outputs/adaptive_map_sim_<token>.csv` (one row per region/cell).
Run tables: `simulation_replay_results.csv`, `frame_alignment_validation.csv`,
`simulation_consistency_check.csv`, `simulation_benchmark_verification.csv`,
`integration_issue_log.csv`, `failsafe_integration_checks.csv`.

## Module reuse (actual)

| Module | Purpose | Used by |
|---|---|---|
| `lidar_loader.py` | LiDAR loading | Model + Simulation (replay_frame) |
| `preprocessing.py` | LiDAR preprocessing | Model + Simulation (frozen path) |
| `data_types.py` | Contracts/constants | Model + Simulation |
| `interface_validator.py` | Contract validation | Model + Simulation (mapper path) |
| `feature_adapter.py` | Perception→regions adapter | Model (first handoff); sim uses `semantic_adapter.aggregate_to_regions` |
| `importance_engine.py` | Importance computation | Core pipeline (both) |
| `resolution_engine.py` | Resolution selection | Core pipeline (both) |
| `mapper_2_5d.py` | Adaptive map generation | Core pipeline (both) |
| `semantic_mapping.py` | Label taxonomy | Model + Simulation (via semantic_adapter) |
| `semantic_adapter.py` | Annotation→sensor→regions | Model + Simulation |
| `visualization.py` | Visualization | Simulation/Demo |
| `robustness.py` | Fail-safe + instrumented runner | Simulation (health/fps); handoff demo runner |

`baselines.py`, `stage1_engines.py` are **not** used by the simulation (kept for history, not claimed).

## Backend layer (17 September 2026)

```
Replay data (manifest + data/processed .npy + v1.0-mini annotations)
   ↓
Backend API (backend/routes/: thin HTTP validation only)
   ↓
Replay service (backend/services/replay_service.py: frame index + load)
   ↓
Preprocessing (src/preprocessing.py, unchanged)
   ↓
Perception (src/semantic_adapter.py + src/semantic_mapping.py, unchanged)
   ↓
Importance Engine (src/importance_engine.py, frozen W_BASE weights)
   ↓
Resolution Engine (src/resolution_engine.py, frozen THRESH_C bands)
   ↓
Adaptive 2.5D Mapper (src/mapper_2_5d.py, unchanged)
   ↓
PipelineResult (backend/services/pipeline_service.py: serialize + time)
   ↓
Frontend (stable JSON contract: schemas/input.py, schemas/output.py, schemas/errors.py)
```

Layer separation: **ML core** (`src/`, frozen) vs **backend orchestration**
(`backend/services/`, timing/serialization/storage only) vs **frontend
visualization** (consumes `PipelineResult` JSON, never ML classes).
Orchestration entry: `src/robustness.run_pipeline_condition` with
`condition="original"`. Config single source of truth:
`results/final/config/final_config.json` via `backend/config.py`.
Result store: process-memory dict (`backend/services/result_service.py`).
Timing: `total_latency_ms` = preprocess + perception + feature + mapping
(compute only); serialization and API overhead reported separately, never
as mapping latency.

## Known limitations

See `results/final/reports/limitations.md` + `simulation_handoff/README.md` §11: annotation references
(not predictions), no calibrated confidence/uncertainty, heuristic dynamics, 7-frame mini subset,
CPU-throughput FPS (not real-time), occupancy = evidence-presence indicator (not probabilistic).
