# Semantic Perception + Geometric Feature Integration — Real LiDAR → Importance/Resolution v1

> 11 September 2026 task. Status: **complete, executed on real data 16 Sept
> 2026** (see "What was verified"). `LIDARSEG_AVAILABLE = False` (lidarseg is
> a separate nuScenes expansion, absent from the mini mirror), so the
> annotation fallback carried the semantics -- per-point source audit in
> `results/semantic_source_audit.csv`.

## Pipeline

```text
REAL nuScenes LIDAR_TOP frame
      ↓  10 Sept: lidar_loader.py / preprocessing.py (output on disk)
Processed N×4 [x, y, z, intensity] (.npy) + *_metadata.csv
      ↓  notebook §4-6: sample_token recovery + nuScenes sample check
Semantic source (§7/10/11):
  ├── lidarseg per-point labels, if available (index i <-> point i, asserted)
  ├── object-annotation boxes otherwise (inside -> mapped class;
  │   outside -> unknown/fallback, never auto-labelled)
  └── existing perception model, if already implemented (real confidence)
      ↓  src/semantic_mapping.py: nuScenes -> project classes
Project classes: vehicle / pedestrian_vru / static_manmade /
                 vegetation / road_driveable / unknown
      ↓  src/semantic_adapter.py: distance, elevation (+ region std/density)
PerceptionResult (existing 9-Sept class; NaN confidence = not applicable
for non-ML sources -- backward-compatible validator extension)
      ↓  2.0 m XY integration grid (same convention as 10 Sept handoff)
RegionFeatures (existing 9-Sept class) + audit fields (semantic_source,
dominant_class_ratio, uncertainty_source, dynamic_source)
      ↓  validate_region_features (mandatory, never weakened)
ImportanceEngine() default config -- UNCHANGED
      ↓  final_importance [0,1]
ResolutionEngine() default config -- UNCHANGED
      ↓  resolution_m in {0.05, 0.10, 0.20, 0.50}
```

New code (additive only): `src/semantic_mapping.py`, `src/semantic_adapter.py`.
`src/interface_validator.py` gains one backward-compatible extension
(NaN point-level confidence = documented "not applicable"; all previous
inputs still pass, `inf`/out-of-range still rejected). `src/data_types.py`
gains a doc comment. No engine, config, or contract redesign.

## Provenance contract: REAL vs DERIVED vs FALLBACK

| Category | Fields |
|---|---|
| REAL MEASURED (sensor + loader) | `x, y, z, intensity`, point counts, `frame_id`, `sample_token`, `lidar_token`, `timestamp`; lidarseg label ids / annotation boxes when present |
| GEOMETRIC DERIVED (deterministic) | per-point `distance = norm(x,y,z)`, `elevation = z`; region `distance = hypot(cx,cy)`, `elevation = mean(z)`, `height_std = std(z)`, `point_density = count / cell^2`, `terrain_complexity = height_std / max(...)` clipped [0,1] -- a **temporary proxy, NOT a terrain classifier** |
| SEMANTIC DERIVED (mapping) | `project_label` via `map_nuscenes_label_to_project` / `map_lidarseg_id_to_project`; region dominant class + `dominant_class_ratio` (**consistency measure, NOT confidence**) |
| DOCUMENTED FALLBACK / PRIOR | `semantic_importance` = project engineering parameters; `dynamic_relevance` = `PROJECT_DYNAMIC_PRIOR` heuristic (**not measured velocity**); `uncertainty` = 0.5 fallback (**not calibrated**); `RegionFeatures.confidence` = 0.0 placeholder for non-ML regions (**not a measurement**); point-level `confidence` = NaN (**not applicable**) |

## Confidence rule

- `semantic_source = "model"` -> `confidence` = actual model confidence in [0,1].
- `lidarseg` / `annotation` / `fallback` -> `confidence` = NaN (point level);
  `build_perception_from_semantics` **rejects** finite confidence for these
  sources so a fabricated `0.95` fails fast instead of flowing downstream.

## What was verified (executed, not claimed)

- **Dataset**: `aadimator/nuscenes-mini` (Kaggle, 4,199,156,436 bytes,
  integrity-checked) extracted to `data/raw/nuscenes` (404 LIDAR_TOP samples,
  `v1.0-mini/` tables; lidarseg absent -> annotation path).
- **Preprocessing**: `src/lidar_loader.py` + `src/preprocessing.py`
  (10-September contract: N x 4, NaN/Inf removal, zero-range + ROI filtering,
  `<sample_token>_LIDAR_TOP_xyzi.npy` + `<sample_token>_metadata.csv`) ran
  over all **404/404 samples, 0 failures** into `data/processed/` (808 files).
- **Notebook 03 executed end to end** (`nbconvert --execute`, 49 cells,
  0 error outputs): `notebooks/03_Semantic_Perception_Integration.executed.ipynb`
  ends with `11 SEPTEMBER SEMANTIC INTEGRATION: PASS`.
- **Real frame 1** (`00889f8a...`, 34,658 pts -> 410 regions):
  point audit `annotation 7144 / fallback 27514`; regions
  `vehicle 31 / pedestrian_vru 1 / unknown 378` (sources: annotation 32,
  fallback 378); importance mean/min/max 0.577/0.373/0.882; resolution
  `{0.10: 328, 0.20: 54, 0.05: 28}`. Scenarios: pedestrian_vru @23.4 m ->
  importance 0.824 -> 0.05 m; vehicle @24.4 m -> 0.788 -> 0.05 m;
  rough-terrain top (terrain 1.0) -> 0.637; static_manmade / road_driveable
  honestly reported "Class not observed in this frame."
- **Multi-frame**: 3 real frames through the identical pipeline (no
  frame-specific code); per-frame point/region/importance/resolution rows in
  the executed notebook.
- **Real-data bug fixed honestly**: annotation boxes are map-frame while
  points are sensor-frame -- first execution yielded 0 assignments and the
  audit said so; fixed via `annotations_to_sensor_frame` (devkit
  global -> ego -> sensor chain) plus the devkit length-along-x box
  convention. Verified: 990/34,590 points assigned on the debug frame.
- `pytest -q`: **122 passed** (96 pre-existing + 26 new), incl. taxonomy,
  mapping closure, geometry, inside-only assignment, sensor-frame transform
  (identity + translation stubs), NaN-confidence honesty, fabricated-
  confidence rejection, region validity, engine-config invariance.
- Outputs: `results/semantic_region_results.csv`,
  `results/real_importance_results.csv`, `results/semantic_source_audit.csv`,
  `results/perception_results/`, five figures in `results/figures/`.

## Current limitations

- Annotation-derived labels are not ML predictions.
- Annotation-derived labels do not provide calibrated ML confidence.
- No semantic segmentation accuracy claim is made without
  prediction-vs-ground-truth evaluation.
- Dynamic relevance may be a heuristic when motion is unavailable
  (vehicle != measured moving vehicle).
- Terrain complexity is currently a geometric proxy.
- Initial semantic-importance values are engineering parameters.
- Importance Engine weights remain prototype parameters.
- `flat.terrain` and the `vehicle.ego` mask map to `unknown` (documented
  overrides in `src/semantic_mapping.py`).
- Annotation boxes use yaw-only oriented containment; velocity/track
  attributes are not consumed.

## Deferred work

- Real-frame execution in Colab once `data/processed/` + nuScenes dataroot
  are mounted (notebook is ready; no code change needed).
- Model-backed `semantic_source = "model"` path (adapter already enforces
  the confidence contract for it).
- Temporal tracking for measured dynamic relevance.
- Trained terrain classifier to replace the roughness proxy.
- Final Adaptive 2.5D Mapper, benchmark optimization, frontend, database
  (all explicitly out of scope).
