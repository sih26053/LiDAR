# First Handoff Integration — Real Processed LiDAR → Importance/Resolution v1

> 10 September 2026 — Together task. Status: **integration code complete and
> tested; real-data execution BLOCKED** (no processed nuScenes frames and no
> `lidar_loader.py` / `preprocessing.py` exist in this checkout; downloading a
> dataset is out of scope for this task). The notebook
> `notebooks/02_Interface_Integration_Test.ipynb` runs sections 1–3 green and
> halts at section 4 with the specified
> `FileNotFoundError("No processed nuScenes LIDAR_TOP frame found.")`.
> It executes end to end the moment Rajashree's
> `data/processed/*_LIDAR_TOP_xyzi.npy` + `*_metadata.csv` are present
> (Colab Drive or local checkout — paths resolve portably).

## Pipeline (all production code, no notebook-only logic)

```text
REAL nuScenes LIDAR_TOP frame
      ↓  Rajashree: lidar_loader.py / preprocessing.py (output on disk)
Processed N×4 [x, y, z, intensity]  (.npy) + *_metadata.csv
      ↓  notebook §7 / run_first_handoff(): LiDARFrame + validate_lidar_frame
Basic geometric features (3D range diagnostic; planar centroid range per contract)
      ↓  2.0 m XY integration grid (engineering scaffold, not final mapping)
Region geometry (centroid, elevation, height variation, count, density)
      ↓  terrain proxy + conservative fallbacks (below)
RegionFeatures  →  validate_region_features (mandatory, never weakened)
      ↓  ImportanceEngine() default config — UNCHANGED
final_importance [0,1]
      ↓  ResolutionEngine() default config — UNCHANGED
resolution_m ∈ {0.05, 0.10, 0.20, 0.50}
```

Reusable entry point: `src/feature_adapter.py::run_first_handoff(points,
frame_metadata, cell_size=2.0)` → `(lidar_frame, region_features,
handoff_rows)`. The notebook proves `notebook path == run_first_handoff()
path` by cross-checking Frame A through both. No core-engine modification is
required or performed (weights `0.30/0.30/0.15/0.15/0.10`, `λ=0.15`,
thresholds `0.75/0.50/0.25` — asserted unchanged by tests).

## Provenance contract (§37): REAL vs DERIVED vs FALLBACK

| Category | Fields |
|---|---|
| REAL MEASURED (sensor + loader) | `x, y, z, intensity`, point counts, `frame_id`, `sample_token`, `lidar_token`, `timestamp` |
| GEOMETRIC DERIVED (computed here, deterministic) | region centroids, `distance = hypot(cx, cy)`, `elevation = mean(z)`, `height_variation = std(z)`, `point_density`, `terrain_complexity = height_variation / max(...)` clipped [0,1] — a **temporary proxy, NOT a terrain classifier** |
| TEMPORARY FALLBACK (no perception model yet) | `semantic_label = "unknown"`, `semantic_importance = 0.0`, `dynamic_relevance = 0.0`, `confidence = 0.0`, `uncertainty = 0.5` — **never measured, never presented as measured** |

Note: these fallbacks are deliberately stricter than the perception-pipeline
proxies in `src/data_types.py` (`DYNAMIC_PROXY unknown → 0.20`, uncertainty
`1 − confidence`); the first handoff forces unknowns to zero + `0.5` so no
output can be mistaken for real perception. Constants live in
`src/feature_adapter.py` (`FIRST_HANDOFF_*`).

## What was verified (executed, not claimed)

- `pytest -q`: **96 passed**, incl. 8 new `test_first_handoff.py` tests that run
  `run_first_handoff()` on synthetic-format N×4 (explicitly labeled stand-ins):
  loader-format acceptance, identity preservation, exact fallbacks, equality
  with direct engine calls, engine configs untouched, determinism across two
  frames, rejection of NaN/wrong-shape/empty/bad-metadata inputs.
- Notebook sections 1–3 executed: env, production imports, full 9-September
  interface field check → `[PASS] 9 September interface intact`.
- All 24 notebook cells smoke-tested on synthetic-format stand-ins in a temp
  dir (repo untouched): 1468 regions validated, engines consumed without
  modification, both CSVs + both figures + report generated,
  `ALL 7 FIRST-HANDOFF CHECKS PASSED`, invalid inputs correctly rejected.
  **This proves the code path, NOT real-data behavior.**

## Limitations / deferred

- No real frame tested (data absent); no loader/preprocessor modules in this
  checkout — they belong to Rajashree's side and were NOT rebuilt here (§36).
- Resolution distribution from any future real run is descriptive only, never
  a performance/compression benchmark. No detection/semantic/mapping accuracy
  claims. No final Adaptive 2.5D Mapper, frontend, or database work.
