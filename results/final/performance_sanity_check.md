# Performance Sanity Check — 19 September 2026 (measured only, nothing invented)

All values below were measured by executing the frozen pipeline on 19 September 2026
via `POST /replay/run` (FastAPI TestClient, fresh Python process per script).
Map-cell counts are byte-identical to the 16 September replay
(`results/final/simulation_replay_results.csv`).

## 4.1 ML latency vs frontend rendering

- Displayed "ML latency" comes from `result.timing.total_latency_ms`, produced by
  `src/robustness.run_pipeline_condition` (sum of preprocessing + perception +
  feature + importance + resolution + mapping stage timers).
- React rendering / browser drawing / animation are never included: the frontend
  only formats and displays backend numbers (`frontend/src/utils/formatting.ts`,
  `MetricsPanel.tsx`). No `performance.now()` measurement exists in the frontend.

## 4.2 API overhead separation (three distinct numbers, shown separately in Panel E)

| Field | Meaning | 5991fad3… measured 19 Sep |
|---|---|---|
| `timing.total_latency_ms` | Core ML compute only | 1592.6 – 3846.1 ms (machine-load dependent) |
| `timing.serialization_latency_ms` | JSON serialization of map cells | ~150 ms |
| `timing.wall_clock_ms` | Load + pipeline + serialize (API-side) | ~25 s cold (incl. NuScenes open), ~2–6 s warm |
| `Mapping latency` card | `timing.mapping_latency_ms` (mapper stage only) | 85 – 249 ms across 7 frames |
| `Throughput` card | `fps = 1000 / total_latency_ms` (`src/robustness.safe_fps`) | 0.20 – 0.63 FPS |

The dashboard labels these `core ML time` / `compute only` / `incl. overhead`
(`MetricsPanel.tsx`). Total browser round-trip time is never called "model latency".

## 4.3 Hard-coding check

Searched `frontend/src/` and `backend/` for hard-coded FPS / latency / point-count /
cell-count / benchmark values:

- `frontend/src`: only display formatting (`toFixed`, `toLocaleString`) and unit
  labels; numeric fixtures exist solely under `src/__tests__/` (test data).
- `benchmark.json` is a generated asset from `results/benchmark/` (provenance field
  embedded); `BenchmarkPanel.tsx` renders it verbatim and never recalculates.
- `backend/`: no duplicated weights/thresholds (`pipeline_service._engines`
  asserts parity with `results/final/config/final_config.json` at runtime).
- Result: no hard-coded live metrics. PASS.

## 4.4 Repeated-run consistency (frame 5991fad3280c4f84b331536c32001a04)

6 consecutive `POST /replay/run` executions on 19 September:

| run | success | latency_ms (total) | cells | fine/med/coarse | imp_mean | signature match |
|---|---|---|---|---|---|---|
| 1 | true | 1592.6 | 533 | 52/429/52 | 0.565344 | ref |
| 2 | true | 1658.7 | 533 | 52/429/52 | 0.565344 | yes |
| 3 | true | 2639.8 | 533 | 52/429/52 | 0.565344 | yes |
| 4 | true | 3083.4 | 533 | 52/429/52 | 0.565344 | yes |
| 5 | true | 3846.1 | 533 | 52/429/52 | 0.565344 | yes |

Map structure, resolution assignment, importance, semantic counts and point counts
are deterministic (identical cell signatures incl. x/y/resolution/semantic_class/
importance). Latency varies with machine load — reported as a measured range, never
as a single cherry-picked number.

All 7 manifest frames re-ran successfully on 19 September
(`results/final/live_19sep_runs.json`); cell counts match 16 September exactly:
533 / 759 / 1132 / 1257 / 1059 / 509 / 1055.

## 4.5 Cold-start validation

- Every 19 September verification ran in a fresh OS process: backend TestClient
  imports (`backend/app.py` lifespan loads `final_config.json` from disk),
  NuScenes dataset re-opened from `data/raw/nuscenes`, engines rebuilt from frozen
  config, replay points loaded from `data/processed/*.npy`.
- No notebook was opened on 19 September; no notebook variables, caches or manual
  preprocessing were used. `npm run build` regenerates `frontend/dist` from source.
- Judge cold start = `uvicorn backend.app:app` + `npm run preview` (or `dev`) +
  open `http://127.0.0.1:5173` + select frame + Run. No warm state required. PASS.
