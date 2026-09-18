# Input / Output Contract — Paradox Protocol backend (actual schemas)

Source of truth: `backend/schemas/input.py`, `backend/schemas/output.py`,
`backend/schemas/errors.py`. The frontend mirrors them in
`frontend/src/types/api.ts`.

## Input

`POST /replay/load` — `{ frame_id: string }` (min length 1)

`POST /replay/run` — `{ frame_id: string, save_output?: bool,
include_visualization?: bool, max_map_cells?: int (1–5000) }`

Underlying replay reference per frame (`GET /frames` → `FrameInfo`):

```
frame_id, scene_id, timestamp, source (local .npy path), point_count
+ LiDAR points (Nx4 float64: x, y, z, intensity)
+ semantic metadata where available (nuScenes annotations; else fallback)
```

## Output

`POST /replay/run` → `{ result: PipelineResult, cache_hit: false }`

`PipelineResult`:

```
frame_id, scene_id, timestamp, status
input_point_count, processed_point_count
map_cells[]: x, y, elevation, occupancy, resolution, importance,
             semantic_class, semantic_source, confidence | null,
             point_count, region_id
map_cell_count
importance: { mean, min, max, count }          # backend output, never recomputed in JS
resolution: { fine_cells, medium_cells, coarse_cells,
              res_20cm_cells, res_50cm_cells,
              average_resolution, distribution }  # backend output
semantic: { mode: "annotation+fallback (no trained model)",
            source_counts, note }              # source preserved per cell
timing: { preprocessing/perception/feature_extraction/importance/
          resolution/mapping/total_latency_ms,
          serialization_latency_ms, wall_clock_ms, fps }
```

`GET /metrics/{frame_id}` → measured-only `DemoMetrics` (same numbers).
`GET /results/{frame_id}` → last stored result or 404 `RESULT_NOT_FOUND`.
`GET /demo/status` → `{ backend, configuration_loaded, replay_available,
available_frames, last_frame_id, last_status }`.

Errors: `{ stage, error_code, message (≤500 chars), frame_id? }` — no stack
traces to clients. Codes: `FRAME_NOT_FOUND, CONFIG_NOT_FOUND, DATA_LOAD_FAILED,
PREPROCESSING_FAILED, PERCEPTION_FAILED, FEATURE_EXTRACTION_FAILED,
IMPORTANCE_FAILED, RESOLUTION_FAILED, MAPPING_FAILED, RESULT_NOT_FOUND,
INVALID_REQUEST`.
