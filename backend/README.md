# Paradox Protocol Backend (17 September)

Local-only SIH prototype backend. Replays validated nuScenes Mini frames
through the **frozen** ML pipeline and returns structured JSON for the frontend.

## 1. Purpose

```
Replay frame / sequence
        ↓
FastAPI routes (thin: validate -> service -> respond)
        ↓
Backend services (orchestration)
        ↓
Existing frozen src/ modules (no copies, no retuning)
        ↓
Frozen ML pipeline (W_BASE+THRESH_C, 15 September)
        ↓
Structured PipelineResult JSON
        ↓
Frontend
```

## 2. Architecture

| Layer | Location | Role |
|---|---|---|
| Routes | `backend/routes/` | HTTP validation + status codes only |
| Services | `backend/services/` | Orchestration, timing, serialization |
| Frozen ML | `src/` | `preprocessing`, `semantic_adapter`, `importance_engine`, `resolution_engine`, `mapper_2_5d` via `src/robustness.run_pipeline_condition` |
| Config | `results/final/config/final_config.json` | Single source of truth, loaded by `backend/config.py` |
| Replay data | `data/processed/*.npy` + `results/final/final_test_manifest.csv` | 7 frozen evaluation frames |
| Dataset | `data/raw/nuscenes` (v1.0-mini) | Annotation reference only |

No importance weights or resolution thresholds exist in any route file.

## 3. Installation

```bash
pip install fastapi uvicorn pydantic httpx numpy pandas nuscenes-devkit pyquaternion
```

(`httpx` is required for the test client.)

## 4. Start the backend (local-only)

```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000/health`. Bind to `127.0.0.1` for the SIH demo.

## 5. API endpoints

```
GET  /health
GET  /config
GET  /frames
POST /replay/load
POST /replay/run
POST /replay/run-sequence
GET  /results/{frame_id}
GET  /metrics/{frame_id}
GET  /demo/status
```

## 6. Request examples

```bash
curl http://127.0.0.1:8000/frames
curl -X POST http://127.0.0.1:8000/replay/load -H "Content-Type: application/json" \
  -d '{"frame_id": "5991fad3280c4f84b331536c32001a04"}'
curl -X POST http://127.0.0.1:8000/replay/run -H "Content-Type: application/json" \
  -d '{"frame_id": "5991fad3280c4f84b331536c32001a04"}'
curl http://127.0.0.1:8000/results/5991fad3280c4f84b331536c32001a04
curl http://127.0.0.1:8000/metrics/5991fad3280c4f84b331536c32001a04
curl http://127.0.0.1:8000/demo/status
```

Sequence replay (single-frame runner applied per frame):

```bash
curl -X POST http://127.0.0.1:8000/replay/run-sequence -H "Content-Type: application/json" \
  -d '{"start_frame": "5991fad3280c4f84b331536c32001a04", "end_frame": "6d1d793b68b24165be5755a54054c82a"}'
```

## 7. Response examples

`/replay/run` returns `{"result": PipelineResult, "cache_hit": false}` where
`PipelineResult` carries `frame_id`, `scene_id`, `timestamp`, `map_cells`,
`importance` (mean/min/max), `resolution` (fine/medium/coarse counts +
average), `semantic` (mode `annotation+fallback`, per-source counts),
`timing` (per-stage + total + fps), and `status`.

Map cell:

```json
{"x": 1.2, "y": -2.3, "elevation": 0.4, "occupancy": 1.0,
 "resolution": 0.05, "importance": 0.86, "semantic_class": "vehicle",
 "semantic_source": "annotation", "confidence": 0.0,
 "point_count": 42, "region_id": 7}
```

`confidence` is the `0.0` non-ML interface placeholder for
annotation/fallback cells -- never ML confidence. Annotation-derived labels
are evaluation references, never model predictions.

## 8. Error handling

Structured `ErrorResponse`: `{"stage", "error_code", "message", "frame_id"}`.
Codes: `FRAME_NOT_FOUND`, `CONFIG_NOT_FOUND`, `DATA_LOAD_FAILED`,
`PREPROCESSING_FAILED`, `PERCEPTION_FAILED`, `FEATURE_EXTRACTION_FAILED`,
`IMPORTANCE_FAILED`, `RESOLUTION_FAILED`, `MAPPING_FAILED`,
`RESULT_NOT_FOUND`, `INVALID_REQUEST`. No tracebacks leak to the client;
full tracebacks stay in server logs.

## 9. Configuration location

`results/final/config/final_config.json` (selection `W_BASE+THRESH_C`).
Loaded once at startup via `backend/config.py`, validated, and injected
into engine constructors. Missing/invalid config fails fast.

## 10. Replay data location

- Frames: `results/final/final_test_manifest.csv` (7 frames)
- Points: `data/processed/<frame_id>_LIDAR_TOP_xyzi.npy`
- Metadata: `data/processed/<frame_id>_metadata.csv`
- Annotation reference: `data/raw/nuscenes` (v1.0-mini)

## 11. Timing definitions

- `total_latency_ms` = preprocessing + perception + feature extraction +
  mapping (compute only; visualization never inside; matches
  `src/robustness.run_pipeline_condition`).
- `fps` = `1000 / total_latency_ms` (measured CPU throughput, not a
  real-time claim).
- `serialization_latency_ms` = map-DataFrame-to-JSON time (API concern).
- `wall_clock_ms` = end-to-end service time incl. dataset lazy-load on
  first call. Never reported as mapping latency.

## 12. Security / local-only behavior

- Binds to `127.0.0.1`; CORS allow-list is loopback origins only.
- No external database, no cloud upload, no secrets in code.
- `GET /config` returns safe metadata only.
- Only the public/non-sensitive nuScenes Mini replay subset is served.
