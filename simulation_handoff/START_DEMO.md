# START_DEMO — Judge Demo (local only)

Runs the frozen pipeline behind the React dashboard using local/non-sensitive
nuScenes Mini replay data. No uploads, no public servers.

## Terminal 1 — Start FastAPI backend

```bash
cd "C:\Users\SARKAR\Documents\Default Project"
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Confirm: open `http://127.0.0.1:8000/health` → `{"status":"ok",...}`.

## Terminal 2 — Start React frontend

```bash
cd "C:\Users\SARKAR\Documents\Default Project\frontend"
npm install        # first time only
npm run build
npm run preview    # serves http://127.0.0.1:5173
```

(For development instead: `npm run dev`.)

Backend URL override if needed (default is `http://127.0.0.1:8000`):

```bash
# frontend/.env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Browser — Open local Vite URL

```
http://127.0.0.1:5173
```

## Demo (judge script, ~3 minutes)

1. **Check backend** — header shows `Backend: Connected` + `Model: Loaded`.
   If not: the dashboard shows `Backend unavailable. Start the local FastAPI
   service and retry.` with a Retry button (never fake values).
2. **Select frame** — Replay Controls dropdown (7 frozen evaluation frames, e.g.
   `5991fad3…` scene-0655). Prev/Next step through them.
3. **Run or Play** — press **Run** for a single frame (frozen pipeline,
   ~10–30 s first run), or **Play** for auto-advance through the backend
   pipeline (speed 0.25x–2x sets display cadence only; measured latency is
   unaffected). Views support scroll-zoom, drag-pan, double-click reset.
4. **Show adaptive map** — Panel C: marker size = backend `resolution`,
   brightness = elevation, white ring = valid semantic source.
5. **Show importance/resolution** — Panel D: red = high importance; blue/teal/
   orange/red marker sizes = 0.05/0.10/0.20/0.50 m tiers from `GET /config`.
6. **Show metrics** — Panel E: input/processed points, cell counts, mapping
   latency (core ML) vs API wall clock (overhead) shown separately, FPS as
   measured (`1000/total ms`), semantic source + frame status.
7. **Show benchmark comparison** — stored Proposed Adaptive vs Uniform 5 cm vs
   Distance-based Adaptive (from `results/benchmark/`, not recalculated) next to
   the **Current replay result** row.

## If something fails

| Symptom | Action |
|---|---|
| `Backend unavailable` banner | Start Terminal 1, press Retry |
| `FRAME_NOT_FOUND` / `RESULT_NOT_FOUND` | Re-select a frame from Panel A, press Run |
| `Benchmark results are not available` | Run `npm run benchmark:asset` in `frontend/` and rebuild |
| Port 8000/5173 busy | Stop the other process; ports are fixed for the demo |

Never edit `results/final/config/final_config.json` or `src/` during the demo —
the ML configuration is frozen.
