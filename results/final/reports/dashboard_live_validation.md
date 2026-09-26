# Dashboard Live Validation (2026-09-25)

## Startup (user workflow, no env-var key setup)
1. `config/local_secrets.py` holds the key (git-ignored; pasted once).
2. Backend: `.\.venv-pb\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000`
3. Frontend: `cd frontend; npm run dev` (serves http://127.0.0.1:5173).
4. Dashboard URL: http://127.0.0.1:5173 (API at http://127.0.0.1:8000).
5. Click "Start Jev Autonomous" (or "Start Manual"), then manual
   Forward/Left/Right/Stop as needed. Stop/Reset call the real loop.

## Transport
- Existing `/ws/live` reused (2 Hz snapshots; 2 s polling fallback
  in `useLiveSimulation`). No new architecture.
- Also live: GET /simulation/state, /system/status,
  POST /simulation/live/{start,stop,reset}, /simulation/action/*.
- Secret audit: snapshot contains no key/Authorization/local_secrets
  (grep + leak tests; TestClient-verified `"leak": false`).

## Live contract (§2, additive over compat flat keys)
pipeline stages 1-9 (LIVE + verified flags; stage7 follows Jev
LIVE/ERROR/FALLBACK/UNAVAILABLE), lidar {point_count, frame_count,
fps measured, status}, map {cell_count, ≤1000 cells, importance,
resolution, semantic_classes}, decision {model jev-1.13, action,
confidence, probabilities, latency, status}, safety {status,
override, reason}, vehicle {x,y,z,yaw,speed,yaw_rate}, execution
{proposed, executed, source}, metrics {5 latencies + speed,
distance, collisions, jev/manual/directional/stop/override counts},
plus `pipeline_result` (current frame) feeding the EXISTING
AdaptiveMapView/ImportanceView/ResolutionView — map updates every
frame, never prerecorded. Stages 1-6 code untouched.

## §15 200-poll live test (backend autonomous loop, dashboard path)
- 200/200 distinct frame_ids; WS delivered live-00001/00008/00015...
- LiDAR 183-202 pts varying; map cells 56-74 varying.
- Jev actions {forward, turn_left, turn_right}; conf 0.31-1.0 varying.
- Safety {SAFE_TO_EXECUTE}; exec {forward, turn_left, stop, turn_right}.
- Poses 119 distinct: (-49.0, 0.0) -> (-19.0, 3.9), distance 32.3 m.
- stage7 LIVE; loop_ms measured; counters: 2960 steps, 592 Jev OK,
  1315 directional / 1645 stop, 0 safety overrides.
- collisions_total 1915 (see limitation below) — displayed, not hidden.

## Verified panels (§4-14)
Jev (model/action/conf/probs/latency/status, never hardcoded);
Safety (proposed/decision/override/reason/executed with live values);
Vehicle (x/y/z/yaw/speed/yaw-rate updating); LiDAR (frame/points/fps/
cells/sim-time); live 2.5D map + importance + resolution from current
frame; pipeline stages with LIVE vs verified distinction; modes
(manual tagged source=manual, excluded from Jev counts); START/STOP/
RESET hit real endpoints; LIVE/STOPPED from backend state; metrics
with NOT AVAILABLE where unmeasured.

## Screenshot status
SCREENSHOT NOT AVAILABLE — HEADLESS HOST (no browser capture here;
panels verified via built bundle + endpoint payload assertions).

## Known limitations (measured, not conjectured)
1. Extended free driving accumulates wall contacts (1915 contact
   steps / 2960): safety by design gates forward motion ≤5 m +
   emergency/invalid-map only — lateral turns are ungated. No
   override fired because the forward rule never tripped at
   decision epochs. Follow-up: lateral clearance safety (not this task).
2. Loop iterations dual-step physics (advance + control), so
   effective speed is ~half the 2 m/s command — pre-existing design.
3. Post-fix Jev confidences skew extreme (up to 1.0) — recorded
   service values; monitoring note.
4. Frontend suite 38/38, build green; backend live tests 55/55.

## Workflow end state
Key in file -> backend up -> frontend up -> dashboard ->
JEV AUTONOMOUS -> live LiDAR/map/action/confidence/safety/vehicle/
metrics. Physical testing: NOT EXECUTED.
