# Final Test Checklist — judge dashboard (executed)

Date: task date 20 September 2026 work session. Backend checks ran in fresh
processes against the frozen pipeline; frontend via vitest + production build.

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Backend starts | PASS | `GET /health` → ok |
| 2 | Frontend starts | PASS | `npm run build` clean; `npm run preview` → HTTP 200 (verified earlier); `npm run dev` serves dashboard |
| 3 | Backend connection works | PASS | header pill from live `/health` + `/config` + `/demo/status` |
| 4 | Real LiDAR frame loads | PASS | `POST /replay/load` 34359 pts, frame/scene/timestamp shown |
| 5 | Simulation/replay works | PASS | Play loop runs load→run→metrics per frame; `POST /replay/run-sequence` 3/3 |
| 6 | Frame synchronization works | PASS | single `useReplay` state; select clears stale result |
| 7 | Raw LiDAR updates | PASS | LidarView re-renders from `currentResult`; maps differ per frame (proven) |
| 8 | Semantic view updates | PASS | semantic mode from backend `semantic_class`; legend + source shown |
| 9 | Importance map updates | PASS | backend `importance`, heatmap re-renders per frame |
| 10 | Resolution map updates | PASS | backend `resolution` tiers 0.05/0.10/0.20/0.50 |
| 11 | Adaptive 2.5D map updates | PASS | `map_cells` per frame (533/1132/509 measured, signatures differ) |
| 12 | Object/terrain panel updates | PASS | grouped from current result; render-tested |
| 13 | Metrics update | PASS | backend timing; mapping vs wall-clock separated |
| 14 | Status updates | PASS | pill + system status from live state, never all-green by default |
| 15 | Benchmark panel loads | PASS | stored asset + current-replay row; vitest-covered |
| 16 | USP panel works | PASS | takeaway block from live result values |
| 17 | Previous/Next works | PASS | implemented + guarded during processing |
| 18 | Play/Pause works | PASS | implemented (interval + overlap guard + speed); loop logic reviewed; manual judge-run step in START_DEMO |
| 19 | Reset works | PASS | stops replay, clears result/metrics/views/events/camera (hook test) |
| 20 | Error handling works | PASS | `failure_path_results.csv` 9/9 (404/validation/semantic/empty/malformed) |
| 21 | Cold-start works | PASS | fresh processes, config/dataset from disk, no notebook |
| 22 | Multi-frame replay works | PASS | 3-frame integration: MAPS_DIFFER_PER_FRAME true, sequence 3/3, no errors |
| 23 | Offline operation works | PASS | localhost + local files only; no external calls in client (except localhost API) |
| 24 | Backup demo works | PASS | 8 files incl. new `semantic_view.png` (800×600 verified); README labels PRE-RENDERED BACKUP |

Frontend suite: 34/34 vitest PASS (7 files). Backend suite: 13/13 pytest PASS
(verified 19 Sept; backend code unchanged since). No ML files modified
(`git status` on `src/`, `backend/`, `config/`, `results/final/config/` clean
apart from intended report additions).

## Known limitations (documented, not fabricated)

1. No true oblique-3D rendering: views are top-down canvas 2D with elevation
   encoding + pan/zoom. Three.js was deliberately not introduced.
2. No live sensor: replay of nuScenes Mini only; Play cadence (8 s/speed) is
   display pacing, unrelated to measured latency.
3. No model confidence: objects/semantics are annotation references; confidence
   column intentionally shows N/A.
4. `road_driveable`/`vegetation` annotation cells are rare in validated frames;
   terrain split honestly shows unclassified (fallback) majority where true.
5. Play auto-advance verified by code review + hook/unit tests; full
   multi-minute judge-preview run should be done once on the demo machine.
