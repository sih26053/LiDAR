# 19 September Full-System Test (executed, clean start per process)

Date: 19 September 2026 (system clock 18 Sep; task date 19 Sep). No notebook opened.
Each backend check ran in a fresh Python process (cold import, config from disk,
NuScenes re-opened, engines rebuilt). Frontend rebuilt from source (`npm run build`).

| Stage | Command / action | Result |
|---|---|---|
| Backend starts cleanly | `uvicorn backend.app:app --host 127.0.0.1 --port 8000` then `GET /health` | PASS (`{"status":"ok","service":"paradox-protocol-backend"}`) |
| Frozen config loads | `GET /config` | PASS (W_BASE+THRESH_C, `results/final/config/final_config.json`) |
| Replay available | `GET /demo/status`, `GET /frames` | PASS (7 frames, replay_available=true) |
| Select known-good frame | `POST /replay/load` 5991fad3… | PASS (scene-0655, ts 1535385092150099.0, 34359 pts) |
| Run model | `POST /replay/run` ×7 frames | PASS all 7 (cells 533/759/1132/1257/1059/509/1055, byte-identical to 16 Sep) |
| Original LiDAR | raw `.npy` plotted in backup + Panel B renders `map_cells`/input | PASS (backup `original_lidar.png` 800×600 from 34,359 pts) |
| Importance map | `result.importance` (mean/min/max/count) | PASS (e.g. mean 0.565344, count 533) |
| Resolution map | `result.resolution` tiers 0.05/0.10/0.20/0.50 | PASS (52/429/52/0) |
| Adaptive 2.5D map | `map_cells` x/y/elevation/occupancy/semantic/importance/resolution | PASS (533×11 fields) |
| Metrics | Panel E backend values with units | PASS (metrics endpoint confirms stored result) |
| Benchmark comparison | `frontend/public/benchmark.json` vs current replay row | PASS (3 methods, provenance embedded, never recalculated) |
| USP / takeaway | ExplanationPanel takeaway block (added 19 Sep, tested, built) | PASS |
| Backend tests | `pytest backend/tests` | PASS 13/13 |
| Frontend tests | `npm run test` (vitest) | PASS 19/19 |
| Frontend build | `npm run build` | PASS (dist + benchmark.json) |
| Frontend serve | `npm run preview` → `GET /127.0.0.1:5173/` | PASS (HTTP 200) |
| Repeated runs | 6× same frame | PASS (identical signatures; latency range 1.6–3.9 s) |
| Failure paths | `failure_path_results.csv` F-01…F-09 | PASS 9/9 (two harness-expectation errors diagnosed, corrected, rerun) |
| Backup demo | `simulation_handoff/backup_demo/` (4 PNG + CSV + metrics + README) | PASS, offline, labelled PRE-RENDERED BACKUP |
| Cold start | fresh processes, no notebook, rebuild from source | PASS |

Known variances (not failures): total latency varies with machine load
(1.6–5.1 s across frames today vs 1.3–1.8 s on 16 Sep); outputs deterministic.
`npm` must be launched via `cmd /c` on this machine (documented sandbox quirk, not a demo issue).
