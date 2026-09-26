# PyBullet + Jev Validation (2026-09-25)

Environment: `.venv-pb`, Python 3.11.0, pybullet 3.2.6 (source-built with
local MSVC 14.51 — PyPI hosts no Windows wheel for any modern pybullet).

## check_pybullet.py — PASS
import / connect(DIRECT) / world / 120 steps @ 0.0089 ms / physics (body
fell 2.0 -> 0.791 m) / disconnect. All ok.

## check_jev.py — BLOCKED (honest)
`{"status": "BLOCKED", "available": false,
"reason": "Jev credentials not configured",
"endpoint": "https://openrouter.ai/api/alpha/decisions",
"endpoint_source": "default", "model": "typesafe/jev-1.13"}`
Unblocks when the user pastes a key into `config/local_secrets.py`.
Key transport verified by mocked tests (correct URL/model/body shape,
Bearer header, no chat-completions, no key in logs/records).

## 200-step run — COMPLETED (degraded safe-stop)
`python scripts/run_pybullet_jev.py --steps 200
--decision-interval-steps 10 --run-id gap19-200`
- 200 new LiDAR frames (182 pts each) -> 200 Stage 1-6 results
  (55-65 map cells) -> 16 distinct 13-D states
- 20 Jev calls, all UNAVAILABLE (no key) -> fallback STOP x200
- safety re-evaluated every step; 0 overrides; 0 collisions
- latencies (mean): sim_step 0.128 ms, lidar 6.09 ms,
  stages_1-6 54.5 ms, jev_path 0.95 ms, safety 0.37 ms,
  action_exec 2.38 ms, loop 64.3 ms
- artifacts: results/pybullet/closed_loop_trace.json,
  results/decisions/closed_loop_decisions.csv (200 rows),
  results/decisions/jev_requests.jsonl (run_id-tagged),
  results/metrics/pybullet_jev_metrics.json

## Collision signal
Ground contact is filtered (plane body / |normal_z|>0.95); only
non-ground contacts count, edge-triggered. Live-verified:
15 free steps -> 0 collisions with ground contacts filtered;
driving into a placed obstacle -> collision flagged with body id.
Tests: test_ground_contact_filtered_live,
test_real_obstacle_contact_counted_live (both PASS live).

## What remains
Live Jev decisions + Jev-directed motion require the user key.
Nothing above is fabricated; degraded values are labelled fallback.
Physical testing: NOT EXECUTED.
