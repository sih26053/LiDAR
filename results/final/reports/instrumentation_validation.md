# Instrumentation Validation (gaps closed, no stages upgraded on this alone)

Date: 2026-09-24. Env: `paradox_pybullet311`. Stages 1–6 untouched.
Safety layer and architecture unchanged (only telemetry added around them).

## 1. Per-decision Jev log — DONE

- `decide_with_safety(..., frame_id, log_path)` now returns `frame_id`,
  `probabilities`, `jev_status`, `jev_error`, `endpoint`, `model`, `usage`,
  `safety_latency_ms` in every record, and appends one JSONL line per Jev
  decision to `results/decisions/jev_requests.jsonl` (best-effort, never raises).
- Log entry: timestamp, frame_id, endpoint, model, request_status,
  selected_action, probabilities, confidence, latency_ms, safety_latency_ms,
  fallback, error, safety_status, executed_action. No credentials by
  construction (test asserts a sentinel secret never appears).
- Verified: 10-step loop wrote `sim-00000`/`sim-00005` entries with full key sets.
- Files: `src/decision/policy_interface.py`, `src/decision/jev_decision.py`
  (endpoint+model on all outcomes), `scripts/run_pybullet_closed_loop.py`
  (passes `frame_id`).

## 2/3. Latency split — DONE

Per step: `lidar_ms` (LiDAR), `stages_ms` (perception/stages 1–6),
`policy_latency_ms` (Jev request), `safety_latency_ms` (safety layer, new),
`exec_ms` (action execution, preserved), `loop_ms` (end-to-end total).
Measured in 10-step verification run: safety 0.25–0.28 ms per evaluation
(safety is microseconds-scale; Jev network waits dominate the loop).

## 4. Collision ground-contact filter — DONE and live-verified

- `PyBulletEnv` stores the ground-plane body id at reset; `step()` ignores
  contacts with the plane or with near-vertical normals (|nz| > 0.95),
  reporting them as `ground_contacts_filtered` plus the offending
  `collision_body` for real hits. `collisions_total` now counts solid hits only.
- Root cause confirmed: the 800 kg body sinks 0.72→0.70 m and rests on the
  plane; resting contact was counted as collision.
- Live proof: empty scene ×15 steps → 0 collisions, 4 ground contacts filtered;
  box-ahead drive → first real collision at step 72 with `collision_body` = the
  obstacle id (≠ plane).
- 10-step loop run: **0/10 collisions** (same profile previously read ~60%+).
- Files: `src/simulation/pybullet_env.py` (telemetry only; dynamics untouched).

## 5. results/metrics populated (existing 18:23Z 200-step run, no fabrication)

- `results/metrics/closed_loop_200_step_per_step.csv` (200 rows: per-step
  latencies, actions, confidence, safety, execution, pose).
- `results/metrics/closed_loop_200_step_summary.json` (means/medians/maxima;
  `safety_latency_ms`/model/probabilities/endpoint recorded as **null with an
  explicit `unlogged_in_this_trace` list** — absent, not invented).
- Key numbers: loop mean 223.6 ms; policy mean 756.3 ms; stages mean 61.4 ms;
  LiDAR mean 7.8 ms; exec mean 1.8 ms.

## 6. No fabrication

All nulls declared; collision history annotated as pre-filter artifact.

## 7. Tests — DONE (all green)

- New `tests/robotics/test_instrumentation.py` (5 tests): record key set,
  log-field set + secret-leak sentinel, live-OK path via monkeypatched
  Decisions payload (model/probabilities/endpoint logged), ground-filter
  settle, obstacle-hit detection. Live tests `importorskip` pybullet.
- Results: paradox env 14/14 (with jev file, env-conditional pair deselected);
  default python 3/3 pure. Pre-existing suites unaffected (4/4 jev tests pass
  on default python).

## 8. Honest artifact note

The 10-step verification run overwrote `results/pybullet/closed_loop_trace.json`
(previous content: 18:23Z 200-step run). That run's measurements are preserved
in `results/metrics/closed_loop_200_step_{per_step.csv,summary.json}` and prior
reports; the live trace now reflects the instrumented 10-step run. Next full
200-step run will carry the new fields natively.

## Remaining gaps (not closed here)

1. Live-field end-to-end (probabilities/model in real Jev log lines) still needs
   a keyed run (covered by monkeypatched test only).
2. The 6 scenario configurations were NOT run (per instruction).
3. sklearn 1.9.1 vs 1.8.0 provenance mismatch persists.

Stage 7/8/9: **unchanged** (7 VERIFIED-decision-generation, 8 PARTIAL, 9 PARTIAL).
