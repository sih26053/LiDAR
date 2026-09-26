# Directional Action Debug — why the closed loop only produces STOP

Date: 2026-09-23. Interpreter: `paradox_pybullet311` (Python 3.11.16).
Rule: no guessing — every verdict below carries measured evidence. Stages 1–6 untouched.

## Phase 1 — one-frame trace (actual values)

Script: `Temp/opencode/trace_frame.py` (repo untouched). Live DIRECT env, seed 42.

- connect: true (client 0) · reset: vehicle id 8, 4 obstacles
- frame_id `debug-frame-001`, LiDAR points **182**
- 13-D state: `[0.3123, 0.2657, 0.2657, 0.3123, 0.3123, 0.2657, 0.2657, 0.3123, 1.0, 0.1475, 0.1803, 0.4562, 0.1688]` (dim 13, all [0,1] — valid, fresh, not stale)
- sector_ranges_m: `[9.37, 7.97, 7.97, 9.37, 9.37, 7.97, 7.97, 9.37]`; nearest forward 12.53 m; emergency_stop false
- structured Jev state: named fields verified (sector_0..7, density, shares, importance, uncertainty)
- Jev endpoint configured: **false** · key set: **false** · model config: `jev`
- Jev response: `status UNAVAILABLE`, proposed **null**, confidence **null**, latency 0.67 ms, error `TYPESAFE_API_KEY not set in environment`
- fallback_reason: `service UNAVAILABLE; safe fallback applies` → candidate `stop`
- safety_decision: `SAFE_TO_EXECUTE`, override_reason `no safety rule fired`, override false
- executed_action: `stop` · control command vx 0.0 m/s, yaw 0.0 rad/s
- vehicle before (−50.0, 0.0, yaw 0.0) → after (−50.0, 0.0, yaw 0.0); collision false (single step)

## Phase 2 — cause determination (12 hypotheses)

1. Jev actually selecting STOP — **NO**. proposed_action null in all 600+ observed decisions (trace + 200-runs + probes).
2. Response parsing failure — **NO** (current shell: nothing to parse; 18:57Z run failed at transport, status FAILED).
3. Invalid action mapping — **NO**. `canonical(0..3)` → forward/turn_left/turn_right/stop verified live.
4. Confidence threshold forcing STOP — **NO**. Threshold (0.6) is evaluated only when status==OK (`policy_interface.py:64-69`); with zero OK responses it influenced 0 decisions.
5. Missing/invalid confidence — **consequence, not cause** (no response exists to carry confidence).
6. Jev timeout — **possible sub-cause of the 18:57Z FAILED run** (policy latency max 8990 ms ≈ 8 s timeout + retry); error text unlogged so unconfirmable — logging gap.
7. Jev API failure / unavailability — **YES, the actual cause.** Current shell: UNAVAILABLE ×200 (no key/endpoint). 18:57Z run: FAILED ×200 (attempted, errored).
8. Safety overriding directionals — **NO**. Matrix: safe forward/left/right → allowed; only forward ≤5.0 m → STOP; unknown → STOP.
9. Executor rejecting directionals — **NO**. All four actions execute live (Phase 4 measurements below).
10. Vehicle control incomplete — **NO**. Forward +1.924 m, left +22.34°, right −22.28°, stop freezes pose (60/60/30-step runs).
11. Stale/invalid RL state — **NO**. Fresh valid 13-D every frame (Phase 1 values).
12. Incorrect state→Jev mapping — **NO**. Adapter output verified against `rl_state_config.json` order; `verify_vector` enforces dim/bounds/dtype.

**Determined cause: #7.** Every downstream stage (fallback, safety, executor) behaves correctly; Jev never returns a decision.

## Phase 3 — controlled Jev cases (actual results, nothing forced)

JevPolicy.decide on crafted valid named states — all four return
`proposed None, confidence None, status UNAVAILABLE`:

- A (fwd 1.0, sides 0.05) → UNAVAILABLE · B (fwd 0.05, left 1.0) → UNAVAILABLE
- C (fwd 0.05, right 1.0) → UNAVAILABLE · D (all 0.03, density 0.9) → UNAVAILABLE

Jev was never reached, so no directional preference could be observed. Cases must be re-run once the service is live.

## Phases 4/5/7 — directional + safety + control evidence (live PyBullet, empty scene)

Safety matrix (12 probes): forward None/20.0/5.01 → allowed; forward 5.0/2.0 → OVERRIDE_TO_STOP with reason; turn_left/right (any distance) → allowed; stop → allowed; unknown `fly` → fail-safe STOP. **Safety never converts safe actions to STOP.**

Live execution (60/60/60/30 steps):
- forward → dx **+1.924 m**, speed 1.914 m/s (≈2.0 config) — motion confirmed
- turn_left → dyaw **+22.34°**, dy +0.171 — turn confirmed
- turn_right → dyaw **−22.28°** — turn confirmed
- stop → speed 0.925→**0.000**, pose frozen — brake confirmed
- unsafe forward (2.0 m) → overridden to stop, vehicle held — override confirmed live

Executor is **not broken**; no fix was needed and none was applied.

## Collision-telemetry anomaly — root-caused

196/200 `collision: true` on a stationary vehicle is a **ground-contact artifact**, not obstacle hits:
probe shows the 800 kg dynamic body sinks under gravity (z 0.72→0.70 over 8 steps;
single early step: 0 contacts) until it rests on the plane (z≈0.70 = half-height),
after which resting contact persists. `getContactPoints(bodyA=vehicle)` counts
ground rest as collision — the detector cannot distinguish plane rest from obstacle
impact. **Do not trust collision metrics until ground contacts (plane body /
vertical normals) are filtered.** No physics code changed in this task (out of scope).

## Phase 8 — closed-loop re-run (19:11:04Z, 200 steps, COMPLETED)

Result: UNAVAILABLE ×200 → stop ×200, 0 confidences, 0 overrides,
displacement 0.0001 m, loop mean 72.9 ms. Directional path is verified and waiting;
autonomy still has no live brain. New LiDAR frames drive new states each iteration
(mechanics confirmed), but decisions cannot vary until Jev responds.

## Remaining blockers

1. Jev credentials/endpoint (STAGE 7 gate).
2. `decide_with_safety` drops Jev `status`/`error` — log them before the next run.
3. Ground-contact filtering for the collision signal.
4. sklearn 1.9.1 vs 1.8.0 provenance mismatch (align to 1.8.0, re-run).
5. Re-run cases A–D + 6 scenario configs once Jev is live; Phase 16 per-action checks are now DONE live (above).
