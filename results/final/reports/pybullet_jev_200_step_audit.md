# PyBullet + Jev 200-step post-run evidence audit

Run audited: `python scripts/run_pybullet_closed_loop.py --steps 200`
Trace: `started_utc 2026-09-23T18:57:11Z`, `status COMPLETED`, 200/200 steps recorded.
Auditor note: process exit success is NOT stage success. Findings below are measured only.

## 0. scikit-learn version alignment (answer first)

Runtime is scikit-learn **1.9.1**; saved estimators (LabelBinarizer, MLPClassifier,
StandardScaler) were pickled under **1.8.0** → `InconsistentVersionWarning` on every
frame, no hard error, pipeline outputs structurally valid (13-D, dim 13).
**Recommendation: align the environment to scikit-learn==1.8.0 before FINAL
validation** (matches training provenance, removes the inference-drift caveat;
1.8.x supports numpy 2.x so the numpy==2.4.4 repair is unaffected). Alternative:
retrain under 1.9.1 and re-freeze artifacts. Either way, re-run the 200-step loop
after alignment so final metrics carry no version warning. This audit proceeds on
the 1.9.1 run as-is, with the caveat recorded — nothing re-fabricated.

## 1. Files inspected (run outputs)

- `results/pybullet/closed_loop_trace.json` (252,855 bytes, written 24-09-2026 00:28 local) — full step records.
- `results/decisions/closed_loop_decisions.csv` (6,199 bytes, same timestamp) — per-step summary rows.
- `results/metrics/` — **empty** (no metric files produced by this run).
- `results/decisions/` — **no per-request Jev logs** despite `jev_config.json` logging block (`log_state/log_response`, `log_dir results/decisions`); `JevDecisionSystem` never writes request logs (logging gap).
- Screenshots: **none**. Dashboard screenshots: **none**. Scenario result files: **none** (`config/pybullet_scenarios.json` scenarios remain NOT EXECUTED).

## 2. Commands inspected / run (audit only, no pipeline edits)

- `python scripts/run_pybullet_closed_loop.py --steps 200` (the audited run; not re-run by auditor).
- Read-only probes of the trace/CSV above; `TYPESAFE_API_KEY`/`JEV_ENDPOINT` presence checks (both currently **absent**).
- No Stages 1–6 code altered during this audit.

## 3. Actual execution path (link-by-link verdict)

| Link | Verdict | Evidence |
|---|---|---|
| PyBullet env/reset/step | EXECUTED | 200 steps, `execution.frame` 2→400, fixed-step DIRECT |
| Simulated LiDAR (rayTest) | EXECUTED | `points: 182` every frame, `lidar_ms` mean 6.1 ms |
| Stages 1–6 → 13-D state | EXECUTED | `stages_ms` mean ~63 ms/step (step 0: 4535 ms incl. model load); decision chain consumed state each step |
| Jev call | ATTEMPTED, 0 success | gate `service FAILED; safe fallback applies` ×200; `policy_latency_ms` mean 1166.9, max 8990.4 (real network attempts, not instant UNAVAILABLE) |
| Action + confidence | NOT RETURNED | `proposed_action: null` ×200, non-null confidence 0/200 |
| Safety layer | EXECUTED, unchallenged | `SAFE_TO_EXECUTE` ×200, overrides 0, stale-overrides 0 — evaluated on fallback `stop` only |
| Action executor → vehicle | EXECUTED (stop only) | 200 `stop` executions (+1 physics-advance step each → frame 400); `pose_before/after` recorded |
| Next frame from new LiDAR | MECHANICS RUN, no variation | frames all 182 pts; vehicle stationary (displacement 0.0001 m) so frames are near-identical; decisions never vary |

## 4. Measured metrics (200-step run)

- simulation steps: 200 loop iterations (400 physics steps: 1 advance + 1 execute per iteration)
- LiDAR frames: 200, all 182 points (zero variance)
- Jev HTTP attempts: gate FAILED ×200 steps; actual request count ≈ 40 (decision at ~every 5th step per 2 Hz policy; **estimate**, per-step call flag not logged)
- successful Jev decisions: **0**; confidences returned: **0**
- decision (policy) latency: mean 1166.9 ms, max 8990.4 ms (failed-request latency, not Jev inference)
- safety overrides: 0; safety verdicts: SAFE_TO_EXECUTE 200/200
- executed actions: `stop` 200/200 (`forward/left/right` 0)
- vehicle movement: displacement 0.0001 m (start −50.0,0.0 → end −50.0,−0.0); speed 0 throughout
- collisions: `collision: true` on **196/200** steps with a stationary vehicle — **suspect artifact** (likely contact-threshold reporting on the resting pose), NOT usable as a metric until root-caused
- loop latency: mean 290.3 ms, median 32.3 ms, max 9021.2 ms (Jev-failure waits dominate the mean)
- scenarios: 0/6 executed; scenario status: NOT EXECUTED

## 5. Jev evidence

- Backend used: **Jev policy** (`decision_model: jev` ×200) — NOT DQN, NOT random. But every call failed at the service layer.
- Credential discrepancy (recorded, unresolved): at run time the client proceeded past the availability probe (FAILED ≠ UNAVAILABLE implies key+endpoint were set in the run shell); at audit time both vars are absent. The underlying error text (auth/timeout/network) is **not propagated** into the trace (`decide_with_safety` drops `status`/`error`) — logging gap to fix.
- Net: Jev was genuinely attempted and genuinely produced nothing usable.

## 6. Safety evidence

- Safety layer evaluated every step (`safety_status`, `safety_reason` recorded ×200).
- It never overrode because the only candidate ever presented was fallback `stop` (`no safety rule fired`).
- The layer's ability to override a *risky Jev proposal* remains **untested** (no live Jev proposal has ever arrived).

## 7. Action-execution evidence

- Executor applied `stop` to the live vehicle 200× with measured before/after pose (all identical — consistent with zero velocity).
- FORWARD/LEFT/RIGHT executor paths remain **unexecuted** in PyBullet (pure-function mapping tested, live motion not).
- Per-action independent verification (Phase 16) is still outstanding.

## 8. Remaining blockers

1. Jev service failures (error text unlogged; credentials currently absent) → no successful decision exists.
2. `decide_with_safety` drops Jev `status`/`error` — add them to the trace before the next run.
3. Collision-signal anomaly (196/200 on a parked vehicle) — root-cause before trusting it.
4. `results/metrics/` empty; no per-request Jev logs; no screenshots; no scenario runs.
5. sklearn 1.9.1 vs 1.8.0 provenance mismatch (see §0).
6. Stages 1–6 untouched by this audit; Stage 1–6 PASS standing is not affected by the above.

## 9. Honest stage statuses

- **Stage 7 (Jev Decision): BLOCKED.** Loop attempts Jev and fails 100%; zero actions and zero confidences returned. Downgrade-proof: exit COMPLETED changes nothing.
- **Stage 8 (PyBullet Action Execution): PARTIAL.** Executor↔vehicle path is live and instrumented, but only `stop` has ever been executed; directional actions unverified; collision telemetry suspect.
- **Stage 9 (Closed-Loop Simulation): PARTIAL (degraded loop).** Sensor→stages→state→safety→executor→vehicle mechanics completed 200 steps, but autonomy ran entirely on safe-stop fallback with identical frames — no intelligent driving and no scenario completion demonstrated.

Physical testing: NOT EXECUTED.
