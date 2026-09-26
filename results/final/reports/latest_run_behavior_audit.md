# Latest Run Behavior Audit — run-20260925T101257

Date: 2026-09-25. Status labels UNCHANGED per instruction
(Stage 7 BLOCKED, Stage 8/9 PARTIAL). No code was changed in this audit.

## 1. Run identity (from results/pybullet/closed_loop_trace.json)
- run_id: run-20260925T101257, status COMPLETED
- decision_interval_steps: 5
- steps recorded: 50 (steps 0-49, one LiDAR frame each)
- start pose: x=-50.0, y=0.0, z=0.71455, yaw=0.0, speed=0.0
- end pose: x=-50.0000158, y=-0.0000195, z=0.69998, yaw=0.00087,
  speed=0.000012
- displacement: ~0.000026 m (effectively zero; numerical noise)

## 2-4. Steps, epochs, live-request outcomes
- Simulation steps: 50. Jev decision epochs: 10 (steps 0,5,...,45).
- Live Jev requests succeeded: 10/10 (status OK, real network
  latencies 419-838 ms, mean 606 ms). 0 FAILED, 0 UNAVAILABLE.
- check_jev.py now: READY (key_source=config.local_secrets,
  endpoint default, model typesafe/jev-1.13; service echoes model
  typesafe/jev-1.13-20260917, a dated snapshot of the same line).

## 5-7. Every actual Jev action / confidence / probability
All 10 epochs proposed STOP. None proposed FORWARD/LEFT/RIGHT.

| step | proposed | conf | p(stop) | p(fwd) | p(left) | p(right) | gate | source |
|------|----------|------|---------|--------|---------|----------|------|--------|
| 0 | stop | 0.59 | 0.69 | 0.24 | 0.03 | 0.04 | conf<0.6 fallback | fallback |
| 5 | stop | 0.56 | 0.68 | 0.26 | 0.03 | 0.03 | fallback | fallback |
| 10 | stop | 0.52 | 0.64 | 0.30 | 0.03 | 0.03 | fallback | fallback |
| 15 | stop | 0.60 | 0.70 | 0.24 | 0.03 | 0.03 | passed | jev |
| 20 | stop | 0.49 | 0.62 | 0.31 | 0.04 | 0.03 | fallback | fallback |
| 25 | stop | 0.49 | 0.62 | 0.32 | 0.03 | 0.03 | fallback | fallback |
| 30 | stop | 0.56 | 0.66 | 0.27 | 0.04 | 0.03 | fallback | fallback |
| 35 | stop | 0.51 | 0.63 | 0.29 | 0.04 | 0.04 | fallback | fallback |
| 40 | stop | 0.48 | 0.60 | 0.32 | 0.04 | 0.04 | fallback | fallback |
| 45 | stop | 0.50 | 0.63 | 0.30 | 0.03 | 0.04 | fallback | fallback |

- Jev selected FORWARD: 0, LEFT: 0, RIGHT: 0, STOP: 10.
- Confidence range 0.48-0.60 (threshold 0.6, a starting value).

## 8-9. Safety and executed sequence
- Allowed by safety (SAFE_TO_EXECUTE): 50/50 steps.
- Overridden by safety: 0.
- Sent to executor: 50 (all stop).
- Executed-action sequence: stop x 50 (steps 0-49, no variation).

## 10-11. Vehicle commands and 4-action proof
STOP maps to control (0.0 m/s, 0.0 rad/s) by design
(config control table; action_to_velocity). Zero motion is the
CORRECT execution of 50 STOPs, not a rejected command.

Independent executor test (same code path, .venv-pb, seed 42):
- control values: forward=(2.0,0.0), left=(1.0,+0.436 rad/s),
  right=(1.0,-0.436 rad/s), stop=(0.0,0.0)
- FORWARD x60: (-50.000,0.000,yaw 0.00,spd 0.000) ->
  (-48.076,0.000,yaw 0.00,spd 1.914): +1.92 m displacement
- LEFT x60: yaw 0.00 -> 22.34 deg with +0.17 m lateral
- RIGHT x60: yaw 22.34 -> 0.06 deg
- STOP x30: speed 0.925 -> 0.000, position frozen
- collisions: none; pose_before/after recorded every call.
Conclusion: throttle/velocity, steering (yaw-rate), and braking
(zero-velocity) commands all reach the vehicle and actuate.
Vehicle init, constraints, stepping, and interval logic are sound.

## 12. Root cause of ~zero displacement
Chain (each link evidenced above):
1. The sim world is a narrow walled corridor (walls ±6.25 m
   lateral) scanned by a 360° LiDAR. Closest returns (7.8 m) are
   the REAL boundary walls; the median return (17.3 m) is the
   ground ring. This is genuine sensor data, not fabrication.
2. Frozen Stage-6 semantics define occupancy as evidence-presence
   (mapper_2_5d.py: occupancy=1.0 wherever LiDAR evidence falls,
   documented prototype behavior). Hence obstacle_density=1.0 and
   all 8 sectors saturate at the nearest-return distance
   (7.97-9.37 m; nearest forward 12.53 m). The state is VALID per
   the frozen contract (fresh each step, finite, in-bounds;
   16 distinct states over 50 steps) — but it describes a
   fully-observed corridor, not open road.
3. Given density=1.0 and ~8 m clearances, Jev genuinely selects
   STOP (p 0.60-0.70) with modest confidence (0.48-0.60). This is
   a real model decision, consistent with the READY probe (open
   probe state -> FORWARD 0.63; cluttered run state -> STOP).
4. The 0.6 confidence gate converts 9/10 STOPs to fallback-STOP;
   the 10th (conf exactly 0.60) passes as jev-STOP. Either way
   the action is STOP — the gate did not suppress any directional
   motion because Jev proposed none.
5. Safety correctly allows STOP (0 overrides), executor correctly
   applies (0,0) x 50 -> displacement ~0.

Explicit root-cause checklist:
- Jev continuously selected STOP: YES (10/10, genuine decisions).
- Jev unavailable/fallback: NO as outage (10/10 OK); fallback
  source on 45/50 steps is the documented low-confidence path.
- Confidence gating forced STOP: CONTRIBUTING but not decisive
  (gated action would have been STOP anyway).
- Safety overrode directionals: NO (0 overrides; nothing to
  override).
- Executor rejected directionals: NO (none proposed; executor
  proven with all four actions).
- Control values ineffective: NO (measured 1.9 m/s, ±22 deg).
- Vehicle init/constraints/stepping/interval: NO (proven above).
- State invalid/stale: NO (fresh, valid, 16 distinct states).

Primary root cause: decision-level, not mechanical — Jev judges
the evidence-saturated corridor state (density 1.0) as stop-worthy
with sub-threshold confidence, so the vehicle correctly never
receives a motion command.

## 13-16. Code changes
NONE. The executor/vehicle is proven healthy; safety is untouched;
no thresholds were weakened; no random/hardcoded actions added.
The honest remediation path (future work, NOT done here) is
simulator-boundary representation work (e.g. ground-return
handling at the LiDAR/boundary layer, corridor scenario design) —
never rewriting frozen Stages 1-6 or bypassing safety/gate.

## Commands run (.venv-pb, Python 3.11)
- python scripts/check_jev.py -> READY (live_decision OK)
- python scripts/run_pybullet_jev.py --steps 200 (prior run;
  this audit covers run-20260925T101257's artifacts)
- trace/CS
...[truncated 1095 chars]