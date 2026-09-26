# Jev Decision Policy Analysis (2026-09-25, NO live calls made)

Scope: 40 recorded live epochs — 24 scenario-loop epochs with full
13-D states (results/pybullet/trace_*.json), 6 Phase-5 probes with
full vectors joined from results/pybullet/controlled_states.json
(same seed/geometry), 10 prior-run epochs with conf/probs/gate
(full per-epoch vectors overwritten by later runs; exemplar step-0
vector on record). Statuses unchanged (7 BLOCKED, 8/9 PARTIAL).

## 1. Decision table (all 40 proposed STOP; safety/exec = SAFE/STOP
## except emergency OVERRIDE_TO_STOP; 24-epoch detail)

| scenario | epochs | fwd m | left m | right m | dens | conf | p(stop) | p(fwd) | p(l/r) |
|---|---|---|---|---|---|---|---|---|---|
| straight_road | 4 | 9.4 | 7.6-8.0 | 7.6-8.0 | 1.00 | 0.50-0.63 | 0.62-0.72 | 0.22-0.30 | ≤0.04 |
| obstacle_ahead | 4 | 8.5 | 7.6-7.8 | 7.6-7.8 | 1.00 | 0.35-0.55 | 0.51-0.66 | 0.23-0.36 | ≤0.08 |
| left_blocked | 4 | 4.6-5.7 | 4.6-5.4 | 7.6-8.0 | 1.00 | 0.43-0.56 | 0.57-0.67 | 0.21-0.28 | ≤0.13 |
| right_blocked | 4 | 4.6-5.7 | 7.6-8.0 | 4.6-5.4 | 1.00 | 0.33-0.47 | 0.49-0.60 | 0.28-0.35 | ≤0.10 |
| both_blocked | 4 | 4.6 | 4.6-5.4 | 4.6-5.4 | 1.00 | 0.27-0.45 | 0.46-0.59 | 0.27-0.30 | ≤0.15 |
| emergency_close | 4 | 2.5 | 7.6-7.8 | 7.6-7.8 | 1.00 | 0.33-0.44 | 0.50-0.58 | 0.19-0.23 | ≤0.21 |

Probes (1 epoch each): straight 0.60, obstacle 0.64, left 0.46,
right 0.37, both 0.48, emergency 0.36 — all STOP.
Prior 10: all STOP, conf 0.48-0.60, p(stop) 0.60-0.70.
Importance 0.45-0.49 and uncertainty 0.14-0.18 are near-constant
across all 40 (no discrimination). Gate passed only 3/40
(conf exactly ≥0.60); safety overrode only the emergency state.

## 2. Scenario comparison
- Most-open state (straight, fwd 9.4 m) gets the HIGHEST p(stop)
  (0.72) — inverted from a clearance-following policy.
- Most-blocked (both_blocked) gets the LOWEST confidence (0.27)
  but still STOP, not a directional escape.
- Emergency (fwd 2.5 m) still assigns p(forward) 0.19-0.23 and
  p(left) up to 0.21 — no sharp all-unsafe separation.
- p(forward) 0.19-0.36 in EVERY scenario including both_blocked
  and emergency; p(left)/p(right) rarely exceed 0.11.
Response surface is flat and diffuse: STOP always wins at 0.5-0.7
regardless of geometry. The model reacts (conf/probs shift with
scenario) but never commits to direction.

## 3. Features correlating with STOP
1. obstacle_density = 1.00 in 40/40 (zero variance). The model
   likely reads this as "surrounded" — it cannot discriminate
   because the value never varies (frozen evidence-presence
   semantics, identical on replay).
2. Clearances 2.5-9.4 m presented only as 8 raw sector fractions;
   no explicit forward/left/right clearance is sent (the
   adapter's decision_criteria is computed but NOT included in
   the Decisions `state`).
3. importance/uncertainty near-constant (no signal).

## 4. Instructions/criteria inspection (config/jev_config.json)
- Question: "Which vehicle action should be selected based only on
  the supplied state?" Criteria: forward="Forward movement is safe
  and useful." left/right="Turning X is the safest useful action."
  stop="Stopping is the safest appropriate action."
- Top instructions: "Prefer the freest clearance; stop when forward
  clearance is low or uncertainty is high."
- VERDICT: STOP is defined too broadly and asymmetrically. Every
  directional criterion contains a "safe...useful" double condition
  while stop needs only "appropriate"; nothing defines "all movement
  unsafe" vs "one safe directional action available"; no clearance
  semantics (what counts as low?); density=1.0 is unexplained, so a
  full-coverage scan reads as fully blocked. A risk-averse model
  satisfies these instructions with STOP on almost any input — which
  is exactly the recorded behavior.

## 5. Proposed minimal change (input/criteria ONLY)
A. Enrich the structured `state` in the Jev adapter layer
   (src/decision/jev_client or jev_state_adapter presentation —
   13-D contract untouched, no Stage 1-6 change):
   forward/left/right/rear clearance in METERS + normalized
   (already computed by decision_criteria), nearest-obstacle
   distance, Stage-6 emergency_stop flag, and a one-line semantic
   note: obstacle_density is LiDAR-coverage evidence-presence,
   NOT a blockage fraction.
B. Sharpen criteria text (config/jev_config.json question +
   instructions, same Choice structure): define stop as "no
   direction offers safe clearance" and each direction as "that
   direction offers the largest safe clearance"; state that
   density≈1.0 is normal full-scan coverage.
C. Change NOTHING else: threshold stays 0.6, safety layer
   untouched and still downstream of every proposal, no hardcoded
   actions, no new features beyond documented derivations.

Why this does not bypass safety: it only alters the PROPOSAL
distribution. Every proposal still passes the confidence gate and
the independent safety layer (which already proved it overrides:
emergency OVERRIDE_TO_STOP x21). A sharper proposal cannot reach
the vehicle unexamined.

## 6. Required validation (in order; stop if any step fails)
1. Unit: enriched-state schema test (keys, [0,1] bounds, meters
   consistency, no raw cloud, no secrets).
2. Mocked-transport test: assert new state shape is what is POSTed;
   request still {model, state, questions}, model pinned.
3. Correspondence re-run: probe_controlled_states (13-D level
   identical — proof Stages 1-6 unaffected).
4. Live Jev A/B (first new calls): same 6 controlled states,
   compare p(stop)/conf before/after. Expect directional mass to
   MOVE with geometry (not necessarily flip).
5. Small closed loop on straight_road only.
6. Full scenarios only if directional proposals appear AND pass
   safety. No status change until Jev-directed motion is measured.
Note: jev_config.json comment still says key comes ONLY from
TYPESAFE_API_KEY env — stale since local_secrets support; correct
the comment when the criteria are edited.

## ROOT CAUSE
STOP-dominance is a decision-policy specification failure, not a
sensor, state, safety, or actuator failure: (a) the single most
salient state feature (density=1.0, invariant) reads as
"surrounded"; (b) clearances arrive as eight unexplained fractions
with no directional aggregation; (c) the criteria make STOP the
easiest choice to justify ("appropriate" vs "safe AND useful").
Recorded response surface (STOP 0.5-0.7 on every geometry,
inverted clearance ordering) matches this mechanism.

## PROPOSED MINIMAL CHANGE
Enrich the Decisions `state` with explicit directional clearances
(meters), nearest-obstacle distance, emergency flag, and a density
semantic note; rewrite criteria to define all-unsafe vs
directional-safe. 13-D contract, threshold (0.6), safety layer,
and Stages 1-6 all preserved. NOT IMPLEMENTED — awaiting approval.

## REQUIRED VALIDATION
Schema unit test -> mocked transport test -> 13-D correspondence
re-run -> 6-state live A/B -> small loop -> full scenarios only on
directional evidence. Statuses stay until Jev-directed motion moves
the vehicle through safety.
