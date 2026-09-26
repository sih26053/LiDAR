# Jev Decision-Policy Fix Validation (2026-09-25)

Fix: enriched Decisions state (directional clearances in meters,
nearest obstacle, emergency flag, density semantics note) + rewritten
Choice criteria distinguishing all-unsafe STOP from directional-safe.
13-D contract preserved (byte-identical vectors pre/post);
threshold 0.6 unchanged; safety untouched; Stages 1-6 untouched.

## Phase A — schema/unit (mocked, 0 live calls): PASS
tests/test_jev_enrichment.py: 13-D preserved; enrichment derivation
(fwd 9.37/left-right 7.97 m on recorded vector); meter fallback
(0.5x30=15.0); emergency True/False/None; density note content;
criteria keys + clearance/emergency wording. 7/7 pass.

## Phase B — mocked transport (0 live calls): PASS
forward->forward, left->turn_left, right->turn_right, stop->stop,
conf/probs parsed; bad-choice/missing-conf/out-of-range -> INVALID;
refused connection -> FAILED with no key in error; no key ->
UNAVAILABLE. Included in 226-pass suite.

## Phase C — 13-D correspondence (0 live calls): PASS
probe_controlled_states.py 12/12; all six 13-D vectors identical
pre/post fix (diffed against backup). Proof Stages 1-6 unaffected.

## Phase D — six-state live A/B (6 calls, all OK)
Pre : STOP x6, conf 0.36-0.64, p(stop) 0.52-0.73.
Post: straight FORWARD 1.00 (dx +0.033), obstacle FORWARD 0.99,
left_blocked turn_right 0.66 (dyaw -0.42 deg),
right_blocked turn_left 0.92, both_blocked turn_right 0.80,
emergency STOP 0.99 + OVERRIDE_TO_STOP. All src=jev except none —
all six passed the gate; all through safety; all executed with
measured response. results/pybullet/jev_scenario_probes.json.
Note: post-fix confidences are extreme (up to 1.0) — recorded
service values, not ours; monitoring note, not a defect claim.

## Phase E — small loop (straight_road x30, 6 epochs): COMPLETED
30/30 FORWARD (conf 0.99-1.0, src=jev), 30 SAFE_TO_EXECUTE,
0 collisions, displacement 0.93 m (effective ~1 m/s: each loop
iteration steps physics once at zero velocity then once under
control — pre-existing dual-step design, documented).
15 distinct states from simulator evolution.

## Phase F — six scenarios x20 (24 calls, all OK): ALL COMPLETED
- straight: forward x20, disp 0.61 m
- obstacle_ahead: forward x20, disp 0.61 m, 0 collisions
  (obstacle 10 m out, safety radius 5 m untriggered — model drove
  toward it; safety-bounded, stated plainly)
- left_blocked: turn_right x20 proposed, x15 executed + x5
  fallback-stop (one epoch conf 0.58 < gate), disp 0.22 m
- right_blocked: turn_left x20, disp 0.30 m
- both_blocked: turn_right x20, disp 0.30 m
- emergency_close: stop x20, OVERRIDE_TO_STOP x20, disp 0
Archives: trace_<sid>.json, closed_loop_decisions_<sid>.csv,
pybullet_jev_metrics_<sid>.json, scenario_results.json.

## Override count (verified, §8)
Step-level live overrides: 21 (20 scenario-loop steps + 1 probe),
ALL emergency_close. Decision-level: 5 (4 loop epochs + 1 probe).
All other scenarios/epochs: 0. Earlier "only emergency" (scenario
grain) and "21" (step grain) are consistent.

## Instrumentation (§7)
Every live decision carries run_id/timestamp/frame_id/model/
endpoint/action/probabilities/confidence/latency/safety/executed/
fallback/error (jev_requests.jsonl, decision_store, CSVs). No key,
no Authorization header anywhere (grep-verified; leak tests pass).
