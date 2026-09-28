# LAYA VALIDATION REPORT — Paradox Protocol (measured evidence only)

All numbers below were executed and observed on this machine (CPU-only,
laya 0.3.20, checkpoint `convaiinnovations/laya/typed-decisions`
@ `55cf4c4e…`). Nothing is invented. Statuses: VERIFIED / MITIGATED /
PARTIALLY VALIDATED / NOT VALIDATED.

## 1. Baseline
- TEST constrained (n=86 rows): accuracy **1.0**, balanced **1.0**, macro F1 **1.0**,
  unsafe **0.0**, emergency-forward proposals **0.0**, ECE 0.34, Brier 0.17.
- TEST unconstrained (n=50): accuracy 0.2, unsafe **0.8**, emergency-forward **1.0**.
- OOD constrained (n=200): accuracy 0.995, unsafe 0.0. OOD unconstrained: unsafe 0.22.
- Latency p50/p95 ≈ 9.3/10.3–13.0 s per Laya call (CPU); loop ≈ 9.4–11.0 s; LiDAR ≈ 3.1–3.3 fps.
- Artifacts: `laya_baseline.{json,csv,md}`. No good/bad verdict — numbers only.

## 2. Dataset coverage
- 45 runs, 47,929 frames inventoried (`laya_dataset_inventory.{json,csv,md}`).
- Labelled: 981 recorded (all D) + 700 generated (A/B/C/E/F 100 each + 200 OOD).
- Splits (seed 20260927, hash `9556324fa45dedb2`): train 1053 / calib 196 /
  valid 146 / test 86 frames / OOD 200. Leakage test PASSED (no frame in two
  splits; no recorded run in two splits). OOD = separate families.

## 3. Category coverage
- Gate (min 50): A 100, B 100, C 100, D 981, E 100, F 100 → all READY.
- E = G_NO_SAFE_ACTION without emergency (diagonal-trap geometry, measured).
- E_BOTH_SIDES_BLOCKED observed as bonus (not required).

## 4. Forward behavior
- Unconstrained: 144/144 inadmissible proposals FORWARD; 66/66 emergency FORWARD
  (rate 1.0). Systematic preference beyond the navigation objective = bias,
  measured (`laya_forward_bias.{json,csv}`).
- Attribution: option-order flip rate **0.0** (12 states) → not option order;
  scenario mix does not explain inadmissible proposals; prompt constrains
  production via subsets; weights unchanged → **model-level behavior, system-level
  mitigated**. Fine-tuning NOT RUN.

## 5. Eligibility mitigation (SYSTEM-LEVEL, not model-level)
- Emergency → [STOP]; blocked directions removed; empty → STOP; single-STOP
  skips inference deterministically. Safety layer remains final.
- Constrained unsafe proposals: **0.0** (all splits incl. OOD).

## 6. Fine-tuning — NOT RUN (PREPARED)
- Installed laya 0.3.20 is inference-only; host CPU-only → training impractical.
- Package `results/laya_finetuning_package/` (1053 oracle-labelled TRAIN examples,
  manifest, config, README) ready for a GPU host. No trained checkpoint exists;
  no model-level claim made. Note: label imbalance (813 forward) documented for
  the future run.

## 7. Calibration
- Metric: answer_confidence. Fit on CALIBRATION only (n=196 constrained).
- Temperature v2: groups n=2,3 DEPLOYED on validation NLL/ECE improvement;
  n=4 REJECTED (NLL worsened — calibration rejected, previous kept for it);
  n=1 N/A. Runtime group: 2-option, T=0.25. Held-out judged (validation_n>0).

## 8. Gate selection
- Grid 0.10–0.90 + fine ±0.02 on VALIDATION (n=146), safety-prioritized objective.
- Outcome: plateau — all thresholds ≤0.40 identical (146/146 correct, 0 unsafe;
  eligibility carries admissibility). Tie-break (documented): retain legacy.
- Legacy 0.35 → validated candidate 0.35, PROMOTED (zero-unsafe validation).
  Optimality NOT claimed.

## 9. Final test (untouched TEST, frozen config)
- `laya_final_test.{json,csv,md}`: constrained acc 1.0, unsafe 0.0,
  coverage@0.35 1.0, STOP rate per data. No tuning on test.

## 10. OOD
- 8 held-out families (widths, sizes, density, pose, noise, dropout, sparse),
  n=200: 1 failure (`ood_wide` D: oracle right vs proposed left — admissible
  preference miss, exact category reported). `laya_ood.{json,csv,md}`.

## 11. Emergency safety
- n=116 emergency states: raw FORWARD 116/116, constrained 0, executed **0**.
  Criterion met → PROCEED. Violation would have stopped deployment.

## 12. Latency
- Cold start 37–52 s; warm 9.3–10.8 s; p95 ≤ 13.0 s; loop ≈ 9.4–11 s;
  3.1–3.3 fps; CPU; backend RSS NOT MEASURED (psutil absent — honest gap).

## 13. Checkpoint
- Pinned `55cf4c4e…`, 5 files, 846 MB, SHA-256 manifest
  (`models/laya/manifest.json`). No floating latest.

## 14. Offline inference — VERIFIED
- Local-path load + typed decision with `HF_HUB_OFFLINE=1` (no Hub requests).

## 15. Server lifecycle — PARTIALLY VALIDATED
- Standalone entry, PID file, health/readiness, reuse-no-duplicates, bounded
  restart, READY/DEGRADED/STARTING/ERROR/STOPPED, `LAYA_ALLOW_SPAWN=0` production.
- In-process survival verified (backend restart, same PID, reconnect).
- Detached-process survival: SANDBOX LIMITATION (sandbox reaps detached
  children) — supervisor configs ready, not validated live.

## 16. Docker — NOT VALIDATED
- `docker-compose.yml` + notes prepared; Docker runtime unavailable on host.
- systemd unit shipped; Windows host → NOT VALIDATED.

## 17. Dashboard
- Live panel shows engine/checkpoint/revision/device/server/PID/uptime/restarts/
  offline/calibration/gate + raw→eligible→constrained→safety→executed chain.
- New Validation section (`/laya/validation`): calibration/test-n/E/temperature/
  gate/fine-tuning/OOD/Docker/offline/hardware badges, evidence-gated.

## 18. Remaining limitations
- Test n=86 frames (136 rows): sufficient for the reported scope, small for
  optimality claims → gate optimality NOT VALIDATED.
- Temperature held-out but small-n (n=30/15); 4-option unscaled.
- Prompt-version uncertainty on the 981 recorded rows (schema identical).
- Fine-tuning, Docker, systemd, detached survival, hardware: NOT VALIDATED
  with exact reasons above. Rollback: promotions registry keeps prior config.
