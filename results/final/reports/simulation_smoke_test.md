# SIMULATION SMOKE TEST — 16 September 2026

Configuration: `results/final/config/final_config.json` (W_BASE+THRESH_C, frozen, unmodified)

Sample: 7-frame final set (`results/final/final_test_manifest.csv`); demo frame
`5991fad3280c4f84b331536c32001a04` (input 34368 → processed 34359, 533 cells,
res 52/429/52/0; per-run wall-clock latency/fps in `simulation_replay_results.csv`,
machine-measured each run).

| Check | Status | Evidence |
|---|---|---|
| Data loading | PASS | 7/7 frames load N×4 finite; `frame_alignment_validation.csv` all aligned |
| Frame alignment | PASS | sample/lidar timestamps match; no off-by-one/stale annotation |
| Preprocessing | PASS | retained 0.998–1.000; empty/NaN inputs raise or are rejected (see `failsafe_integration_checks.csv`) |
| Perception/annotation | PASS | sources `annotation`+`fallback` only; no `model` source; confidence NaN→0.0 placeholder |
| Importance Engine | PASS | all outputs finite in [0, 1]; frozen weights asserted equal |
| Resolution Engine | PASS | all outputs in {0.05, 0.10, 0.20, 0.50}; distribution matches frozen exactly |
| Adaptive Mapper | PASS | `validate_adaptive_map_df` 7/7; no NaN/Inf; cells match frozen (533/759/1132/1257/1059/509/1055) |
| Visualization | PASS | `representative_demo_*.png`, `coordinate_frame_check.png` generated from measured data |
| Benchmark verification | PASS* | cells/importance/resolution MATCH frozen; sim wall-clock latency differs honestly (timing-scope difference, SIM-001 DOCUMENTED); frozen results untouched |
| Handoff execution | PASS | clean `simulation_handoff/` run (see `results/final/handoff_validation.json`) |
| Multi-frame replay | PASS | 7/7 success, 0 failures (`simulation_replay_results.csv`) |

Known issues: SIM-001 (latency timing-scope difference, documented non-blocking);
SIM-002 (driver vocabulary guess, resolved). No blocking integration issue.
