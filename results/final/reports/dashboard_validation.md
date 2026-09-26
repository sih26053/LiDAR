# Dashboard Validation (2026-09-25)

## Build & tests
- `npm run build` (frontend/): PASS (also fixed two pre-existing HEAD
  breakages: duplicate `std_mapping_latency_ms`, missing
  `TrackingResult` import).
- `npm test -- --run`: 8 files / 38 tests PASS.

## New live console — `LiveControlPanel`
- JEV AUTONOMOUS MODE: Start Jev Autonomous; displays decision model
  (Jev 1.13), Jev status (LIVE / FALLBACK / UNAVAILABLE / ERROR /
  MANUAL), action, confidence, probabilities, decision latency,
  safety, executed action + source. All from live backend, `—`
  before data exists.
- MANUAL CONTROL (labelled MANUAL TEST, never Jev): Forward / Left /
  Right / Stop -> POST /simulation/action/*, source=manual.
- SIMULATION (PyBullet): vehicle x/y/heading, collision, LiDAR
  points+frame, 2.5D map cells, loop latency, sim time.
- Stream: `/ws/live` preferred, 2 s polling fallback; no secrets in
  payloads (verified: no `Bearer`, no `OPENROUTER_API_KEY`).

## Backend endpoints exercised live (TestClient, .venv-pb)
- GET /system/status 200 (pybullet_importable=true,
  jev_available=false, physical_testing=NOT EXECUTED)
- POST /simulation/live/start (autonomous + manual) / stop / reset
- GET /simulation/state 200 with full snapshot
  (frame live-00669, 182 pts, 55 cells, 13-D rl_state)
- POST /simulation/action/forward -> source=manual,
  SAFE_TO_EXECUTE, executed forward (vehicle moved)
- GET /decision/current|history, /metrics/current 200
- WS /ws/live streams live frames (verified 2 consecutive frames)

## Pre-existing panels preserved
SimulationPanel (offline replay), DecisionPanel, and all replay
panels untouched in behavior; new console is additive.
