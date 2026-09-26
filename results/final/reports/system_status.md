# SYSTEM STATUS (2026-09-25T11:34:33Z)

Stage 1 (Data Acquisition): PASS
  evidence: replay loader + preprocessing green (216 passed 2026-09-25, .venv-pb)
  test: python -m pytest tests/ backend/tests/ -q
  path: results/final/

Stage 2 (Perception): PASS
  evidence: trained MLP live inference measured per frame (216 passed 2026-09-25)
  test: python -m pytest tests/ backend/tests/ -q
  path: results/segmentation/metrics.json

Stage 3 (Scene Analysis): PASS
  evidence: six factors with provenance; unit-tested (216 passed 2026-09-25)
  test: python -m pytest backend/tests/test_flow.py -q
  path: src/scene_analysis.py

Stage 4 (Adaptive Resolution): PASS
  evidence: frozen tiers + live quadtree stats (216 passed 2026-09-25)
  test: python -m pytest tests/ backend/tests/ -q
  path: src/quadtree.py

Stage 5 (2.5D Semantic Map): PASS
  evidence: diagram fields test-verified; map_validator gate live (216 passed 2026-09-25)
  test: python -m pytest tests/ backend/tests/ -q
  path: src/map_validator.py

Stage 6 (RL/Decision State): PASS
  evidence: 13-dim vector verified at runtime incl. live frames (216 passed 2026-09-25)
  test: python -m pytest tests/test_pybullet_jev_live.py -q
  path: config/rl_state_config.json

Stage 7 (Jev Decision): JEV DECISION VERIFIED
  evidence: live Jev decisions succeeding (choice+confidence)
  test: python scripts/check_jev.py
  path: results/decisions/jev_requests.jsonl

Stage 8 (PyBullet Action Execution): PYBULLET AUTONOMOUS ACTION EXECUTION VERIFIED
  evidence: Jev actions via safety move vehicle
  test: python scripts/run_pybullet_jev.py --steps 200
  path: results/pybullet/

Stage 9 (Closed-Loop Simulation): PYBULLET CLOSED-LOOP SIMULATION VERIFIED
  evidence: full intelligent loop with scenario evidence
  test: python scripts/run_pybullet_jev.py --steps 200
  path: results/metrics/pybullet_jev_metrics.json

Physical Testing: NOT EXECUTED
PyBullet simulation is never described as physical testing.