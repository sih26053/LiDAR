"""Canonical PyBullet + Jev closed-loop runner (GAP 6).

Flow per loop step:
    PyBullet -> simulated LiDAR -> LiDARFrame -> Stages 1-6 ->
    13-D decision state -> named Jev state -> Jev (action + confidence) ->
    confidence gate -> safety layer -> PyBullet action executor ->
    vehicle -> next frame.

Reuses scripts/run_pybullet_closed_loop.py (no duplicated pipeline).

Run:
    python scripts/run_pybullet_jev.py --steps 200
    python scripts/run_pybullet_jev.py --steps 200 --decision-interval-steps 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_pybullet_closed_loop import main  # noqa: E402


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PyBullet + Jev autonomous closed loop")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--decision-interval-steps", type=int, default=None,
                    help="Jev call interval in loop steps "
                         "(default: from jev_config decision_frequency_hz)")
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--scenario", type=str, default=None,
                    help="scenario id from config/pybullet_scenarios.json")
    args = ap.parse_args()
    t0 = time.perf_counter()
    trace = main(args.steps, args.gui, args.decision_interval_steps,
                 args.run_id, args.scenario)
    summary = {k: v for k, v in trace.items() if k != "steps"}
    summary["wall_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    print(json.dumps(summary, indent=2))
