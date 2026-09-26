"""Phase 4 — Controlled simulator states (NO Jev calls).

For each scenario in config/pybullet_scenarios.json: reset the live env,
capture LiDAR -> LiDARFrame -> Stages 1-6 -> 13-D state -> named Jev
state, and check the values correspond to the scenario geometry.
Saves results/pybullet/controlled_states.json.

Run:  python scripts/probe_controlled_states.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> dict:
    from src.decision.jev_state_adapter import decision_criteria, to_named_state
    from src.simulation.pybullet_env import PyBulletEnv
    from src.simulation.pybullet_lidar import PyBulletLidar, to_lidar_frame
    from src.simulation.stages_runner import run_stages_1_to_6

    scenarios = json.loads(
        (PROJECT_ROOT / "config" / "pybullet_scenarios.json").read_text())["scenarios"]
    env = PyBulletEnv()
    env.connect()
    out: dict = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "states": {}, "checks": []}
    try:
        for sc in scenarios:
            sid = sc["id"]
            env.reset(seed=42, scenario={"obstacles": sc["obstacles"]})
            lidar = PyBulletLidar()
            pts = lidar.to_points(lidar.scan(env))
            frame = to_lidar_frame(pts, f"controlled-{sid}", 0.0)
            res = run_stages_1_to_6(frame.points, frame.frame_id, frame.timestamp)
            st = res["state"]
            named = to_named_state(st["state_vector"], st["sector_ranges_m"])
            crit = decision_criteria(named)
            rec = {"lidar_points": int(pts.shape[0]),
                   "map_cells": len(res["result"].get("map_cells", [])),
                   "state_vector": st["state_vector"],
                   "sector_ranges_m": st["sector_ranges_m"],
                   "named_state": named, "criteria": crit,
                   "nearest_forward_obstacle_m": st["safety"]["nearest_forward_obstacle_m"],
                   "emergency_stop": st["safety"]["emergency_stop"]}
            out["states"][sid] = rec
            out["checks"].extend(_check(sid, rec))
    finally:
        env.close()
    out["all_checks_passed"] = all(c["passed"] for c in out["checks"])
    path = PROJECT_ROOT / "results" / "pybullet" / "controlled_states.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return out


def _check(sid: str, rec: dict) -> list:
    crit = rec["criteria"]
    checks = []

    def add(name, passed, detail):
        checks.append({"scenario": sid, "check": name,
                       "passed": bool(passed), "detail": detail})

    fwd = crit["forward_clearance"] * 30.0
    left = crit["left_clearance"] * 30.0
    right = crit["right_clearance"] * 30.0
    nf = rec["nearest_forward_obstacle_m"]
    add("state_valid", len(rec["state_vector"]) == 13
        and all(0.0 <= v <= 1.0 for v in rec["state_vector"]),
        f"13-D in [0,1], cells={rec['map_cells']}")
    if sid == "straight_road":
        add("forward_open", nf is not None and nf > 10.0,
            f"nearest_fwd={nf}, fwd_clear={fwd:.1f}m")
    elif sid == "obstacle_ahead":
        add("forward_blocked", nf is not None and nf < 12.0,
            f"nearest_fwd={nf} (obstacle 10 m ahead)")
    elif sid == "left_blocked":
        add("left_tighter_than_right", left < right,
            f"left={left:.1f}m right={right:.1f}m")
    elif sid == "right_blocked":
        add("right_tighter_than_left", right < left,
            f"right={right:.1f}m left={left:.1f}m")
    elif sid == "both_blocked":
        add("both_sides_tight", left < 12.0 and right < 12.0,
            f"left={left:.1f}m right={right:.1f}m")
    elif sid == "emergency_close":
        add("emergency_flagged", rec["emergency_stop"] is True,
            f"nearest_fwd={nf} (obstacle ~3.5 m ahead)")
    return checks


if __name__ == "__main__":
    out = main()
    for c in out["checks"]:
        print(("PASS" if c["passed"] else "FAIL"), c["scenario"], c["check"], "-", c["detail"])
    print("ALL PASS" if out["all_checks_passed"] else "SOME CHECKS FAILED")
