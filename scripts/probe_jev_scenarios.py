"""Phase 5 — Minimal live Jev verification on controlled states.

One live Jev call per configured scenario (cheap: ~6 calls): record
action + confidence + probabilities + safety + executed action +
one-step vehicle response. Reports whatever Jev actually returns;
no expected action is asserted.

Run:  python scripts/probe_jev_scenarios.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> dict:
    from src.decision.jev_state_adapter import enrich_for_jev, to_named_state
    from src.decision.policy_interface import decide_with_safety, get_policy
    from src.simulation.pybullet_action_executor import PyBulletActionExecutor
    from src.simulation.pybullet_env import PyBulletEnv
    from src.simulation.pybullet_lidar import PyBulletLidar, to_lidar_frame
    from src.simulation.stages_runner import run_stages_1_to_6

    scenarios = json.loads(
        (PROJECT_ROOT / "config" / "pybullet_scenarios.json").read_text())["scenarios"]
    policy = get_policy("jev")
    out: dict = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "results": []}
    for sc in scenarios:
        env = PyBulletEnv()
        env.connect()
        env.reset(seed=42, scenario={"obstacles": sc["obstacles"]})
        try:
            lidar = PyBulletLidar()
            pts = lidar.to_points(lidar.scan(env))
            frame = to_lidar_frame(pts, f"jev-{sc['id']}", 0.0)
            res = run_stages_1_to_6(frame.points, frame.frame_id, frame.timestamp)
            st = res["state"]
            named = to_named_state(st["state_vector"], st["sector_ranges_m"])
            named = enrich_for_jev(named, st["safety"])
            dec = decide_with_safety(
                policy, named, st["safety"]["nearest_forward_obstacle_m"],
                True, bool(st["safety"]["emergency_stop"]),
                frame_id=frame.frame_id, run_id="phase5-probe")
            before = env.get_vehicle_state()
            exec_info = PyBulletActionExecutor(env).execute(dec["executed_action"])
            after = env.get_vehicle_state()
            out["results"].append({
                "scenario": sc["id"],
                "fwd_clear_m": round(min(named["sector_7_range"],
                                         named["sector_0_range"]) * 30.0, 2),
                "jev_status": dec["jev_status"],
                "proposed_action": dec["proposed_action"],
                "confidence": dec["confidence"],
                "probabilities": dec["probabilities"],
                "jev_latency_ms": dec["policy_latency_ms"],
                "gate": dec["gate"],
                "source": dec["source"],
                "safety_status": dec["safety_status"],
                "executed_action": dec["executed_action"],
                "pose_before": {k: round(before[k], 4) for k in ("x", "y", "yaw_deg")},
                "pose_after": {k: round(after[k], 4) for k in ("x", "y", "yaw_deg")},
                "speed_after": round(after["speed_ms"], 4),
                "collision": exec_info["collision"],
            })
        finally:
            env.close()
    path = PROJECT_ROOT / "results" / "pybullet" / "jev_scenario_probes.json"
    path.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    out = main()
    for r in out["results"]:
        print(f"{r['scenario']}: jev={r['jev_status']} proposed={r['proposed_action']} "
              f"conf={r['confidence']} probs={json.dumps(r['probabilities'])} "
              f"safety={r['safety_status']} exec={r['executed_action']} "
              f"src={r['source']} dx={round(r['pose_after']['x'] - r['pose_before']['x'], 4)} "
              f"dyaw={round(r['pose_after']['yaw_deg'] - r['pose_before']['yaw_deg'], 3)} "
              f"jev_ms={r['jev_latency_ms']}")
