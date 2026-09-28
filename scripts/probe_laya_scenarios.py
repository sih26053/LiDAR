"""Laya gate-calibration probe (recorded states, NO threshold change).

One live Laya call per configured scenario: records proposed action,
answer_confidence, raw confidence, probabilities, gate outcome at the
configured gate bar (see resolve_gate_config()), safety verdict. Used
to calibrate the Laya gate on evidence instead of copying blindly.
For the full pipeline (eval + sweep + artifacts) prefer
scripts/evaluate_laya_navigation.py + scripts/calibrate_laya_gate.py.

Run:  python scripts/probe_laya_scenarios.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> dict:
    from src.decision.jev_state_adapter import enrich_for_laya, to_named_state
    from src.decision.policy_interface import decide_with_safety, get_policy
    from src.simulation.pybullet_env import PyBulletEnv
    from src.simulation.pybullet_lidar import PyBulletLidar, to_lidar_frame
    from src.simulation.stages_runner import run_stages_1_to_6

    scenarios = json.loads(
        (PROJECT_ROOT / "config" / "pybullet_scenarios.json").read_text())["scenarios"]
    from backend.services import laya_manager

    boot = laya_manager.ensure_started(wait_s=600.0)
    print("laya server:", boot["health"].get("healthy"),
          "reused:" , bool(boot.get("reused")), flush=True)
    policy = get_policy("laya")
    out: dict = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "gate_threshold": policy.min_confidence,
                 "results": []}
    for sc in scenarios:
        env = PyBulletEnv()
        env.connect()
        env.reset(seed=42, scenario={"obstacles": sc["obstacles"]})
        try:
            lidar = PyBulletLidar()
            pts = lidar.to_points(lidar.scan(env))
            frame = to_lidar_frame(pts, f"laya-{sc['id']}", 0.0)
            res = run_stages_1_to_6(frame.points, frame.frame_id, frame.timestamp)
            st = res["state"]
            named = to_named_state(st["state_vector"], st["sector_ranges_m"])
            named = enrich_for_laya(named, st["safety"])
            dec = decide_with_safety(
                policy, named, st["safety"]["nearest_forward_obstacle_m"],
                True, bool(st["safety"]["emergency_stop"]),
                frame_id=frame.frame_id, run_id="laya-calib")
            out["results"].append({
                "scenario": sc["id"],
                "fwd_clear_m": round(min(named["sector_7_range"],
                                         named["sector_0_range"]) * 30.0, 2),
                "laya_status": dec["engine_status"],
                "proposed_action": dec["proposed_action"],
                "answer_confidence": dec["answer_confidence"],
                "raw_confidence": dec["engine_confidence_raw"],
                "probabilities": dec["probabilities"],
                "laya_latency_ms": dec["policy_latency_ms"],
                "gate": dec["gate"],
                "source": dec["source"],
                "safety_status": dec["safety_status"],
                "executed_action": dec["executed_action"],
            })
        finally:
            env.close()
    path = PROJECT_ROOT / "results" / "pybullet" / "laya_gate_calibration.json"
    path.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    out = main()
    for r in out["results"]:
        print(f"{r['scenario']}: status={r['laya_status']} proposed={r['proposed_action']} "
              f"aconf={r['answer_confidence']} raw={r['raw_confidence']} "
              f"probs={json.dumps(r['probabilities'])} gate={r['gate']} src={r['source']}")
