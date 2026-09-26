"""Phase 7 — Run the six configured scenarios (small loops, live Jev).

Per scenario: 20 steps, Jev every 5 steps (4 epochs). Archives that
run's trace/metrics/decisions/jev-log slice under results/pybullet/
and results/metrics/ with the scenario id, then writes
results/pybullet/scenario_results.json and marks scenario statuses
in config/pybullet_scenarios.json from the measured outcomes.

Screenshots/video: NOT AVAILABLE (headless DIRECT mode).

Run:  python scripts/run_scenarios.py
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

STEPS = 20
INTERVAL = 5


def main() -> dict:
    from scripts.run_pybullet_closed_loop import main as run_loop

    cfg_path = PROJECT_ROOT / "config" / "pybullet_scenarios.json"
    cfg = json.loads(cfg_path.read_text())
    summary: dict = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "steps_per_scenario": STEPS,
                     "interval_steps": INTERVAL,
                     "scenarios": {}}
    for sc in cfg["scenarios"]:
        sid = sc["id"]
        trace = run_loop(steps=STEPS, gui=False,
                         decision_interval_steps=INTERVAL,
                         run_id=f"scenario-{sid}",
                         scenario_id=sid)
        steps = trace.get("steps", [])
        rec = {
            "status": trace.get("status"),
            "steps_recorded": len(steps),
            "proposed": dict(Counter(str(s["decision"]["proposed_action"]) for s in steps)),
            "confidence": sorted({s["decision"]["confidence"] for s in steps
                                  if s["decision"].get("jev_called_this_step")}),
            "sources": dict(Counter(str(s["decision"]["source"]) for s in steps)),
            "safety": dict(Counter(str(s["decision"]["step_safety_verdict"]) for s in steps)),
            "executed": dict(Counter(str(s["decision"]["executed_action"]) for s in steps)),
            "collisions": sum(1 for s in steps if s["execution"]["collision"]),
            "displacement_m": _disp(trace),
            "heading_change_deg": _dhead(trace),
            "loop_latency_ms_mean": _mean([s.get("loop_ms") for s in steps]),
        }
        summary["scenarios"][sid] = rec
        _archive(sid)
        sc["status"] = ("executed" if trace.get("status") == "COMPLETED"
                        else f"{trace.get('status')}")
    (PROJECT_ROOT / "results" / "pybullet" / "scenario_results.json").write_text(
        json.dumps(summary, indent=2))
    cfg_path.write_text(json.dumps(cfg, indent=2))
    return summary


def _disp(trace: dict) -> float | str:
    try:
        a, b = trace["start_pose"], trace["end_pose"]
        return round(((b["x"] - a["x"]) ** 2 + (b["y"] - a["y"]) ** 2) ** 0.5, 4)
    except (KeyError, TypeError):
        return "NOT_AVAILABLE"


def _dhead(trace: dict) -> float | str:
    try:
        return round(trace["end_pose"]["yaw_deg"] - trace["start_pose"]["yaw_deg"], 4)
    except (KeyError, TypeError):
        return "NOT_AVAILABLE"


def _mean(xs) -> float | str:
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 2) if xs else "NOT_AVAILABLE"


def _archive(sid: str) -> None:
    pairs = [
        (PROJECT_ROOT / "results" / "pybullet" / "closed_loop_trace.json",
         PROJECT_ROOT / "results" / "pybullet" / f"trace_{sid}.json"),
        (PROJECT_ROOT / "results" / "decisions" / "closed_loop_decisions.csv",
         PROJECT_ROOT / "results" / "decisions" / f"closed_loop_decisions_{sid}.csv"),
        (PROJECT_ROOT / "results" / "metrics" / "pybullet_jev_metrics.json",
         PROJECT_ROOT / "results" / "metrics" / f"pybullet_jev_metrics_{sid}.json"),
    ]
    for src, dst in pairs:
        if src.is_file():
            shutil.copyfile(src, dst)


if __name__ == "__main__":
    summary = main()
    for sid, rec in summary["scenarios"].items():
        print(f"{sid}: {rec['status']} proposed={rec['proposed']} conf={rec['confidence']} "
              f"exec={rec['executed']} safety={rec['safety']} disp={rec['displacement_m']}m "
              f"coll={rec['collisions']}")
