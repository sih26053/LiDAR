"""MODE A entry point — works on any machine with the repo + nuScenes Mini.

nuScenes replay -> Stages 1-6 (frozen) -> RL state -> untrained-DQN
decision (data path only) -> safety verdict -> recorded action.
No simulator, no training, no hardware. Run from the project root:

    python scripts/run_replay.py [--frames N] [--save]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nuscenes.nuscenes import NuScenes
from backend.services import replay_service
from backend.config import load_final_config, load_failsafe_config
from src.robustness import run_pipeline_condition
from src.importance_engine import ImportanceEngine
from src.resolution_engine import ResolutionEngine
from src import rl_state, rl_agent
from src.action_executor import SimulationActionExecutor


def main() -> int:
    ap = argparse.ArgumentParser(description="Paradox Protocol MODE A: nuScenes replay")
    ap.add_argument("--frames", type=int, default=3, help="frames from the frozen manifest (max available)")
    ap.add_argument("--save", action="store_true", help="write results/rl/replay_decisions.json")
    args = ap.parse_args()

    manifest = list(__import__("csv").DictReader(open(PROJECT_ROOT / "results" / "final" / "final_test_manifest.csv")))
    cfg = load_final_config()
    failsafe = load_failsafe_config()
    nusc = NuScenes(version="v1.0-mini", dataroot=str(PROJECT_ROOT / "data" / "raw" / "nuscenes"), verbose=False)
    ie = ImportanceEngine(weights=dict(cfg["importance_weights"]), max_distance=float(cfg["max_distance_m"]),
                          lambda_uncertainty=float(cfg["uncertainty_lambda"]))
    re_ = ResolutionEngine(levels=[(float(t), float(r)) for t, r in cfg["resolution_levels"]])
    agent = rl_agent.get_agent()
    executor = SimulationActionExecutor()
    out = []
    for row in manifest[: max(1, args.frames)]:
        fid = row["frame_id"]
        raw, _ = replay_service.load_frame_points(fid)
        t0 = time.perf_counter()
        rec = run_pipeline_condition(raw, {"frame_id": fid}, nusc.get("sample", fid), nusc, ie, re_,
            {"condition": "original", "condition_type": "original", "condition_parameter": None, "seed": 42},
            cell_size=float(cfg["integration_cell_size_m"]), failsafe_config=failsafe)
        if not rec.get("success"):
            print(f"{fid[:8]}: PIPELINE FAILED: {rec.get('failure_reason')}")
            return 1
        st = rl_state.build_state(rec["_map_df"].to_dict("records"))
        dec = rl_agent.decide(st["state_vector"], st["safety"])
        exe = executor.execute(dec["network_action"], st["safety"]["nearest_forward_obstacle_m"], map_valid=True)
        out.append({"frame_id": fid, "map_cells": len(rec["_map_df"]),
                    "state_dim": st["state_dim"], "q_values": dec["q_values"],
                    "network_action": dec["network_action"], "final_action": exe["final_action"],
                    "safety_verdict": exe["verdict"], "dqn_trained": dec["trained"],
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1)})
        print(f"{fid[:8]}: cells={len(rec['_map_df'])} action={dec['network_action']} -> {exe['final_action']} ({exe['verdict']})")
    print(f"MODE A COMPLETE: {len(out)} frames, DQN untrained (data path only), no vehicle touched.")
    if args.save:
        p = PROJECT_ROOT / "results" / "rl" / "replay_decisions.json"
        p.write_text(json.dumps(out, indent=2))
        print(f"saved {p.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
