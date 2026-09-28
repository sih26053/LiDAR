"""DQN evaluation on REAL Stage-6 states (nuScenes replay frames).

No simulator needed: loads an optional checkpoint (else random-init agent),
runs inference on real RL states, records Q-values/action/latency.
Writes results/rl/dqn_evaluation.csv. Run from the project root:

    python scripts/evaluate_dqn.py [--checkpoint models/rl/dqn_weights.pkl]
"""
from __future__ import annotations

import argparse
import csv
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate DQN on real Stage-6 states")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--frames", nargs="*", default=None)
    args = ap.parse_args()

    frames = args.frames or [r["frame_id"] for r in
        list(__import__("csv").DictReader(open(PROJECT_ROOT / "results" / "final" / "final_test_manifest.csv")))]
    cfg = load_final_config()
    nusc = NuScenes(version="v1.0-mini", dataroot=str(PROJECT_ROOT / "data" / "raw" / "nuscenes"), verbose=False)
    ie = ImportanceEngine(weights=dict(cfg["importance_weights"]), max_distance=float(cfg["max_distance_m"]),
                          lambda_uncertainty=float(cfg["uncertainty_lambda"]))
    re_ = ResolutionEngine(levels=[(float(t), float(r)) for t, r in cfg["resolution_levels"]])
    agent = rl_agent.DQNAgent()
    trained = False
    if args.checkpoint:
        meta = agent.load_policy(args.checkpoint)
        trained = bool(meta.get("trained", True))
    import numpy as np
    rows = []
    for fid in frames:
        raw, _ = replay_service.load_frame_points(fid)
        rec = run_pipeline_condition(raw, {"frame_id": fid}, nusc.get("sample", fid), nusc, ie, re_,
            {"condition": "original", "condition_type": "original", "condition_parameter": None, "seed": 42},
            cell_size=float(cfg["integration_cell_size_m"]), failsafe_config=load_failsafe_config())
        if not rec.get("success"):
            rows.append({"frame_id": fid, "status": "pipeline_failed", "error": rec.get("failure_reason")})
            continue
        st = rl_state.build_state(rec["_map_df"].to_dict("records"))
        t0 = time.perf_counter()
        out = agent.act(np.array(st["state_vector"]))
        ims = (time.perf_counter() - t0) * 1000.0
        q = out["q_values"]
        rows.append({"frame_id": fid, "status": "ok", "trained": trained,
                     "q0": q[0], "q1": q[1], "q2": q[2], "q3": q[3],
                     "q_finite": bool(np.isfinite(q).all()),
                     "action": out["action"], "action_valid": out["action"] in rl_agent.ACTIONS,
                     "inference_latency_ms": round(ims, 3)})
        print(f"{fid[:8]}: action={out['action']} q={[round(v,3) for v in q]} {ims:.2f} ms trained={trained}")
    p = PROJECT_ROOT / "results" / "rl" / "dqn_evaluation.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"evaluation: {len(rows)} frames, trained={trained}, saved {p.relative_to(PROJECT_ROOT)}")
    if not trained:
        print("NOTE: random-init network — Q-values demonstrate the data path only, no driving competence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
