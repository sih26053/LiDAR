"""Replay recorded frames through Stages 1-6 + Jev + safety (no vehicle).

Recorded frame -> load N x 4 -> Stages 1-6 -> named -> enrich ->
JevDecisionSystem -> confidence gate -> safety -> record replay
result (source=replay-jev). The live PyBullet vehicle is NEVER
commanded by this script.

Run:
    python scripts/replay_jev.py --frame-id <FRAME_ID>
    python scripts/replay_jev.py --run-id <RUN_ID> [--limit 50]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services import replay_jev as rj  # noqa: E402


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Replay Jev on recorded frames")
    ap.add_argument("--frame-id", type=str, default=None)
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--replay-run-id", type=str, default=None)
    ap.add_argument("--semantic-mode", type=str, default="model")
    args = ap.parse_args()
    if args.frame_id:
        out = rj.run_replay_decision(args.frame_id,
                                     args.replay_run_id or "replay-manual",
                                     semantic_mode=args.semantic_mode)
        print(json.dumps(out, indent=2, default=str))
    elif args.run_id:
        out = rj.run_batch(run_id=args.run_id, limit=args.limit,
                           replay_run_id=args.replay_run_id,
                           semantic_mode=args.semantic_mode)
        print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=2))
        for r in out["results"]:
            print(f"{r['frame_id']}: {r['source']} {r['action']} conf={r['confidence']} "
                  f"safety={r['safety']['status']} err={r['error']}")
        for e in out["errors"]:
            print(f"ERROR {e['frame_id']}: {e['error']}")
    else:
        ap.error("Provide --frame-id or --run-id.")
