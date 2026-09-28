"""Replay recorded frames through Stages 1-6 + local Laya + safety.

Recorded frame -> load N x 4 -> Stages 1-6 -> named -> enrich ->
eligibility -> LayaDecisionSystem (constrained subset or unconstrained)
-> confidence gate -> safety -> record replay result (source=replay-laya).
The live PyBullet vehicle is NEVER commanded by this script. No cloud
calls, no API keys.

Run:
    python scripts/replay_laya.py --frame-id <FRAME_ID> [--mode constrained|unconstrained]
    python scripts/replay_laya.py --run-id <RUN_ID> [--limit 50] [--mode constrained]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services import replay_laya as rl  # noqa: E402


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Replay Laya on recorded frames")
    ap.add_argument("--frame-id", type=str, default=None)
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--replay-run-id", type=str, default=None)
    ap.add_argument("--semantic-mode", type=str, default="model")
    ap.add_argument("--mode", type=str, default="constrained",
                    choices=("constrained", "unconstrained"),
                    help="constrained=production behaviour (default); "
                         "unconstrained=diagnostics (raw model preference)")
    args = ap.parse_args()
    if args.frame_id:
        out = rl.run_replay_decision(args.frame_id,
                                     args.replay_run_id or "replay-laya-manual",
                                     semantic_mode=args.semantic_mode,
                                     mode=args.mode)
        print(json.dumps(out, indent=2, default=str))
    elif args.run_id:
        out = rl.run_batch(run_id=args.run_id, limit=args.limit,
                           replay_run_id=args.replay_run_id,
                           semantic_mode=args.semantic_mode,
                           mode=args.mode)
        print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=2))
        for r in out["results"]:
            print(f"{r['frame_id']}: {r['source']} mode={r.get('mode')} "
                  f"action={r['action']} conf={r['confidence']} "
                  f"eligible={r.get('eligible_actions')} "
                  f"raw={r.get('raw_laya_action')} safety={r['safety']['status']} "
                  f"err={r['error']}")
        for e in out["errors"]:
            print(f"ERROR {e['frame_id']}: {e['error']}")
    else:
        ap.error("Provide --frame-id or --run-id.")
