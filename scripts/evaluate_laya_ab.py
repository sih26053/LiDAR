"""A/B comparison on identical recorded frames.

A: unconstrained Laya (raw model preference, diagnostics only).
B: constrained + calibrated Laya (production behaviour).

Same exact frames, both modes, measured side-by-side:
action distribution, unsafe proposals, emergency forward proposals,
unsafe executions, confidence distributions, accepted coverage,
STOP rate, latency, calibration metrics. B is never called "better"
without naming the measured criterion — the numbers speak.

Run:
    python scripts/evaluate_laya_ab.py --input results/laya_navigation_evaluation.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _dist(rows: list, key: str) -> dict:
    d: dict = {}
    for r in rows:
        d[r.get(key)] = d.get(r.get(key), 0) + 1
    return d


def _mean(rows: list, key: str):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def main() -> dict:
    ap = argparse.ArgumentParser(description="Laya A/B comparison")
    ap.add_argument("--input", default="results/laya_navigation_evaluation.json")
    ap.add_argument("--out", default="results/laya_ab_comparison.json")
    args = ap.parse_args()

    payload = json.loads((PROJECT_ROOT / args.input).read_text())
    rows = payload["evaluation"]["rows"]
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime()),
              "input": args.input,
              "criterion_note": ("B is 'better' than A only on a named "
                                 "measured criterion below (e.g. fewer unsafe "
                                 "proposals); no blanket superiority claim."),
              "arms": {}}
    for mode in ("unconstrained_laya", "constrained_laya"):
        mrows = [r for r in rows if r.get("mode") == mode]
        n = len(mrows)
        report["arms"][mode] = {
            "n": n,
            "action_distribution": _dist(mrows, "proposed_action"),
            "executed_distribution": _dist(mrows, "executed_action"),
            "unsafe_proposals": sum(1 for r in mrows if r.get("proposed_unsafe")),
            "unsafe_proposal_rate": (round(sum(1 for r in mrows
                                               if r.get("proposed_unsafe")) / n, 4)
                                     if n else 0.0),
            "emergency_forward_proposals": sum(
                1 for r in mrows if r.get("category") == "F_EMERGENCY"
                and (r.get("raw_laya_action") or r.get("proposed_action"))
                in ("forward", "FORWARD")),
            "unsafe_executions": sum(
                1 for r in mrows if r.get("executed_action") not in
                (r.get("oracle_admissible") or [])),
            "mean_answer_confidence": _mean(mrows, "answer_confidence"),
            # Gate coverage/acceptance lives in
            # models/laya/calibration/gate_selection.json (threshold sweep).
            "stop_rate": (round(_dist(mrows, "executed_action").get("stop", 0)
                                / n, 4) if n else 0.0),
            "mean_policy_latency_ms": _mean(mrows, "policy_latency_ms"),
            "oracle_preferred_accuracy": (
                round(sum(1 for r in mrows if r.get("executed_matches_oracle"))
                      / n, 4) if n else 0.0),
        }
    a, b = (report["arms"]["unconstrained_laya"],
            report["arms"]["constrained_laya"])
    report["delta_B_minus_A"] = {
        "unsafe_proposals": b["unsafe_proposals"] - a["unsafe_proposals"],
        "emergency_forward_proposals": (b["emergency_forward_proposals"]
                                        - a["emergency_forward_proposals"]),
        "unsafe_executions": b["unsafe_executions"] - a["unsafe_executions"],
        "oracle_preferred_accuracy": round(
            b["oracle_preferred_accuracy"] - a["oracle_preferred_accuracy"], 4),
        "stop_rate": round(b["stop_rate"] - a["stop_rate"], 4),
    }
    out = PROJECT_ROOT / args.out
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return report


if __name__ == "__main__":
    main()
