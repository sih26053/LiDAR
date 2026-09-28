"""Build the oracle-labelled navigation dataset from RECORDED decisions.

Uses only measured, stored values: recorded proposed_action,
answer_confidence, probabilities, recorded clearances, recorded
emergency flags. Labels (category, oracle admissible/preferred) come
from the deterministic oracle in action_eligibility.py — NEVER from
Laya output. No Laya calls, no simulator runs.

Includes accepted (source=laya) AND gate-rejected (source=fallback,
same checkpoint/revision) rows so the confidence distribution is not
accept-only biased. Historical gate thresholds are preserved per row.

Output: results/laya_recorded_dataset.json (rows + category counts).
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXEC_TO_CHOICE = {"forward": "forward", "turn_left": "left",
                  "turn_right": "right", "stop": "stop"}
PINNED_REV = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"


def main() -> dict:
    from src.decision.action_eligibility import (
        categorize_frame, navigation_oracle)
    from backend.services import store

    con = store._connect()
    rows = con.execute(
        "SELECT d.frame_id, d.source, d.proposed_action, d.confidence, "
        "d.answer_confidence, d.probabilities_json, d.latency_ms, "
        "d.mode, d.eligible_json, d.raw_action, d.constrained_action, "
        "d.checkpoint_revision, d.gate_threshold, f.run_id, f.scenario, "
        "s.forward_clearance_m, s.left_clearance_m, s.right_clearance_m, "
        "s.emergency_flag, s.status AS safety_status "
        "FROM decisions d JOIN frames f ON f.frame_id=d.frame_id "
        "LEFT JOIN safety_events s ON s.frame_id=d.frame_id "
        "AND s.replay_run_id IS NULL "
        "WHERE d.source IN ('laya','fallback') "
        "AND d.checkpoint_revision=? "
        "AND d.answer_confidence IS NOT NULL", (PINNED_REV,)).fetchall()
    labelled, skipped = [], 0
    for r in rows:
        d = dict(r)
        if d["forward_clearance_m"] is None:
            skipped += 1
            continue
        named = {"forward_clearance_m": d["forward_clearance_m"],
                 "left_clearance_m": d["left_clearance_m"],
                 "right_clearance_m": d["right_clearance_m"],
                 "emergency_flag": bool(d["emergency_flag"])}
        safety = {"emergency_stop": bool(d["emergency_flag"])}
        try:
            oracle = navigation_oracle(named, safety)
            category = oracle["category"]
        except (ValueError, TypeError):
            skipped += 1
            continue
        try:
            probs = json.loads(d["probabilities_json"] or "{}")
        except ValueError:
            probs = {}
        prop = EXEC_TO_CHOICE.get(str(d["proposed_action"] or ""), None)
        labelled.append({
            "frame_id": d["frame_id"], "run_id": d["run_id"],
            "scenario": d["scenario"], "origin": "live-recorded",
            "historical_source": d["source"],
            "historical_gate_threshold": d["gate_threshold"],
            "category": category,
            "oracle_admissible": oracle["admissible"],
            "oracle_preferred": oracle["preferred"],
            "oracle_rule": oracle["rule"],
            "proposed_action": prop,
            "answer_confidence": d["answer_confidence"],
            "probabilities": probs,
            "latency_ms": d["latency_ms"],
            "mode": d["mode"],
            "oracle_correct": (prop == oracle["preferred"]),
            "unsafe": (prop not in oracle["admissible"]) if prop else None,
        })
    cats = dict(Counter(r["category"] for r in labelled))
    out = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime()),
           "checkpoint_revision": PINNED_REV,
           "n_labelled": len(labelled), "n_skipped": skipped,
           "categories": cats,
           "note": ("Recorded production-distribution decisions "
                    "(constrained_laya accepts + same-revision fallback "
                    "rejects). Historical gates preserved per row; sweep "
                    "uses raw answer_confidence + oracle labels."),
           "rows": labelled}
    path = PROJECT_ROOT / "results" / "laya_recorded_dataset.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"labelled={len(labelled)} skipped={skipped}")
    print(f"categories={json.dumps(cats)}")
    print(f"wrote {path}")
    store.close()
    return out


if __name__ == "__main__":
    main()
