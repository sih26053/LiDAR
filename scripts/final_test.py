"""Final untouched test (Phase 16): frozen config, TEST split only.

Reads TEST rows (recorded historical + fresh SQLite bench decisions);
computes accuracy/balanced-accuracy/macro-F1/ECE/Brier/coverage/unsafe/
emergency/STOP/latency. Makes NO decisions, changes NOTHING.
Writes results/laya_final_test.{json,csv,md}.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXEC_TO_CHOICE = {"forward": "forward", "turn_left": "left",
                  "turn_right": "right", "stop": "stop"}


def main() -> dict:
    import scripts.evaluate_dataset as ED
    from src.decision import calibration as cal
    from backend.services import store

    splits = json.loads((PROJECT_ROOT / "results" / "laya_dataset_splits.json"
                         ).read_text())
    recorded = {r["frame_id"]: r for r in json.loads(
        (PROJECT_ROOT / "results" / "laya_recorded_dataset.json"
         ).read_text())["rows"]}
    bench = {r["frame_id"]: r for r in json.loads(
        (PROJECT_ROOT / "results" / "benchmark_labels.json").read_text())["rows"]}
    gate = cal.load_gate_selection()
    con = store._connect()
    rows = []
    for fid in splits["splits"]["test"]:
        if fid in recorded:
            r = recorded[fid]
            rows.append({
                "frame_id": fid, "origin": "live-recorded",
                "category": r["category"],
                "oracle_preferred": r["oracle_preferred"],
                "mode": "constrained_laya",
                "proposed_action": r["proposed_action"],
                "answer_confidence": r["answer_confidence"],
                "latency_ms": r["latency_ms"]})
        elif fid in bench:
            b = bench[fid]
            for d in con.execute(
                    "SELECT * FROM decisions WHERE frame_id=? AND "
                    "source='benchmark-laya' ORDER BY id", (fid,)):
                dd = dict(d)
                if dd.get("mode") not in ("constrained_laya",
                                          "unconstrained_laya"):
                    continue
                prop = EXEC_TO_CHOICE.get(
                    str(dd.get("proposed_action") or ""), None)
                rows.append({
                    "frame_id": fid, "origin": "benchmark-generated",
                    "category": b["category"],
                    "oracle_preferred": b["oracle_preferred"],
                    "mode": dd["mode"], "proposed_action": prop,
                    "answer_confidence": dd["answer_confidence"],
                    "latency_ms": dd["latency_ms"]})
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime()),
              "n": len(rows),
              "n_frames": len({r["frame_id"] for r in rows}),
              "gate_threshold": gate.get("current_gate_threshold"),
              "checkpoint_revision": gate.get("checkpoint_revision"),
              "note": ("Frozen-config report; no decisions made here. Bench "
                       "test decisions predate temperature v2 (n=2 rows used "
                       "v1, identical T=0.25, same revision)."),
              "by_mode": {}}
    for mode in ("constrained_laya", "unconstrained_laya"):
        mrows = [dict(r, unsafe=(
            r["proposed_action"] not in
            (bench.get(r["frame_id"], {}).get("oracle_admissible")
             or recorded.get(r["frame_id"], {}).get("oracle_admissible")
             or [])))
            for r in rows if r["mode"] == mode]
        m = ED.metrics_for(mrows)
        emer_exec_fwd = None
        if mode == "constrained_laya":
            # executed actions for bench rows from replay records
            emer_exec_fwd = 0
            for r in mrows:
                if r["category"] == "F_EMERGENCY":
                    b = bench.get(r["frame_id"])
                    if b and r["proposed_action"] not in (
                            b["oracle_admissible"] or []):
                        emer_exec_fwd += 1
        m["emergency_executed_forward_estimate"] = emer_exec_fwd
        cov_rows = [r for r in mrows if r.get("answer_confidence") is not None]
        thr = gate.get("current_gate_threshold", 0.35)
        m["coverage_at_gate"] = round(sum(
            1 for r in cov_rows if r["answer_confidence"] >= thr)
            / len(cov_rows), 4) if cov_rows else None
        m["stop_rate"] = round(sum(
            1 for r in mrows if r["proposed_action"] == "stop") / len(mrows),
            4) if mrows else None
        report["by_mode"][mode] = m
    (PROJECT_ROOT / "results" / "laya_final_test.json").write_text(
        json.dumps(report, indent=2, default=str))
    with open(PROJECT_ROOT / "results" / "laya_final_test.csv", "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mode", "n", "accuracy", "balanced_accuracy", "macro_f1",
                    "ece", "brier", "coverage_at_gate", "unsafe_rate",
                    "emer_fwd_proposal", "stop_rate", "lat_p50", "lat_p95"])
        for mode, m in report["by_mode"].items():
            w.writerow([mode, m["n"], m["accuracy"], m["balanced_accuracy"],
                        m["macro_f1"], m["ece"], m["brier"],
                        m["coverage_at_gate"], m["unsafe_raw_action_rate"],
                        m["emergency_forward_proposal_rate"], m["stop_rate"],
                        m["latency_p50_ms"], m["latency_p95_ms"]])
    md = ["# Laya final test (untouched TEST split, frozen config)",
          f"Date: {report['generated_utc']}, n={report['n']}",
          f"Gate: {report['gate_threshold']}, "
          f"rev: {report['checkpoint_revision']}", ""]
    for mode, m in report["by_mode"].items():
        md.append(f"## {mode}: acc={m['accuracy']} bal={m['balanced_accuracy']} "
                  f"F1={m['macro_f1']} ece={m['ece']} brier={m['brier']} "
                  f"coverage={m['coverage_at_gate']} "
                  f"unsafe={m['unsafe_raw_action_rate']} "
                  f"emer_fwd={m['emergency_forward_proposal_rate']} "
                  f"stop={m['stop_rate']}")
    (PROJECT_ROOT / "results" / "laya_final_test.md").write_text("\n".join(md))
    print(json.dumps({k: {m: report["by_mode"][m][k] for m in report["by_mode"]}
                      for k in ("accuracy", "unsafe_raw_action_rate",
                                "emergency_forward_proposal_rate")}, indent=2))
    store.close()
    return report


if __name__ == "__main__":
    main()
