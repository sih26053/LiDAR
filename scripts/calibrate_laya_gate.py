"""Calibrate the Laya confidence gate on recorded-frame data.

Pipeline:
    1. Load labelled rows (default: results/laya_navigation_evaluation.json
       constrained_laya rows; --input may point at a compatible JSON).
    2. Deterministic stratified split -> TRAIN/CALIBRATION, VALIDATION,
       TEST (stratified by category; no frame in two splits).
    3. Sweep candidate thresholds on CALIBRATION.
    4. Select the operating point on VALIDATION under the documented
       objective (PRIMARY minimize unsafe accepted; SECONDARY maximize
       coverage; TERTIARY minimize unnecessary STOP).
    5. Report TEST once (no tuning on test).
    6. Fit temperature scaling on the held-out calibration split,
       grouped by (question_type, n_options).
    7. Write models/laya/calibration/gate_selection.json +
       temperature.json and update config/laya_config.json
       confidence_gate.threshold to the selected value.

The selected threshold is a "selected threshold under the defined
validation objective" — NOT claimed optimal (small-n, CPU-only,
single-checkpoint evidence).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXEC_TO_CHOICE = {"forward": "forward", "turn_left": "left",
                  "turn_right": "right", "stop": "stop"}


def load_labelled_rows(eval_path: Path, mode: str = "constrained_laya") -> list:
    payload = json.loads(eval_path.read_text())
    rows = payload["evaluation"]["rows"]
    out = []
    for r in rows:
        if r.get("mode") != mode:
            continue
        if r.get("answer_confidence") is None:
            continue
        prop = r.get("proposed_action")
        admissible = r.get("oracle_admissible") or []
        out.append({
            "frame_id": f"{r['frame_id']}::{r['mode']}",
            "category": r.get("category"),
            "answer_confidence": float(r["answer_confidence"]),
            "probabilities": r.get("probabilities"),
            "oracle_preferred": r.get("oracle_preferred"),
            "oracle_correct": (prop == r.get("oracle_preferred")),
            "unsafe": (prop not in admissible),
            "laya_skipped": bool(r.get("laya_skipped")),
            "n_options": (len(r["probabilities"])
                          if isinstance(r.get("probabilities"), dict) else None),
        })
    return out


def stratified_split(rows: list) -> dict:
    """Deterministic stratified split (~60/20/20, no leakage)."""
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for r in rows:
        groups[r.get("category") or "UNKNOWN"].append(r)
    splits: dict = {"calibration": [], "validation": [], "test": []}
    for cat in sorted(groups):
        members = sorted(groups[cat], key=lambda r: hashlib.sha256(
            str(r["frame_id"]).encode()).hexdigest())
        for i, r in enumerate(members):
            # 3-1-1 round-robin over 5 -> ~60/20/20 even for tiny groups.
            slot = ["calibration", "calibration", "calibration",
                    "validation", "test"][i % 5]
            splits[slot].append(r)
    return splits


def main() -> dict:
    from src.decision import calibration as cal
    from src.decision.checkpoint_info import checkpoint_info

    ap = argparse.ArgumentParser(description="Calibrate the Laya gate")
    ap.add_argument("--input", default="results/laya_navigation_evaluation.json")
    ap.add_argument("--mode", default="constrained_laya",
                    choices=("constrained_laya", "unconstrained_laya"))
    ap.add_argument("--dataset-id", default="laya-nav-balanced-v1")
    ap.add_argument("--apply-config", action="store_true",
                    help="write the selected threshold into laya_config.json")
    args = ap.parse_args()

    rows = load_labelled_rows(PROJECT_ROOT / args.input, args.mode)
    if not rows:
        raise SystemExit("no labelled rows found")
    splits = stratified_split(rows)
    rev = checkpoint_info().get("revision")
    artifacts = {}
    for split_name in ("calibration", "validation", "test"):
        payload = cal.calibrate_gate(
            splits[split_name], split=split_name,
            dataset_id=args.dataset_id, checkpoint_revision=rev)
        artifacts[split_name] = payload
    # Select on VALIDATION (never on test).
    sel = cal.select_threshold(artifacts["validation"]["sweep"])
    selected = sel["threshold"]
    final = {
        "artifact": "gate_selection",
        "version": 1,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_id": args.dataset_id,
        "dataset_hash": artifacts["calibration"]["dataset_hash"],
        "mode_rows": args.mode,
        "n_records": len(rows),
        "splits": {k: {"n": len(splits[k]),
                       "frame_ids": [r["frame_id"] for r in splits[k]]}
                   for k in splits},
        "confidence_metric": "answer_confidence",
        "sweeps": {k: artifacts[k]["sweep"] for k in artifacts},
        "split_metrics": {k: {"ece": artifacts[k]["ece"],
                              "brier": artifacts[k]["brier"]}
                          for k in artifacts},
        "selected_threshold": selected,
        "selection_objective": sel["reason"],
        "selection_split": "validation",
        "zero_unsafe_feasible": sel["zero_unsafe_feasible"],
        "test_report": next(r for r in artifacts["test"]["sweep"]
                            if r["threshold"] == selected),
        "checkpoint_revision": rev,
        "candidate_thresholds": cal.CANDIDATE_THRESHOLDS,
        "note": ("Selected threshold under the defined validation objective "
                 "on a small-n split (n=%d); NOT claimed optimal." % len(rows)),
    }
    cal.save_gate_selection(final)

    # Temperature scaling on the held-out calibration split (model-call rows
    # only; deterministic STOP skips carry no measured distribution).
    temp_rows = [dict(r, oracle_correct=r["oracle_correct"])
                 for r in splits["calibration"] if not r["laya_skipped"]]
    groups: dict = {}
    for r in temp_rows:
        groups.setdefault((r["n_options"] or 4), []).append(r)
    temp_payload = {
        "artifact": "temperature",
        "version": 1,
        "date_utc": final["date_utc"],
        "dataset_id": args.dataset_id,
        "dataset_hash": cal.dataset_hash(temp_rows),
        "split": "calibration",
        "checkpoint_revision": rev,
        "groups": {},
    }
    for (n_opts, grec) in sorted(groups.items()):
        fit = cal.fit_temperature(grec, question_type="choice",
                                  n_options=n_opts)
        temp_payload["groups"][str(n_opts)] = fit
    # Runtime uses one group: the dominant option count; store it flat too.
    dominant = max(groups, key=lambda k: len(groups[k])) if groups else 4
    dom = temp_payload["groups"].get(str(dominant), {})
    temp_payload.update({
        "question_type": "choice",
        "n_options": dominant,
        "temperature": dom.get("temperature", 1.0),
        "fitted": dom.get("fitted", False),
        "ece_before": dom.get("ece_before"),
        "ece_after": dom.get("ece_after"),
        "brier_before": dom.get("brier_before"),
        "brier_after": dom.get("brier_after"),
        "n_records": dom.get("n_records", 0),
    })
    cal.save_temperature(temp_payload)

    if args.apply_config:
        cfg_path = PROJECT_ROOT / "config" / "laya_config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg.setdefault("confidence_gate", {})["threshold"] = selected
        cfg["confidence_gate"]["calibrated_from"] = args.dataset_id
        cfg["confidence_gate"]["calibrated_utc"] = final["date_utc"]
        cfg["confidence_gate"]["artifact"] = \
            "models/laya/calibration/gate_selection.json"
        cfg_path.write_text(json.dumps(cfg, indent=2))

    print(json.dumps({
        "selected_threshold": selected,
        "selection_split": "validation",
        "test_report_at_selected": final["test_report"],
        "split_metrics": final["split_metrics"],
        "temperature": {k: temp_payload.get(k) for k in
                        ("n_options", "temperature", "fitted", "ece_before",
                         "ece_after", "brier_before", "brier_after")},
        "note": final["note"],
    }, indent=2))
    return final


if __name__ == "__main__":
    main()
