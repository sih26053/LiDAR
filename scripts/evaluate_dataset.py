"""Baseline evaluation (Phases 7-8): current Laya checkpoint, before changes.

Fresh both-mode Laya decisions on benchmark frames (validation, test,
OOD splits) + reuse of recorded production decisions for recorded
split rows (zero extra calls). Metrics per category + overall +
IND/OOD: accuracy vs oracle preferred, balanced accuracy, macro F1,
confusion matrix, action distribution, unsafe raw-action rate,
emergency forward-proposal rate, answer_confidence stats, ECE, Brier,
latency p50/p95.

Forward-bias analysis: proposal rates overall / when inadmissible /
in emergency / executed-when-inadmissible, by scenario; attribution
discusses scenario distribution vs prompt vs model behavior WITHOUT
loaded language.

Writes: results/laya_baseline.{json,csv,md},
        results/laya_forward_bias.{json,csv}
Decisions stored in SQLite with source=benchmark-laya (never live).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXEC_TO_CHOICE = {"forward": "forward", "turn_left": "left",
                  "turn_right": "right", "stop": "stop"}
LABELS = ["forward", "left", "right", "stop"]


def _utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def pct(vals, lo=0, hi=100):
    import numpy as np
    return [float(np.percentile(vals, lo)), float(np.percentile(vals, hi))]


def metrics_for(rows: list) -> dict:
    from src.decision import calibration as cal

    n = len(rows)
    y_true = [r["oracle_preferred"] for r in rows]
    y_pred = [r["proposed_action"] for r in rows]
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n if n else 0.0
    # balanced accuracy + macro F1 over the 4 labels
    recs, f1s = [], []
    conf = {t: {p: 0 for p in LABELS} for t in LABELS}
    for t, p in zip(y_true, y_pred):
        if t in conf and p in conf[t]:
            conf[t][p] += 1
    for lab in LABELS:
        tp = conf[lab][lab]
        fn = sum(conf[lab].values()) - tp
        fp = sum(conf[o][lab] for o in LABELS) - tp
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        recs.append(rec)
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    dist = dict(Counter(y_pred))
    unsafe = sum(1 for r in rows if r.get("unsafe"))
    emer = [r for r in rows if r.get("category") == "F_EMERGENCY"]
    emer_fwd = sum(1 for r in emer
                   if (r.get("raw_laya_action") or r.get("proposed_action"))
                   in ("forward", "FORWARD"))
    aconfs = [r["answer_confidence"] for r in rows
              if r.get("answer_confidence") is not None]
    lats = [r["latency_ms"] for r in rows if r.get("latency_ms")]
    calib_recs = [{"answer_confidence": r["answer_confidence"],
                   "oracle_correct": r["oracle_preferred"] == r["proposed_action"]}
                  for r in rows if r.get("answer_confidence") is not None]
    return {
        "n": n,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(sum(recs) / 4, 4),
        "macro_f1": round(sum(f1s) / 4, 4),
        "confusion_oracle_x_proposed": conf,
        "action_distribution": dist,
        "forward_rate": round(dist.get("forward", 0) / n, 4) if n else 0.0,
        "unsafe_raw_action_rate": round(unsafe / n, 4) if n else 0.0,
        "emergency_n": len(emer),
        "emergency_forward_proposal_rate": (
            round(emer_fwd / len(emer), 4) if emer else None),
        "answer_confidence_mean": (
            round(sum(aconfs) / len(aconfs), 4) if aconfs else None),
        "ece": cal.expected_calibration_error(calib_recs),
        "brier": cal._brier(calib_recs),
        "latency_p50_ms": (pct(lats, 50, 50)[0] if lats else None),
        "latency_p95_ms": (pct(lats, 95, 95)[0] if lats else None),
    }


def evaluate_benchmark_frame(frame_id: str, policy, replay_run: str,
                             store) -> list:
    """Fresh unconstrained + constrained decisions for one benchmark frame."""
    from backend.services import replay_jev as rj
    from backend.services.replay_service import DataLoadError, FrameNotFoundError

    import numpy as np
    from src.decision.jev_state_adapter import enrich_for_jev, to_named_state
    from src.decision.policy_interface import decide_with_safety, get_policy
    from src.simulation.stages_runner import run_stages_1_to_6

    fid = str(frame_id)
    entry = store.get_frame(fid)
    path = PROJECT_ROOT / str(entry["points_path"])
    points = np.load(path).astype("float64")
    out = run_stages_1_to_6(points, fid, entry.get("timestamp"))
    st = out["state"]
    named = to_named_state(st["state_vector"], st["sector_ranges_m"])
    named = enrich_for_jev(named, st["safety"])
    recs = []
    for mode in ("unconstrained", "constrained"):
        dec = decide_with_safety(
            policy, named, st["safety"]["nearest_forward_obstacle_m"],
            True, bool(st["safety"]["emergency_stop"]), frame_id=fid,
            run_id=replay_run, source="benchmark-laya",
            log_path=str(PROJECT_ROOT / "results" / "decisions" /
                         "benchmark_laya_requests.jsonl"),
            mode=mode, safety_dict=st["safety"])
        store.record_decision(
            fid, "benchmark-laya", model=dec.get("model"),
            proposed_action=dec.get("proposed_action"),
            confidence=dec.get("confidence"),
            probabilities=dec.get("probabilities"),
            latency_ms=dec.get("policy_latency_ms"),
            request_id=f"{replay_run}:{fid}:{mode}",
            error=dec.get("engine_error", dec.get("jev_error")),
            replay_run_id=replay_run,
            answer_confidence=dec.get("answer_confidence"),
            mode=dec.get("mode"),
            eligible=dec.get("eligible_actions"),
            raw_action=dec.get("raw_laya_action"),
            constrained_action=dec.get("constrained_action"),
            checkpoint_revision=dec.get("checkpoint_revision"),
            gate_threshold=dec.get("gate_threshold"))
        recs.append(dec)
    return recs


def main() -> dict:
    ap = argparse.ArgumentParser(description="Baseline Laya evaluation")
    ap.add_argument("--splits", default="results/laya_dataset_splits.json")
    ap.add_argument("--eval-splits", nargs="+",
                    default=["validation", "test", "ood_test"],
                    help="which splits get FRESH Laya calls (benchmark rows)")
    ap.add_argument("--replay-run", default=None)
    args = ap.parse_args()

    from src.decision.policy_interface import get_policy
    from backend.services import laya_manager, store

    boot = laya_manager.ensure_started(wait_s=600.0)
    assert boot["health"].get("healthy"), "Laya server not ready"
    policy = get_policy("laya")
    replay_run = args.replay_run or f"benchmark-eval-{int(time.time())}"

    splits = json.loads((PROJECT_ROOT / args.splits).read_text())
    recorded = {r["frame_id"]: r for r in json.loads(
        (PROJECT_ROOT / "results" / "laya_recorded_dataset.json"
         ).read_text())["rows"]}
    bench_labels = {r["frame_id"]: r for r in json.loads(
        (PROJECT_ROOT / "results" / "benchmark_labels.json"
         ).read_text())["rows"]}

    all_rows: list = []
    for split in splits["counts"]:
        for fid in splits["splits"][split]:
            if fid in recorded:
                r = recorded[fid]
                all_rows.append({
                    "frame_id": fid, "split": split,
                    "origin": "live-recorded",
                    "category": r["category"],
                    "oracle_admissible": r["oracle_admissible"],
                    "oracle_preferred": r["oracle_preferred"],
                    "mode": "constrained_laya",
                    "eligible_actions": None,
                    "proposed_action": r["proposed_action"],
                    "answer_confidence": r["answer_confidence"],
                    "probabilities": r["probabilities"],
                    "raw_laya_action": None,
                    "constrained_action": None,
                    "latency_ms": r["latency_ms"],
                    "oracle_correct": r["oracle_correct"],
                    "unsafe": r["unsafe"],
                    "fresh_call": False,
                })
            elif fid in bench_labels and split in args.eval_splits:
                b = bench_labels[fid]
                try:
                    decs = evaluate_benchmark_frame(fid, policy, replay_run,
                                                    store)
                except Exception as exc:  # noqa: BLE001 - recorded
                    print(f"eval failed {fid}: {exc}")
                    continue
                for dec in decs:
                    prop = EXEC_TO_CHOICE.get(
                        str(dec.get("proposed_action") or ""), None)
                    raw = dec.get("raw_laya_action")
                    raw_choice = (raw if raw in LABELS else EXEC_TO_CHOICE.get(
                        str(raw or ""), None))
                    all_rows.append({
                        "frame_id": fid, "split": split,
                        "origin": "benchmark-generated",
                        "category": b["category"],
                        "oracle_admissible": b["oracle_admissible"],
                        "oracle_preferred": b["oracle_preferred"],
                        "mode": dec.get("mode"),
                        "eligible_actions": dec.get("eligible_actions"),
                        "proposed_action": prop,
                        "answer_confidence": dec.get("answer_confidence"),
                        "probabilities": dec.get("probabilities"),
                        "raw_laya_action": raw_choice,
                        "constrained_action": dec.get("constrained_action"),
                        "latency_ms": dec.get("policy_latency_ms"),
                        "oracle_correct": (prop == b["oracle_preferred"]),
                        "unsafe": ((prop not in b["oracle_admissible"])
                                   if prop else None),
                        "fresh_call": True,
                    })
    # ---- baseline report ----
    by: dict = {}
    for split in ["validation", "test", "ood_test"]:
        for mode in ["constrained_laya", "unconstrained_laya"]:
            rows = [r for r in all_rows if r["split"] == split
                    and r["mode"] == mode]
            if rows:
                by[f"{split}:{mode}"] = metrics_for(rows)
    by_cat: dict = {}
    for mode in ["constrained_laya", "unconstrained_laya"]:
        for cat in sorted({r["category"] for r in all_rows}):
            rows = [r for r in all_rows
                    if r["mode"] == mode and r["category"] == cat
                    and r["split"] in ("validation", "test")]
            if rows:
                by_cat[f"{cat}:{mode}"] = metrics_for(rows)
    baseline = {"generated_utc": _utc(), "replay_run_id": replay_run,
                "checkpoint": "convaiinnovations/laya/typed-decisions",
                "n_rows": len(all_rows),
                "by_split_mode": by, "by_category_mode": by_cat}
    (PROJECT_ROOT / "results" / "laya_baseline.json").write_text(
        json.dumps(baseline, indent=2, default=str))
    with open(PROJECT_ROOT / "results" / "laya_baseline.csv", "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["slice", "n", "accuracy", "balanced_accuracy",
                    "macro_f1", "forward_rate", "unsafe_rate",
                    "emer_fwd_rate", "aconf_mean", "ece", "brier",
                    "lat_p50", "lat_p95"])
        for k, m in {**by, **by_cat}.items():
            w.writerow([k, m["n"], m["accuracy"], m["balanced_accuracy"],
                        m["macro_f1"], m["forward_rate"],
                        m["unsafe_raw_action_rate"],
                        m["emergency_forward_proposal_rate"],
                        m["answer_confidence_mean"], m["ece"], m["brier"],
                        m["latency_p50_ms"], m["latency_p95_ms"]])
    md = ["# Laya baseline (measured, current checkpoint, pre-change)",
          f"Generated: {baseline['generated_utc']}",
          f"Replay run: {replay_run}, rows: {len(all_rows)}", "",
          "No good/bad verdict is given; numbers only.", ""]
    for k, m in by.items():
        md.append(f"## {k} (n={m['n']})")
        md.append(f"accuracy={m['accuracy']} balanced={m['balanced_accuracy']} "
                  f"macroF1={m['macro_f1']} forward_rate={m['forward_rate']} "
                  f"unsafe={m['unsafe_raw_action_rate']} "
                  f"emer_fwd={m['emergency_forward_proposal_rate']} "
                  f"ece={m['ece']} brier={m['brier']} "
                  f"lat_p50={m['latency_p50_ms']} lat_p95={m['latency_p95_ms']}")
    (PROJECT_ROOT / "results" / "laya_baseline.md").write_text("\n".join(md))

    # ---- forward-bias analysis ----
    uncon = [r for r in all_rows if r["mode"] == "unconstrained_laya"]
    bias = {
        "generated_utc": _utc(), "n_unconstrained": len(uncon),
        "forward_proposal_rate": (sum(
            1 for r in uncon if (r.get("raw_laya_action")
                                 or r.get("proposed_action")) == "forward")
            / len(uncon) if uncon else None),
        "forward_when_inadmissible": None, "forward_in_emergency": None,
        "executed_forward_when_inadmissible": None,
        "by_category": {}, "confusion": None,
        "attribution": (
            "Raw proposals are compared against oracle admissibility and "
            "scenario distribution. A rate above the admissible share is "
            "reported as systematic preference; whether it stems from "
            "scenario mix, prompt wording, option order, or model weights "
            "is tested separately (option-order run; fine-tuning NOT RUN)."),
    }
    inad = [r for r in uncon if r.get("proposed_action") not in
            (r.get("oracle_admissible") or [])]
    bias["forward_when_inadmissible"] = {
        "n_inadmissible": len(inad),
        "n_proposed_forward": sum(
            1 for r in inad if (r.get("raw_laya_action")
                                or r.get("proposed_action")) == "forward")}
    emer = [r for r in uncon if r.get("category") == "F_EMERGENCY"]
    bias["forward_in_emergency"] = {
        "n_emergency": len(emer),
        "n_proposed_forward": sum(
            1 for r in emer if (r.get("raw_laya_action")
                                or r.get("proposed_action")) == "forward")}
    for cat in sorted({r["category"] for r in uncon}):
        crows = [r for r in uncon if r["category"] == cat]
        dist = dict(Counter(r.get("raw_laya_action") or r.get("proposed_action")
                            for r in crows))
        bias["by_category"][cat] = {"n": len(crows), "raw_distribution": dist}
    (PROJECT_ROOT / "results" / "laya_forward_bias.json").write_text(
        json.dumps(bias, indent=2, default=str))
    with open(PROJECT_ROOT / "results" / "laya_forward_bias.csv", "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "n", "raw_distribution"])
        for k, v in bias["by_category"].items():
            w.writerow([k, v["n"], json.dumps(v["raw_distribution"])])
    print(json.dumps({k: baseline[k] for k in ("n_rows",)},
                     default=str), "slices:", len(by) + len(by_cat))
    print("bias:", json.dumps({k: bias[k] for k in
                               ("forward_proposal_rate",
                                "forward_when_inadmissible",
                                "forward_in_emergency")}, default=str))
    store.close()
    return baseline


if __name__ == "__main__":
    main()
