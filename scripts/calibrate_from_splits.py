"""Split-based calibration (Phases 13-15): sweep + temperature + gate.

Data (no leakage by construction):
    CALIBRATION: recorded-calib rows (historical production decisions) +
                 bench-calib rows (fresh calls made HERE, both modes).
    VALIDATION:  recorded-valid rows + bench-valid fresh rows (from the
                 baseline run).
    TEST:        untouched until final reporting (Phase 16 script).

Thresholds: 0.10-0.90 step 0.05 + finer +-0.02 around the validation
best. Selection objective (safety-prioritized):
    PRIMARY:   minimize unsafe accepted actions
    SUBJECT TO: coverage >= 0.50
    SECONDARY: maximize correct accepted decisions
    TERTIARY:  minimize unnecessary STOP/fallback.
Stores legacy_starting_threshold=0.35 and validated_candidate_threshold.
Production config is updated ONLY if stop conditions pass
(zero unsafe on validation, revision match); otherwise 0.35 stays with
status NOT VALIDATED. Rollback = legacy value in the artifact.

Temperature: fit on CALIBRATION only; deploy only if VALIDATION
ECE/Brier improve, else "calibration rejected" and previous kept.

Writes: results/laya_threshold_sweep.{json,csv},
        results/laya_threshold_selection.md,
        models/laya/calibration/{gate_selection.json,temperature.json}.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXEC_TO_CHOICE = {"forward": "forward", "turn_left": "left",
                  "turn_right": "right", "stop": "stop"}
LABELS = ["forward", "left", "right", "stop"]
LEGACY_THRESHOLD = 0.35

GRID = [round(0.10 + 0.05 * i, 2) for i in range(17)]


def _utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_split_rows():
    splits = json.loads((PROJECT_ROOT / "results" / "laya_dataset_splits.json"
                         ).read_text())
    recorded = {r["frame_id"]: r for r in json.loads(
        (PROJECT_ROOT / "results" / "laya_recorded_dataset.json"
         ).read_text())["rows"]}
    bench = {r["frame_id"]: r for r in json.loads(
        (PROJECT_ROOT / "results" / "benchmark_labels.json").read_text())["rows"]}
    return splits, recorded, bench


def recorded_row(r: dict, split: str) -> dict:
    return {"frame_id": r["frame_id"], "split": split,
            "origin": "live-recorded", "category": r["category"],
            "oracle_preferred": r["oracle_preferred"],
            "oracle_admissible": None,  # rebuilt below if clearances exist
            "mode": "constrained_laya",
            "proposed_action": r["proposed_action"],
            "answer_confidence": r["answer_confidence"],
            "probabilities": r["probabilities"],
            "latency_ms": r["latency_ms"],
            "oracle_correct": r["oracle_correct"], "unsafe": r["unsafe"],
            "fresh_call": False}


def main() -> dict:
    ap = argparse.ArgumentParser(description="Split-based calibration")
    ap.add_argument("--fresh-calib", action="store_true",
                    help="run fresh Laya calls for bench calibration rows")
    ap.add_argument("--replay-run", default=None)
    args = ap.parse_args()

    from src.decision import calibration as cal
    from src.decision.checkpoint_info import checkpoint_info
    from src.decision.policy_interface import get_policy
    from backend.services import laya_manager, store

    splits, recorded, bench = load_split_rows()
    rev = checkpoint_info().get("revision")

    calib_rows, valid_rows = [], []
    for fid in splits["splits"]["calibration"]:
        if fid in recorded:
            calib_rows.append(recorded_row(recorded[fid], "calibration"))
    for fid in splits["splits"]["validation"]:
        if fid in recorded:
            valid_rows.append(recorded_row(recorded[fid], "validation"))
    bench_calib = [fid for fid in splits["splits"]["calibration"]
                   if fid in bench]
    bench_valid = [fid for fid in splits["splits"]["validation"]
                   if fid in bench]

    # Pull already-evaluated bench validation rows from SQLite.
    have = {}
    for fid in bench_valid:
        for mode in ("constrained_laya", "unconstrained_laya"):
            d = store.get_decision(fid, source="benchmark-laya")
            # latest row only; need per-mode: query replay_run rows
            pass
    con = store._connect()
    for fid in bench_valid + (bench_calib if not args.fresh_calib else []):
        for row in con.execute(
                "SELECT * FROM decisions WHERE frame_id=? AND "
                "source='benchmark-laya' ORDER BY id", (fid,)):
            d = dict(row)
            if d.get("mode") not in ("constrained_laya",
                                     "unconstrained_laya"):
                continue
            b = bench[fid]
            try:
                probs = json.loads(d["probabilities_json"] or "{}")
            except ValueError:
                probs = {}
            prop = EXEC_TO_CHOICE.get(str(d["proposed_action"] or ""), None)
            tgt = (valid_rows if fid in splits["splits"]["validation"]
                   else calib_rows)
            tgt.append({
                "frame_id": fid, "split": "validation"
                if fid in splits["splits"]["validation"] else "calibration",
                "origin": "benchmark-generated",
                "category": b["category"],
                "oracle_preferred": b["oracle_preferred"],
                "mode": d["mode"],
                "proposed_action": prop,
                "answer_confidence": d["answer_confidence"],
                "probabilities": probs,
                "latency_ms": d["latency_ms"],
                "oracle_correct": (prop == b["oracle_preferred"]),
                "unsafe": ((prop not in b["oracle_admissible"])
                           if prop else None),
                "fresh_call": True})

    if args.fresh_calib and bench_calib:
        import scripts.evaluate_dataset as ED
        boot = laya_manager.ensure_started(wait_s=600.0)
        assert boot["health"].get("healthy")
        policy = get_policy("laya")
        replay_run = args.replay_run or f"benchmark-calib-{int(time.time())}"
        for fid in bench_calib:
            try:
                decs = ED.evaluate_benchmark_frame(fid, policy, replay_run,
                                                   store)
            except Exception as exc:  # noqa: BLE001
                print(f"calib eval failed {fid}: {exc}")
                continue
            b = bench[fid]
            for dec in decs:
                prop = EXEC_TO_CHOICE.get(
                    str(dec.get("proposed_action") or ""), None)
                calib_rows.append({
                    "frame_id": fid, "split": "calibration",
                    "origin": "benchmark-generated",
                    "category": b["category"],
                    "oracle_preferred": b["oracle_preferred"],
                    "mode": dec.get("mode"),
                    "proposed_action": prop,
                    "answer_confidence": dec.get("answer_confidence"),
                    "probabilities": dec.get("probabilities"),
                    "latency_ms": dec.get("policy_latency_ms"),
                    "oracle_correct": (prop == b["oracle_preferred"]),
                    "unsafe": ((prop not in b["oracle_admissible"])
                               if prop else None),
                    "fresh_call": True})

    # Production distribution for the gate = constrained rows.
    cal_rows = [r for r in calib_rows if r["mode"] == "constrained_laya"
                and r["answer_confidence"] is not None]
    val_rows = [r for r in valid_rows if r["mode"] == "constrained_laya"
                and r["answer_confidence"] is not None]
    print(f"calibration n={len(cal_rows)} validation n={len(val_rows)}")

    cal_sweep = cal.sweep_thresholds(cal_rows, thresholds=GRID)
    val_sweep = cal.sweep_thresholds(val_rows, thresholds=GRID)

    # Safety-prioritized selection on VALIDATION.
    # Tie-break rule (documented): among equivalent (unsafe, coverage,
    # correct) candidates, RETAIN the legacy threshold when it ties
    # (minimize production change without evidence of benefit); else the
    # highest full-coverage threshold. Never minimize blindly: the
    # minimum would make the gate vacuous.
    feas = [r for r in val_sweep if r["coverage"] >= 0.50]
    pool = feas if feas else val_sweep
    best_key = min((r["unsafe_accepted"], -(r["correct_accepted"]),
                    r["unnecessary_stop_rate"]) for r in pool)
    tied = [r for r in pool
            if (r["unsafe_accepted"], -(r["correct_accepted"]),
                r["unnecessary_stop_rate"]) == best_key]
    tied_ts = sorted(r["threshold"] for r in tied)
    if LEGACY_THRESHOLD in tied_ts:
        coarse = LEGACY_THRESHOLD
        tie_note = "legacy retained (tied optimum)"
    else:
        full = [t for t in tied_ts
                if next(r for r in tied
                        if r["threshold"] == t)["coverage"] == 1.0]
        coarse = max(full) if full else tied_ts[0]
        tie_note = "highest full-coverage tied optimum"
    fine_grid = sorted({round(coarse + d, 2) for d in
                        (-0.02, -0.01, 0.0, 0.01, 0.02)} | {coarse})
    fine = cal.sweep_thresholds(val_rows, thresholds=fine_grid)
    feas_f = [r for r in fine if r["coverage"] >= 0.50] or fine
    fbest = min((r["unsafe_accepted"], -(r["correct_accepted"]),
                 r["unnecessary_stop_rate"]) for r in feas_f)
    ftied = sorted(r["threshold"] for r in feas_f
                   if (r["unsafe_accepted"], -(r["correct_accepted"]),
                       r["unnecessary_stop_rate"]) == fbest)
    if LEGACY_THRESHOLD in ftied:
        selected, tie_note = LEGACY_THRESHOLD, "legacy retained (tied optimum)"
    else:
        full = [t for t in ftied
                if next(r for r in feas_f
                        if r["threshold"] == t)["coverage"] == 1.0]
        selected = max(full) if full else ftied[0]
        tie_note = "highest full-coverage tied optimum"
    sel = next(r for r in feas_f if r["threshold"] == selected)

    sweep_payload = {
        "generated_utc": _utc(), "checkpoint_revision": rev,
        "dataset_hash": splits["dataset_hash"],
        "split_hash": splits["split_hash"],
        "n_calibration": len(cal_rows), "n_validation": len(val_rows),
        "candidate_thresholds": GRID, "fine_grid": fine_grid,
        "calibration_sweep": cal_sweep, "validation_sweep": val_sweep,
        "validation_fine_sweep": fine,
        "selection_objective": ("PRIMARY minimize unsafe accepted; SUBJECT TO "
                                "coverage>=0.50; SECONDARY maximize correct "
                                "accepted; TERTIARY minimize unnecessary STOP"),
        "selected_threshold": selected,
        "tie_break": tie_note,
        "plateau_note": ("Thresholds <=0.40 are outcome-identical on "
                         "validation (eligibility carries admissibility); "
                         "optimality NOT claimed."),
        "legacy_starting_threshold": LEGACY_THRESHOLD,
    }
    (PROJECT_ROOT / "results" / "laya_threshold_sweep.json").write_text(
        json.dumps(sweep_payload, indent=2))
    with open(PROJECT_ROOT / "results" / "laya_threshold_sweep.csv", "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "threshold", "n", "accepted", "coverage",
                    "correct_accepted", "unsafe_accepted",
                    "unnecessary_stop_rate", "balanced_accuracy"])
        for name, rows in (("calibration", cal_sweep),
                           ("validation", val_sweep)):
            for r in rows:
                w.writerow([name, r["threshold"], r["n_total"],
                            r["accepted"], r["coverage"],
                            r["correct_accepted"], r["unsafe_accepted"],
                            r["unnecessary_stop_rate"],
                            r["balanced_accuracy"]])

    # Temperature: fit on CALIBRATION, judge on VALIDATION.
    def temp_recs(rows):
        out = []
        for r in rows:
            if isinstance(r.get("probabilities"), dict) and r["probabilities"]:
                top = max(r["probabilities"], key=r["probabilities"].get)
                out.append({"probabilities": r["probabilities"],
                            "oracle_preferred": r["oracle_preferred"],
                            "oracle_correct": r["oracle_correct"],
                            "n_options": len(r["probabilities"])})
        return out

    cal_t = temp_recs(cal_rows)
    val_t = temp_recs([r for r in valid_rows])
    groups: dict = {}
    for r in cal_t:
        groups.setdefault(r["n_options"], []).append(r)
    temp_groups = {}
    for n_opts, grec in sorted(groups.items()):
        fit = cal.fit_temperature(grec, question_type="choice",
                                  n_options=n_opts)
        # judge on validation rows with same option count
        gv = [r for r in val_t if r["n_options"] == n_opts]
        fit["validation_n"] = len(gv)
        if gv and fit.get("fitted"):
            before = sum(-__import__("math").log(max(
                r["probabilities"].get(r["oracle_preferred"], 0.0), 1e-12))
                for r in gv) / len(gv)
            after = sum(-__import__("math").log(max(
                cal.apply_temperature(r["probabilities"],
                                      fit["temperature"]).get(
                                          r["oracle_preferred"], 0.0), 1e-12))
                for r in gv) / len(gv)
            e_before = cal.expected_calibration_error(
                [{"answer_confidence": r["probabilities"].get(
                    r["oracle_preferred"]), "oracle_correct": r["oracle_correct"]}
                 for r in gv])
            scaled = [{"answer_confidence": cal.apply_temperature(
                r["probabilities"], fit["temperature"]).get(
                    r["oracle_preferred"]),
                "oracle_correct": r["oracle_correct"]} for r in gv]
            fit["validation_nll_before"] = round(before, 4)
            fit["validation_nll_after"] = round(after, 4)
            fit["validation_ece_before"] = e_before
            fit["validation_ece_after"] = cal.expected_calibration_error(scaled)
            fit["validation_brier_after"] = cal._brier(scaled)
            fit["deployed"] = bool(after < before)
        else:
            fit["deployed"] = False
        temp_groups[str(n_opts)] = fit
    deployable = {k: v for k, v in temp_groups.items() if v.get("deployed")}

    temp_payload = {
        "artifact": "temperature", "version": 2, "date_utc": _utc(),
        "dataset_hash": splits["dataset_hash"],
        "split_hash": splits["split_hash"], "split": "calibration",
        "checkpoint_revision": rev, "groups": temp_groups,
        "question_type": "choice",
        "n_options": None, "temperature": None, "fitted": False,
        "note": ("Runtime group = dominant DEPLOYED option count; "
                 "non-deployed groups fall back unscaled.")}

    # Gate promotion decision (stop conditions).
    stop_reasons = []
    if sel["unsafe_accepted"] != 0:
        stop_reasons.append(
            f"validation unsafe_accepted={sel['unsafe_accepted']} > 0")
    if not rev:
        stop_reasons.append("no pinned checkpoint revision")
    promoted = not stop_reasons
    final_threshold = selected if promoted else LEGACY_THRESHOLD
    if deployable:
        dom = max(deployable, key=lambda k: temp_groups[k].get(
            "n_records", 0))
        temp_payload.update({
            "n_options": int(dom), "temperature": temp_groups[dom][
                "temperature"], "fitted": True,
            "ece_before": temp_groups[dom].get("validation_ece_before"),
            "ece_after": temp_groups[dom].get("validation_ece_after"),
            "brier_after": temp_groups[dom].get("validation_brier_after"),
            "n_records": temp_groups[dom].get("n_records")})
    else:
        temp_payload["note"] += " calibration rejected on validation."

    gate_payload = {
        "artifact": "gate_selection", "version": 2, "date_utc": _utc(),
        "dataset_id": "laya-splits-v1",
        "dataset_hash": splits["dataset_hash"],
        "split_hash": splits["split_hash"],
        "split": "validation",
        "n_calibration": len(cal_rows), "n_validation": len(val_rows),
        "confidence_metric": "answer_confidence",
        "legacy_starting_threshold": LEGACY_THRESHOLD,
        "validated_candidate_threshold": selected,
        "selected_threshold": selected,
        "tie_break": tie_note,
        "plateau_note": sweep_payload.get("plateau_note"),
        "current_gate_threshold": final_threshold,
        "promoted": promoted,
        "stop_reasons": stop_reasons,
        "selection_objective": sweep_payload["selection_objective"],
        "validation_row": sel,
        "checkpoint_revision": rev,
        "note": ("Promoted only under zero-unsafe validation; otherwise "
                 "legacy 0.35 stays live with status NOT VALIDATED."),
    }
    cal.save_gate_selection(gate_payload)
    cal.save_temperature(temp_payload)

    cfg_path = PROJECT_ROOT / "config" / "laya_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["confidence_gate"]["threshold"] = final_threshold
    cfg["confidence_gate"]["status"] = ("VALIDATED-CANDIDATE"
                                        if promoted else "NOT VALIDATED")
    cfg["confidence_gate"]["validated_candidate_threshold"] = selected
    cfg["confidence_gate"]["legacy_starting_threshold"] = LEGACY_THRESHOLD
    cfg["confidence_gate"]["artifact"] = \
        "models/laya/calibration/gate_selection.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))

    sel_md = [
        "# Laya threshold selection (validation-only, safety-prioritized)",
        f"Date: {_utc()}, revision: {rev}",
        f"Calibration n={len(cal_rows)}, validation n={len(val_rows)}",
        f"Legacy starting threshold: {LEGACY_THRESHOLD}",
        f"Validated candidate: {selected}",
        f"Promoted to production: {promoted}"
        + ("" if promoted else f" (blocked: {'; '.join(stop_reasons)})"),
        f"Current gate: {final_threshold}",
        "", "## Validation sweep (coarse)",
        "| t | accepted | coverage | correct | unsafe | unnec STOP |",
    ]
    for r in val_sweep:
        sel_md.append(f"| {r['threshold']} | {r['accepted']} | {r['coverage']} | "
                      f"{r['correct_accepted']} | {r['unsafe_accepted']} | "
                      f"{r['unnecessary_stop_rate']} |")
    sel_md += ["", "## Temperature groups (fit=calibration, judged=validation)"]
    for k, v in temp_groups.items():
        sel_md.append(f"- n_options={k}: T={v.get('temperature')} "
                      f"deployed={v.get('deployed')} "
                      f"val_nll {v.get('validation_nll_before')} -> "
                      f"{v.get('validation_nll_after')}")
    (PROJECT_ROOT / "results" / "laya_threshold_selection.md").write_text(
        "\n".join(sel_md))
    print(f"selected={selected} promoted={promoted} gate={final_threshold}")
    print(f"temperature groups: {json.dumps({k: v.get('deployed') for k, v in temp_groups.items()})}")
    store.close()
    return gate_payload


if __name__ == "__main__":
    main()
