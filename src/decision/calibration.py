"""Confidence-gate calibration (threshold sweep + temperature scaling).

All inputs are measured project data (recorded-frame decisions labelled by
the deterministic navigation oracle in action_eligibility.py). Nothing here
invents model outputs.

Splits: callers must pass TRAIN/CALIBRATION, VALIDATION, TEST splits built
without leakage (no frame in two splits; scenario families separated where
possible). Thresholds are swept on CALIBRATION, the operating point is
selected on VALIDATION under a documented objective, TEST is reported once.

Selection objective (documented, in priority order):
    PRIMARY:   minimize unsafe accepted actions (unsafe_accepted == 0 required)
    SECONDARY: maximize coverage (accepted / total)
    TERTIARY:  minimize unnecessary STOP (rejected-but-oracle-correct / total)

Temperature scaling: single-parameter softmax scaling on pseudo-logits
log(p) (we do not have raw logits), fitted by minimizing NLL on the held-out
calibration split, grouped by (question_type, n_options) as required.
Stored under models/laya/calibration/temperature.json with dataset hash,
split id, checkpoint revision, date, ECE/Brier before/after. Applied at
runtime ONLY when the artifact matches the exact checkpoint/configuration;
otherwise the runtime falls back to unscaled probabilities and reports
CALIBRATION: NOT LOADED.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CALIBRATION_DIR = PROJECT_ROOT / "models" / "laya" / "calibration"
GATE_SELECTION_PATH = CALIBRATION_DIR / "gate_selection.json"
TEMPERATURE_PATH = CALIBRATION_DIR / "temperature.json"

CANDIDATE_THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55,
                        0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def dataset_hash(records: Sequence[Dict[str, Any]]) -> str:
    """Stable hash of the calibration records (frame ids + confidences)."""
    h = hashlib.sha256()
    for r in sorted(records, key=lambda d: str(d.get("frame_id", ""))):
        h.update(json.dumps({
            "frame_id": r.get("frame_id"),
            "answer_confidence": r.get("answer_confidence"),
            "oracle_correct": r.get("oracle_correct"),
            "unsafe": r.get("unsafe"),
        }, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def _brier(records: Sequence[Dict[str, Any]],
           key: str = "answer_confidence") -> float | None:
    vals = [(float(r[key]), 1.0 if r.get("oracle_correct") else 0.0)
            for r in records
            if r.get(key) is not None and r.get("oracle_correct") is not None]
    if not vals:
        return None
    return round(sum((c - y) ** 2 for c, y in vals) / len(vals), 4)


def expected_calibration_error(
        records: Sequence[Dict[str, Any]], n_bins: int = 10,
        key: str = "answer_confidence") -> float | None:
    """ECE of answer_confidence vs oracle correctness (measured only)."""
    vals = [(float(r[key]), 1.0 if r.get("oracle_correct") else 0.0)
            for r in records
            if r.get(key) is not None and r.get("oracle_correct") is not None]
    if not vals:
        return None
    bins: List[List[float]] = [[] for _ in range(n_bins)]
    accs: List[List[float]] = [[] for _ in range(n_bins)]
    for c, y in vals:
        i = min(int(c * n_bins), n_bins - 1)
        bins[i].append(c)
        accs[i].append(y)
    ece = 0.0
    for b, a in zip(bins, accs):
        if a:
            ece += abs(sum(b) / len(b) - sum(a) / len(a)) * (len(a) / len(vals))
    return round(ece, 4)


def sweep_thresholds(
    records: Sequence[Dict[str, Any]],
    thresholds: Sequence[float] | None = None,
    confidence_key: str = "answer_confidence",
) -> List[Dict[str, Any]]:
    """Evaluate every candidate gate on labelled records.

    Record schema: {frame_id, answer_confidence, oracle_correct (bool),
    unsafe (bool: accepted action would be inadmissible/unsafe)}.
    Records with missing confidence are always rejected (fallback STOP).
    """
    recs = list(records)
    total = len(recs)
    rows = []
    for t in (list(thresholds) if thresholds else list(CANDIDATE_THRESHOLDS)):
        accepted = [r for r in recs
                    if r.get(confidence_key) is not None
                    and float(r[confidence_key]) >= float(t)]
        rejected = [r for r in recs if r not in accepted]
        unsafe_acc = sum(1 for r in accepted if r.get("unsafe"))
        correct_acc = sum(1 for r in accepted if r.get("oracle_correct"))
        unnec_stop = sum(1 for r in rejected
                         if r.get("oracle_correct") and not r.get("unsafe"))
        # Balanced accuracy over the accepted set vs fallback: treat gate as
        # a binary classifier (accept-correct vs everything else).
        tp = correct_acc
        fp = len(accepted) - correct_acc
        fn = sum(1 for r in rejected if r.get("oracle_correct"))
        tn = len(rejected) - fn
        tpr = (tp / (tp + fn)) if (tp + fn) else 0.0
        tnr = (tn / (tn + fp)) if (tn + fp) else 0.0
        rows.append({
            "threshold": round(float(t), 2),
            "n_total": total,
            "accepted": len(accepted),
            "rejected": len(rejected),
            "coverage": round(len(accepted) / total, 4) if total else 0.0,
            "fallback_stop_rate": round(len(rejected) / total, 4) if total else 0.0,
            "correct_accepted": correct_acc,
            "unsafe_accepted": unsafe_acc,
            "unnecessary_rejections": unnec_stop,
            "unnecessary_stop_rate": round(unnec_stop / total, 4) if total else 0.0,
            "balanced_accuracy": round((tpr + tnr) / 2, 4),
            "brier_accepted": _brier(accepted, confidence_key),
        })
    return rows


def select_threshold(sweep_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply the documented objective; returns {threshold, reason, row}."""
    rows = sorted(sweep_rows, key=lambda r: r["threshold"])
    feasible = [r for r in rows if r["unsafe_accepted"] == 0]
    pool = feasible if feasible else rows
    key = (lambda r: (r["unsafe_accepted"], -r["coverage"],
                      r["unnecessary_stop_rate"], -r["threshold"]))
    best = min(pool, key=key)
    reason = ("PRIMARY minimize unsafe_accepted "
              f"({'0 feasible' if feasible else 'NO zero-unsafe threshold; minimum taken'}); "
              "SECONDARY maximize coverage; TERTIARY minimize unnecessary STOP.")
    return {"threshold": best["threshold"], "reason": reason,
            "zero_unsafe_feasible": bool(feasible), "row": best}


def calibrate_gate(records: Sequence[Dict[str, Any]],
                   split: str,
                   dataset_id: str,
                   checkpoint_revision: str | None,
                   confidence_key: str = "answer_confidence",
                   thresholds: Sequence[float] | None = None) -> Dict[str, Any]:
    """Sweep + select on one split; returns the artifact payload (not saved)."""
    rows = sweep_thresholds(records, thresholds, confidence_key)
    sel = select_threshold(rows)
    return {
        "artifact": "gate_selection",
        "version": 1,
        "date_utc": _utc(),
        "dataset_id": dataset_id,
        "dataset_hash": dataset_hash(records),
        "split": split,
        "n_records": len(records),
        "confidence_metric": confidence_key,
        "candidate_thresholds": [r["threshold"] for r in rows],
        "sweep": rows,
        "selected_threshold": sel["threshold"],
        "selection_objective": sel["reason"],
        "zero_unsafe_feasible": sel["zero_unsafe_feasible"],
        "ece": expected_calibration_error(records, key=confidence_key),
        "brier": _brier(records, confidence_key),
        "checkpoint_revision": checkpoint_revision,
        "note": ("Selected threshold under the defined validation objective; "
                 "NOT claimed optimal until independently evaluated."),
    }


def save_gate_selection(payload: Dict[str, Any],
                        path: Path | None = None) -> Path:
    path = path or GATE_SELECTION_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_gate_selection(path: Path | None = None) -> Dict[str, Any] | None:
    try:
        return json.loads((path or GATE_SELECTION_PATH).read_text())
    except (OSError, ValueError):
        return None


# ---- temperature scaling -------------------------------------------------

def apply_temperature(probs: Dict[str, float], temperature: float) -> Dict[str, float]:
    """Softmax(log(p)/T) rescaling of a probability dict (measured I/O)."""
    t = max(float(temperature), 1e-3)
    logits = {k: math.log(max(float(v), 1e-12)) for k, v in probs.items()}
    mx = max(logits.values())
    exps = {k: math.exp((v - mx) / t) for k, v in logits.items()}
    s = sum(exps.values())
    return {k: round(v / s, 6) for k, v in exps.items()}


def _nll(probs: Dict[str, float], correct_choice: str) -> float:
    return -math.log(max(float(probs.get(correct_choice, 0.0)), 1e-12))


def fit_temperature(records: Sequence[Dict[str, Any]],
                    question_type: str = "choice",
                    n_options: int = 4) -> Dict[str, Any]:
    """Grid-fit T minimizing NLL on held-out records.

    Record schema: {probabilities: {choice: p}, oracle_preferred: choice}.
    """
    usable = [r for r in records
              if isinstance(r.get("probabilities"), dict)
              and r.get("oracle_preferred") in (r.get("probabilities") or {})]
    if not usable:
        return {"fitted": False, "reason": "no usable probability records",
                "question_type": question_type, "n_options": n_options}
    before = sum(_nll(r["probabilities"], r["oracle_preferred"])
                 for r in usable) / len(usable)
    ece_before = expected_calibration_error(
        [{"answer_confidence": r["probabilities"][r["oracle_preferred"]],
          "oracle_correct": r.get("oracle_correct", True)} for r in usable])
    brier_before = _brier(
        [{"answer_confidence": r["probabilities"][r["oracle_preferred"]],
          "oracle_correct": r.get("oracle_correct", True)} for r in usable])
    best_t, best_nll = 1.0, before
    t = 0.25
    while t <= 4.0:
        nll = sum(_nll(apply_temperature(r["probabilities"], t),
                       r["oracle_preferred"]) for r in usable) / len(usable)
        if nll < best_nll:
            best_nll, best_t = nll, round(t, 2)
        t = round(t + 0.25, 2)
    scaled = [{
        "answer_confidence": apply_temperature(
            r["probabilities"], best_t)[r["oracle_preferred"]],
        "oracle_correct": r.get("oracle_correct", True)} for r in usable]
    return {
        "fitted": True,
        "question_type": question_type,
        "n_options": n_options,
        "temperature": best_t,
        "nll_before": round(before, 4),
        "nll_after": round(best_nll, 4),
        "ece_before": ece_before,
        "ece_after": expected_calibration_error(scaled),
        "brier_before": brier_before,
        "brier_after": _brier(scaled),
        "n_records": len(usable),
    }


def save_temperature(payload: Dict[str, Any],
                     path: Path | None = None) -> Path:
    path = path or TEMPERATURE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_temperature(path: Path | None = None) -> Dict[str, Any] | None:
    try:
        return json.loads((path or TEMPERATURE_PATH).read_text())
    except (OSError, ValueError):
        return None


def _revision_match(artifact: Dict[str, Any] | None,
                    checkpoint_revision: str | None) -> bool:
    """Strict match: both revisions present and equal.

    An unpinned artifact (revision None) NEVER matches a pinned
    checkpoint — stale calibration must not silently apply
    (safety invariant 9/10). Before provisioning (no pinned revision
    anywhere) an existing artifact is reported for visibility but the
    runtime treats calibration as NOT LOADED.
    """
    if not artifact:
        return False
    art_rev = artifact.get("checkpoint_revision")
    if not art_rev or not checkpoint_revision:
        return False
    return art_rev == checkpoint_revision


def calibration_status(checkpoint_revision: str | None = None) -> Dict[str, Any]:
    """Runtime view: LOADED only when artifact matches checkpoint/config."""
    gate = load_gate_selection()
    temp = load_temperature()
    gate_ok = _revision_match(gate, checkpoint_revision)
    temp_ok = bool(temp and temp.get("fitted")) and _revision_match(
        temp, checkpoint_revision)
    return {
        "gate": {"loaded": bool(gate_ok),
                 "threshold": (gate or {}).get("selected_threshold"),
                 "metric": (gate or {}).get("confidence_metric"),
                 "dataset": (gate or {}).get("dataset_id"),
                 "state": "LOADED" if gate_ok else "NOT LOADED"},
        "temperature": {"loaded": bool(temp_ok),
                        "value": (temp or {}).get("temperature"),
                        "state": "LOADED" if temp_ok else "NOT LOADED"},
    }
