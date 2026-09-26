"""Evaluate the trained MLP on held-out frames (honest reference only).

Reference labels are the SAME weak-label procedure as training
(box-derived + ground-heuristic + sampled unknown) on three frames the
model never saw. These are NOT independent human point labels, so every
metric is reported as "vs annotation-derived reference". Bins with too
little labelled data are marked insufficient_data instead of fabricated.

Distance bins: 0-10 / 10-30 / 30-60 / 60-100 m (planar range), matching
the baseline distance policy in src/baselines.py.

Outputs: results/segmentation/{overall_metrics.csv, per_class_metrics.csv,
distance_metrics.csv, metrics.json}

Run:  python scripts/evaluate_segmentation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services import replay_service  # noqa: E402
from backend.services.pipeline_service import _nusc_dataset  # noqa: E402
from src.semantic_model import MODEL_CLASSES, predict_points  # noqa: E402
from scripts.train_segmentation_mlp import (  # noqa: E402
    GROUND_MARGIN_M,
    weak_labels,
)

EVAL_FRAMES = [
    "6d1d793b68b24165be5755a54054c82a",
    "dc4c80e8a1534ff8a3eb80307341161c",
    "5c85640c10d9485c86bf6eb44778c002",
]
BINS = [(0.0, 10.0), (10.0, 30.0), (30.0, 60.0), (60.0, 100.0)]
MIN_BIN_POINTS = 50
OUT_DIR = PROJECT_ROOT / "results" / "segmentation"


def prf_iou(y_true: np.ndarray, y_pred: np.ndarray, cls: str):
    t = y_true == cls
    p = y_pred == cls
    tp = int((t & p).sum())
    fp = int((~t & p).sum())
    fn = int((t & ~p).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    return {
        "support": int(t.sum()),
        "predicted": int(p.sum()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def _clean(value):
    """NaN/inf -> None so stored JSON is strict-parseable (null = unmeasurable)."""
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def main() -> None:
    nusc = _nusc_dataset()
    all_true, all_pred, all_conf, all_rng = [], [], [], []
    per_frame_rows = []
    for fid in EVAL_FRAMES:
        raw, _ = replay_service.load_frame_points(fid)
        pts = np.asarray(raw, dtype=np.float64)
        sample = nusc.get("sample", fid)
        ref = weak_labels(pts, sample, nusc)
        mask = ref != ""
        labels, sources, conf, ms = predict_points(pts)
        ok = bool((np.asarray(sources) == "model").all())
        acc = float((np.asarray(labels)[mask] == ref[mask]).mean()) if mask.sum() else float("nan")
        per_frame_rows.append(
            {
                "frame_id": fid,
                "n_points": int(len(pts)),
                "n_reference_points": int(mask.sum()),
                "accuracy_vs_reference": acc,
                "mean_confidence": float(np.mean(conf)),
                "inference_latency_ms": round(ms, 2),
                "all_sources_model": ok,
            }
        )
        print(f"eval frame {fid[:8]}: acc_vs_ref={acc:.4f} infer_ms={ms:.1f}", flush=True)
        all_true.append(ref[mask])
        all_pred.append(np.asarray(labels)[mask])
        all_conf.append(np.asarray(conf)[mask])
        all_rng.append(np.hypot(pts[mask][:, 0], pts[mask][:, 1]))
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    conf_all = np.concatenate(all_conf)
    rng_all = np.concatenate(all_rng)

    overall_acc = float((y_true == y_pred).mean())
    per_class_rows = []
    ious = []
    for cls in MODEL_CLASSES:
        m = prf_iou(y_true, y_pred, cls)
        m["class"] = cls
        per_class_rows.append(m)
        if np.isfinite(m["iou"]):
            ious.append(m["iou"])
    mean_iou = float(np.mean(ious)) if ious else float("nan")

    # F1 + confusion matrix for the /model/info contract (measured only).
    cm_labels = list(MODEL_CLASSES)
    cm = np.zeros((len(cm_labels), len(cm_labels)), dtype=np.int64)
    idx_of = {c: i for i, c in enumerate(cm_labels)}
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[idx_of[str(t)], idx_of[str(p)]] += 1
    f1_per_class = {}
    f1_vals = []
    for cls in MODEL_CLASSES:
        f1 = next(r["f1"] for r in per_class_rows if r["class"] == cls)
        f1_per_class[cls] = f1
        if np.isfinite(f1):
            f1_vals.append(f1)
    f1_macro = float(np.mean(f1_vals)) if f1_vals else float("nan")

    dist_rows = []
    for lo, hi in BINS:
        m = (rng_all >= lo) & (rng_all < hi)
        if int(m.sum()) < MIN_BIN_POINTS:
            dist_rows.append(
                {
                    "distance_bin_m": f"{lo:g}-{hi:g}",
                    "n_points": int(m.sum()),
                    "status": "insufficient_data",
                    "accuracy_vs_reference": float("nan"),
                    "mean_iou": float("nan"),
                    "mean_confidence": float("nan"),
                }
            )
            continue
        yt, yp = y_true[m], y_pred[m]
        bi = [prf_iou(yt, yp, c)["iou"] for c in MODEL_CLASSES]
        bi = [v for v in bi if np.isfinite(v)]
        dist_rows.append(
            {
                "distance_bin_m": f"{lo:g}-{hi:g}",
                "n_points": int(m.sum()),
                "status": "measured",
                "accuracy_vs_reference": float((yt == yp).mean()),
                "mean_iou": float(np.mean(bi)) if bi else float("nan"),
                "mean_confidence": float(np.mean(conf_all[m])),
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_frame_rows).to_csv(OUT_DIR / "overall_metrics.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(OUT_DIR / "per_class_metrics.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(OUT_DIR / "distance_metrics.csv", index=False)
    payload = {
        "reference": "annotation-derived weak labels on held-out frames (NOT human point labels)",
        "eval_frames": EVAL_FRAMES,
        "n_eval_points": int(len(y_true)),
        "overall_accuracy_vs_reference": overall_acc,
        "mean_iou_vs_reference": mean_iou,
        "mean_confidence": float(np.mean(conf_all)),
        "f1_macro_vs_reference": f1_macro,
        "f1_per_class_vs_reference": f1_per_class,
        "confusion_matrix_rows_true_cols_pred": cm.tolist(),
        "confusion_labels": cm_labels,
        "per_class": per_class_rows,
        "distance_bins": dist_rows,
        "classes_without_support": ["vegetation (no boxes; never predicted)"],
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(_clean(payload), indent=2))
    print(f"overall acc={overall_acc:.4f} mIoU={mean_iou:.4f} points={len(y_true)}", flush=True)


if __name__ == "__main__":
    main()
