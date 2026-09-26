"""Train the point-wise MLP segmentation network (weak supervision, honest).

Supervision (NOT human point labels -- nuScenes Mini ships no lidarseg):
  1. Points inside a nuScenes 3D box -> box project class (vehicle /
     pedestrian_vru / static_manmade). Annotation-derived weak labels.
  2. Outside-box points with z <= frame_ground_p5 + 0.25 m ->
     road_driveable (documented ground-plane heuristic pseudo-label).
  3. Sampled remaining outside-box points -> unknown.

Frame-level split (no point leakage across split):
  train: b26e7915, c844bf5a, 858a1ece, 5991fad3 (box-rich)
  held-out eval is done separately by scripts/evaluate_segmentation.py on
  6d1d793b, dc4c80e8, 5c85640c.

Outputs:
  models/segmentation/weights/{mlp_model.pkl, scaler.pkl}
  models/segmentation/{model_config.json, label_mapping.json,
    training_config.json, training_history.csv, validation_metrics.json}

Run:  python scripts/train_segmentation_mlp.py
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services import replay_service  # noqa: E402
from backend.services.pipeline_service import _engines, _nusc_dataset  # noqa: E402
from src.semantic_adapter import (  # noqa: E402
    annotations_to_sensor_frame,
    assign_annotation_semantics,
)
from src.semantic_model import (  # noqa: E402
    FEATURE_NAMES,
    MODEL_CLASSES,
    MODEL_DIR,
    MODEL_NAME,
    WEIGHTS_DIR,
    ground_estimate,
    prepare_input,
)

TRAIN_FRAMES = [
    "b26e791522294bec90f86fd72226e35c",
    "c844bf5a9f2243ff8f4bf2c85fe218ff",
    "858a1ece22cf45d9bc71e42336604b78",
    "5991fad3280c4f84b331536c32001a04",
]
CAP_PER_CLASS = 4000
GROUND_MARGIN_M = 0.25
RANDOM_STATE = 42


def weak_labels(points: np.ndarray, sample, nusc) -> np.ndarray:
    """Per-point weak-label strings ('' = excluded from training)."""
    anns = annotations_to_sensor_frame(nusc, sample)
    labels, sources, _ = assign_annotation_semantics(points, anns)
    labels = np.asarray(labels, dtype=object)
    sources = np.asarray(sources, dtype=object)
    out = np.array([""] * len(points), dtype=object)
    annotated = sources == "annotation"
    out[annotated] = labels[annotated]  # vehicle / pedestrian_vru / static_manmade
    g = ground_estimate(points[:, :3])
    outside = ~annotated
    ground = outside & (points[:, 2] <= g + GROUND_MARGIN_M)
    out[ground] = "road_driveable"
    rest = outside & ~ground
    out[rest] = "unknown"
    # Keep only classes the network actually outputs.
    keep = np.array([str(s) in MODEL_CLASSES for s in out])
    out[~keep] = ""
    return out


def main() -> None:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    t_all = time.perf_counter()
    _engines()  # validate frozen config parity (pipeline unchanged)
    nusc = _nusc_dataset()
    rng = np.random.RandomState(RANDOM_STATE)

    X_parts, y_parts = [], []
    per_frame_counts = {}
    for fid in TRAIN_FRAMES:
        raw, _ = replay_service.load_frame_points(fid)
        sample = nusc.get("sample", fid)
        wl = weak_labels(np.asarray(raw, dtype=np.float64), sample, nusc)
        feats = prepare_input(np.asarray(raw, dtype=np.float64))
        frame_count = {}
        for cls in MODEL_CLASSES:
            idx = np.flatnonzero(wl == cls)
            frame_count[cls] = int(len(idx))
            if len(idx) > CAP_PER_CLASS:
                idx = rng.choice(idx, size=CAP_PER_CLASS, replace=False)
            if len(idx):
                X_parts.append(feats[idx])
                y_parts.append(np.array([cls] * len(idx), dtype=object))
        per_frame_counts[fid[:8]] = frame_count
        print(f"train frame {fid[:8]}: {frame_count}", flush=True)

    X = np.vstack(X_parts).astype(np.float64)
    y_str = np.concatenate(y_parts)
    # Integer codes aligned to MODEL_CLASSES order (sklearn validation
    # scoring cannot handle object-dtype string labels).
    code_of = {c: i for i, c in enumerate(MODEL_CLASSES)}
    y = np.array([code_of[str(s)] for s in y_str], dtype=np.int64)
    print(f"training matrix: {X.shape}, classes: {sorted(set(y_str.tolist()))}", flush=True)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=512,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    t0 = time.perf_counter()
    mlp.fit(Xs, y)
    train_ms = (time.perf_counter() - t0) * 1000.0
    # Restore project-class labels in trained column order (codes were
    # MODEL_CLASSES-aligned, so this is exact, not a remap).
    mlp.classes_ = np.array(list(MODEL_CLASSES), dtype=object)
    print(
        f"trained: n_iter={mlp.n_iter_} loss={mlp.loss_:.4f} "
        f"best_val={mlp.best_validation_score_:.4f} time={train_ms / 1000:.1f}s",
        flush=True,
    )

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_DIR / "mlp_model.pkl", "wb") as fh:
        pickle.dump(mlp, fh)
    with open(WEIGHTS_DIR / "scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)

    (MODEL_DIR / "training_history.csv").write_text(
        pd.DataFrame(
            {"epoch": np.arange(1, len(mlp.loss_curve_) + 1),
             "train_loss": mlp.loss_curve_}
        ).to_csv(index=False)
    )
    (MODEL_DIR / "validation_metrics.json").write_text(
        json.dumps(
            {
                "n_iter": int(mlp.n_iter_),
                "final_train_loss": float(mlp.loss_),
                "best_validation_score": float(mlp.best_validation_score_),
                "validation_fraction": 0.15,
                "note": "sklearn internal validation split (15% of the training matrix); "
                "frame-level held-out evaluation is in results/segmentation/.",
            },
            indent=2,
        )
    )
    (MODEL_DIR / "training_config.json").write_text(
        json.dumps(
            {
                "model_name": MODEL_NAME,
                "architecture": "sklearn MLPClassifier",
                "hidden_layer_sizes": [64, 32],
                "activation": "relu",
                "solver": "adam",
                "alpha": 1e-4,
                "batch_size": 512,
                "learning_rate_init": 1e-3,
                "max_iter": 500,
                "early_stopping": True,
                "validation_fraction": 0.15,
                "n_iter_no_change": 25,
                "random_state": RANDOM_STATE,
                "features": FEATURE_NAMES,
                "classes": MODEL_CLASSES,
                "cap_per_class": CAP_PER_CLASS,
                "ground_margin_m": GROUND_MARGIN_M,
                "train_frames": TRAIN_FRAMES,
                "per_frame_weak_label_counts": per_frame_counts,
                "n_train_points": int(len(y)),
                "train_wall_ms": round((time.perf_counter() - t_all) * 1000.0, 1),
            },
            indent=2,
        )
    )
    (MODEL_DIR / "model_config.json").write_text(
        json.dumps(
            {
                "model_name": MODEL_NAME,
                "architecture": "point-wise MLP classifier (sklearn MLPClassifier, backprop-trained)",
                "hidden_layer_sizes": [64, 32],
                "features": FEATURE_NAMES,
                "classes": MODEL_CLASSES,
                "train_frames": TRAIN_FRAMES,
                "eval_frames": [
                    "6d1d793b68b24165be5755a54054c82a",
                    "dc4c80e8a1534ff8a3eb80307341161c",
                    "5c85640c10d9485c86bf6eb44778c002",
                ],
                "n_train_points": int(len(y)),
                "final_train_loss": float(mlp.loss_),
                "label_provenance": "weak supervision: nuScenes box-derived labels + "
                "ground-plane heuristic pseudo-labels; NO human point labels "
                "(nuScenes Mini ships no lidarseg)",
            },
            indent=2,
        )
    )
    import shutil

    shutil.copy(
        PROJECT_ROOT / "config" / "semantic_class_mapping.json",
        MODEL_DIR / "label_mapping.json",
    )
    print("saved weights + configs to models/segmentation/", flush=True)


if __name__ == "__main__":
    main()
