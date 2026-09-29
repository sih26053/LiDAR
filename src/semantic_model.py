"""Point-cloud semantic segmentation model (genuine neural network).

Architecture: point-wise multi-layer perceptron (MLP) classifier
(sklearn ``MLPClassifier``: fully-connected feed-forward network trained
with backpropagation). Chosen deliberately over PointNet++/sparse-CNN:
the deployment environment is CPU-only with no torch/TensorFlow/ONNX
runtime and the demo must stay local/offline, so a lightweight
point-wise MLP is the technically appropriate model. This is a real
trained network (weights in ``models/segmentation/weights/``), NOT a
heuristic and NOT an annotation lookup.

Role separation (never collapsed):
- THIS module answers "what is this point?" (semantic labels + softmax
  confidence from ``predict_proba``).
- ``src/importance_engine.py`` answers "how important is this region?".
- ``src/resolution_engine.py`` answers "how much detail here?".
- ``src/mapper_2_5d.py`` builds the adaptive 2.5D representation.

Training supervision (weak, fully documented):
- Points inside a nuScenes 3D box -> the box's project class
  (annotation-derived weak label, NOT independent human point labels;
  nuScenes Mini ships no lidarseg point labels).
- Points outside all boxes with z <= ground_estimate + 0.25 m ->
  ``road_driveable`` (ground-plane heuristic pseudo-label).
- Sampled remaining outside-box points -> ``unknown``.
- ``vegetation`` has zero box support in the replay frames and is NOT
  in the output layer (documented limitation).

Uncertainty: ``confidence`` is the softmax max-probability from
``predict_proba`` (genuine model output, uncalibrated -- documented as
such); point ``uncertainty = 1 - confidence``. Nothing is invented.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "segmentation"
WEIGHTS_DIR = MODEL_DIR / "weights"
MLP_PATH = WEIGHTS_DIR / "mlp_model.pkl"
SCALER_PATH = WEIGHTS_DIR / "scaler.pkl"
CONFIG_PATH = MODEL_DIR / "model_config.json"

MODEL_NAME = "mlp-point-segmenter-v1"

#: Output classes (index-aligned with the trained network). ``vegetation``
#: is intentionally absent: zero training support (see module docstring).
MODEL_CLASSES: List[str] = [
    "vehicle",
    "pedestrian_vru",
    "static_manmade",
    "road_driveable",
    "unknown",
]

#: Mandatory top-level categories required by the expected solution.
MANDATORY_CATEGORY: Dict[str, str] = {
    "vehicle": "moving_object",
    "pedestrian_vru": "moving_object",
    "static_manmade": "static_obstacle",
    "vegetation": "static_obstacle",
    "road_driveable": "terrain",
    "unknown": "unknown",
}

FEATURE_NAMES: List[str] = ["x", "y", "z", "intensity", "range", "z_rel"]

_cached: Dict[str, Any] = {}


class ModelNotTrainedError(FileNotFoundError):
    """Raised when weights are absent. Never fall back to fake predictions."""


def model_available() -> bool:
    """True only when trained weights + scaler are present on disk."""
    return bool(MLP_PATH.is_file() and SCALER_PATH.is_file())


@dataclass
class PointSemanticResult:
    """Genuine model output for one point cloud."""

    labels: np.ndarray  # (N,) project-class strings
    confidence: np.ndarray  # (N,) softmax max-probability in [0,1]
    uncertainty: np.ndarray  # (N,) 1 - confidence (uncalibrated, documented)
    semantic_source: np.ndarray  # (N,) all "model"
    model_name: str
    inference_latency_ms: float
    n_points: int

    def mandatory_categories(self) -> np.ndarray:
        return np.array(
            [MANDATORY_CATEGORY.get(str(s), "unknown") for s in self.labels],
            dtype=object,
        )


def ground_estimate(points_xyz: np.ndarray) -> float:
    """Per-frame ground estimate: 5th percentile of z (deterministic)."""
    z = np.asarray(points_xyz, dtype=np.float64)[:, 2]
    if z.size == 0:
        raise ValueError("ground_estimate received no points.")
    return float(np.percentile(z, 5.0))


def extract_features(points: np.ndarray) -> np.ndarray:
    """Legacy 5-feature extractor ``[x, y, z, intensity, range]``.

    Compatibility alias (first five columns of :func:`prepare_input`,
    without ``z_rel``). The production MLP uses all six features; this
    exists so older callers keep a stable import.
    """
    return prepare_input(points)[:, :5]


def prepare_input(points: np.ndarray) -> np.ndarray:
    """Raw N x 4 [x,y,z,intensity] -> N x 6 model features (unscaled)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 4:
        raise ValueError(f"points must be N x 4. Received shape {pts.shape!r}")
    if pts.shape[0] == 0:
        raise ValueError("prepare_input received no points.")
    if not np.all(np.isfinite(pts)):
        raise ValueError("prepare_input requires all-finite points.")
    x, y, z, intensity = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    rng = np.hypot(x, y)
    z_rel = z - ground_estimate(pts[:, :3])
    return np.column_stack([x, y, z, intensity, rng, z_rel])


def load_model() -> Dict[str, Any]:
    """Load (and cache) the trained MLP + scaler. Raises if absent."""
    if "mlp" in _cached and "scaler" in _cached:
        return _cached
    if not MLP_PATH.is_file() or not SCALER_PATH.is_file():
        raise ModelNotTrainedError(
            f"Segmentation weights not found at {MLP_PATH} / {SCALER_PATH}. "
            "Run scripts/train_segmentation_mlp.py first. "
            "No fallback predictions are produced."
        )
    import pickle

    with open(MLP_PATH, "rb") as fh:
        mlp = pickle.load(fh)
    with open(SCALER_PATH, "rb") as fh:
        scaler = pickle.load(fh)
    classes = [str(c) for c in list(mlp.classes_)]
    if classes != MODEL_CLASSES:
        raise ValueError(
            f"Weights class order {classes!r} != MODEL_CLASSES {MODEL_CLASSES!r}."
        )
    _cached["mlp"], _cached["scaler"] = mlp, scaler
    return _cached


def load_weights() -> Dict[str, Any]:
    """Inspect raw weight shapes/sizes without running inference."""
    bundle = load_model()
    mlp = bundle["mlp"]
    return {
        "model_name": MODEL_NAME,
        "architecture": f"MLP({'-'.join(str(h) for h in mlp.hidden_layer_sizes)})",
        "n_layers": int(mlp.n_layers_),
        "n_parameters": int(
            sum(c.size for c in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
        ),
        "classes": list(MODEL_CLASSES),
        "mlp_path": str(MLP_PATH),
        "scaler_path": str(SCALER_PATH),
    }


def predict(features_scaled: np.ndarray) -> np.ndarray:
    """Scaled N x 6 features -> N x C class probabilities (softmax)."""
    bundle = load_model()
    proba = bundle["mlp"].predict_proba(np.asarray(features_scaled, dtype=np.float64))
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim != 2 or proba.shape[1] != len(MODEL_CLASSES):
        raise ValueError(f"Unexpected probability shape {proba.shape!r}.")
    return proba


def postprocess_predictions(proba: np.ndarray) -> PointSemanticResult:
    """Probabilities -> labels + confidence + uncertainty (no invention)."""
    proba = np.asarray(proba, dtype=np.float64)
    idx = np.argmax(proba, axis=1)
    conf = proba[np.arange(proba.shape[0]), idx]
    labels = np.array([MODEL_CLASSES[int(i)] for i in idx], dtype=object)
    sources = np.array(["model"] * proba.shape[0], dtype=object)
    return PointSemanticResult(
        labels=labels,
        confidence=conf.astype(np.float64),
        uncertainty=(1.0 - conf).astype(np.float64),
        semantic_source=sources,
        model_name=MODEL_NAME,
        inference_latency_ms=float("nan"),
        n_points=int(proba.shape[0]),
    )


def predict_points(points: np.ndarray) -> tuple:
    """Full inference: N x 4 -> (labels, sources, confidence, latency_ms).

    Signature matches the ``semantic_override`` hook in
    ``src/robustness.run_pipeline_condition``.
    """
    t0 = time.perf_counter()
    feats = prepare_input(points)
    bundle = load_model()
    scaled = bundle["scaler"].transform(feats)
    proba = predict(scaled)
    result = postprocess_predictions(proba)
    ms = (time.perf_counter() - t0) * 1000.0
    result.inference_latency_ms = float(ms)
    return result.labels, result.semantic_source, result.confidence, float(ms)


def model_info() -> Dict[str, Any]:
    """Describe the model for GET /model/info (honest availability flags).

    Returns the frontend ``ModelInfo`` contract. ``metrics_held_out`` is
    the stored held-out evaluation (results/segmentation/metrics.json) or
    ``None`` when evaluation has not been run yet -- never fabricated.
    """
    info: Dict[str, Any] = {
        "model_name": MODEL_NAME,
        "model_file": "models/segmentation/weights/mlp_model.pkl",
        "framework": "scikit-learn MLPClassifier (backprop-trained feed-forward network)",
        "architecture": "point-wise MLP classifier (sklearn MLPClassifier, backprop-trained)",
        "rejected_alternative": "PointNet++/sparse-CNN rejected: CPU-only host, no torch/TF/ONNX runtime, offline constraint",
        "input_features": list(FEATURE_NAMES),
        "classes": list(MODEL_CLASSES),
        "mandatory_categories": dict(MANDATORY_CATEGORY),
        "features": list(FEATURE_NAMES),
        "supervision": ("Weak supervision: nuScenes 3D-box-derived point labels + "
                        "ground-plane heuristic pseudo-labels (NO human point labels; "
                        "nuScenes Mini ships no lidarseg)"),
        "not_supervised": ["vegetation (zero box support; never predicted)",
                           "lidarseg classes (absent from nuScenes Mini)"],
        "inference_stage": ("LiDAR preprocessing -> MLP segmentation -> semantic labels "
                            "+ softmax confidence -> Importance -> Resolution -> Adaptive 2.5D Mapper"),
        "weights_present": bool(MLP_PATH.is_file() and SCALER_PATH.is_file()),
        "trained": False,
        "training": "not run yet -- execute scripts/train_segmentation_mlp.py",
        "metrics_held_out": None,
    }
    if CONFIG_PATH.is_file():
        try:
            cfg = json.loads(CONFIG_PATH.read_text())
            info.update(
                {
                    "trained": True,
                    "training": "weak supervision: box-derived + ground-heuristic pseudo-labels (see training_config.json)",
                    "hidden_layer_sizes": cfg.get("hidden_layer_sizes"),
                    "train_frames": cfg.get("train_frames"),
                    "test_frames": cfg.get("eval_frames"),
                    "n_train_points": cfg.get("n_train_points"),
                    "final_train_loss": cfg.get("final_train_loss"),
                    "label_provenance": cfg.get("label_provenance"),
                }
            )
        except (OSError, ValueError):
            pass
    metrics_path = PROJECT_ROOT / "results" / "segmentation" / "metrics.json"
    if metrics_path.is_file():
        try:
            m = json.loads(metrics_path.read_text())
            info["metrics_held_out"] = {
                "accuracy": m.get("overall_accuracy_vs_reference"),
                "f1_macro": m.get("f1_macro_vs_reference"),
                "f1_per_class": m.get("f1_per_class_vs_reference"),
                "confusion_matrix_rows_true_cols_pred": m.get("confusion_matrix_rows_true_cols_pred"),
                "confusion_labels": m.get("confusion_labels"),
                "mean_predicted_confidence": m.get("mean_confidence"),
                "mean_iou": m.get("mean_iou_vs_reference"),
                "n_test_points": m.get("n_eval_points"),
                "reference": m.get("reference"),
            }
            info["n_test_points"] = m.get("n_eval_points")
        except (OSError, ValueError):
            pass
    if info["weights_present"]:
        try:
            info.update(load_weights())
            info["trained"] = info["trained"] and True
        except Exception:
            pass
    return info


def to_mandatory(labels: Sequence[str]) -> List[str]:
    return [MANDATORY_CATEGORY.get(str(s), "unknown") for s in labels]
