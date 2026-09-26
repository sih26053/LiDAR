"""Model + segmentation-evaluation endpoints (measured/stored only).

GET /model/info -- live model descriptor (architecture, classes, weights
  presence, training provenance). Never claims training that did not happen.
GET /segmentation/metrics -- stored held-out evaluation
  (results/segmentation/metrics.json). 404 when not yet generated.
GET /resource/metrics -- stored resource comparison
  (results/resource/resource.json). 404 when not yet generated.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.semantic_model import model_info

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SEGMENTATION_METRICS = PROJECT_ROOT / "results" / "segmentation" / "metrics.json"
RESOURCE_METRICS = PROJECT_ROOT / "results" / "resource" / "resource.json"


@router.get("/model/info")
def get_model_info():
    return model_info()


@router.post("/model/predict/{frame_id}")
def predict_frame(frame_id: str):
    """Genuine model inference on one replay frame (measured only).

    Returns per-class predicted counts, mean softmax confidence,
    agreement with the annotation reference on box interiors
    (evaluation-only, reference is annotation-derived), and timed
    inference latency. 404/503 when frame or weights are missing.
    """
    import numpy as np

    from backend.services import replay_service
    from backend.services.pipeline_service import _nusc_dataset
    from src.semantic_adapter import (
        annotations_to_sensor_frame,
        assign_annotation_semantics,
    )
    from src.semantic_model import MODEL_CLASSES, model_available, predict_points

    if not model_available():
        return JSONResponse(
            status_code=503,
            content={"message": "Trained model weights missing; run scripts/train_segmentation_mlp.py first."},
        )
    try:
        raw_points, _ = replay_service.load_frame_points(str(frame_id))
    except Exception as exc:
        return JSONResponse(status_code=404, content={"message": f"Frame not found: {exc}"})
    pts = np.asarray(raw_points, dtype=np.float64)
    labels, sources, conf, infer_ms = predict_points(pts)
    labels = np.asarray(labels, dtype=object)
    dist = {c: int((labels == c).sum()) for c in MODEL_CLASSES}
    agreement = None
    try:
        nusc = _nusc_dataset()
        sample = nusc.get("sample", str(frame_id))
        anns = annotations_to_sensor_frame(nusc, sample)
        ref_labels, ref_sources, _ = assign_annotation_semantics(pts, anns)
        ref_labels = np.asarray(ref_labels, dtype=object)
        ref_sources = np.asarray(ref_sources, dtype=object)
        mask = (ref_sources == "annotation") & np.array(
            [str(s) in MODEL_CLASSES for s in ref_labels])
        if int(mask.sum()) > 0:
            agreement = float((labels[mask] == ref_labels[mask]).mean())
    except Exception:
        agreement = None
    return {
        "frame_id": str(frame_id),
        "model_name": "mlp-point-segmenter-v1",
        "semantic_source": "model",
        "input_point_count": int(len(pts)),
        "predicted_distribution": dist,
        "mean_confidence": float(np.mean(conf)) if len(conf) else None,
        "agreement_with_annotation_foreground": agreement,
        "agreement_note": ("Share of box-interior points where the model agrees with the "
                           "box class. Evaluation-only reference (annotation-derived, "
                           "not human point labels)."),
        "timing": {"inference_latency_ms": round(float(infer_ms), 2)},
    }


def _stored(path: Path, missing_message: str):
    if not path.is_file():
        return JSONResponse(status_code=404, content={"message": missing_message})
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return JSONResponse(status_code=500, content={"message": f"Cannot read {path.name}: {exc}"})


@router.get("/segmentation/metrics")
def get_segmentation_metrics():
    return _stored(SEGMENTATION_METRICS, "Segmentation evaluation not yet generated (run scripts/evaluate_segmentation.py).")


@router.get("/resource/metrics")
def get_resource_metrics():
    return _stored(RESOURCE_METRICS, "Resource benchmark not yet generated (run scripts/measure_resources.py).")
