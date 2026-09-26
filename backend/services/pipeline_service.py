"""Pipeline orchestration service (Step 4). The ONLY place pipeline logic lives.

Flow: InputFrame -> replay_service -> preprocessing -> semantic channel
(annotation reference OR trained MLP point classifier) -> feature
extraction -> Importance Engine -> Resolution Engine -> Adaptive 2.5D
Mapper -> structured PipelineResult dict.

The trained segmentation model lives in ``models/segmentation/`` and is
invoked only through ``src/semantic_model.predict_points`` (the
``semantic_override`` hook in ``src/robustness``); it never touches the
Importance/Resolution engines directly. Reuses
``src/robustness.run_pipeline_condition`` (which itself reuses
preprocessing / semantic_adapter / mapper_2_5d verbatim) with engines
bound to the frozen final config. No weights/thresholds are duplicated
here -- they come from ``backend/config.py``.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from backend.config import load_failsafe_config, load_final_config
from backend.services import replay_service
from backend.services.replay_service import DataLoadError, FrameNotFoundError

logger = logging.getLogger("paradox.backend.pipeline")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NUSC_ROOT = PROJECT_ROOT / "data" / "raw" / "nuscenes"

_nusc = None
_importance_engine = None
_resolution_engine = None


class PipelineStageError(RuntimeError):
    def __init__(self, stage: str, code: str, message: str):
        super().__init__(message)
        self.stage = stage
        self.code = code


def _engines():
    """Build frozen engines from the final config (no hard-coded values)."""
    global _importance_engine, _resolution_engine
    if _importance_engine is not None and _resolution_engine is not None:
        return _importance_engine, _resolution_engine
    from src.importance_engine import ImportanceEngine
    from src.resolution_engine import ResolutionEngine

    cfg = load_final_config()
    _importance_engine = ImportanceEngine(
        weights=dict(cfg["importance_weights"]),
        max_distance=float(cfg["max_distance_m"]),
        lambda_uncertainty=float(cfg["uncertainty_lambda"]),
    )
    levels = [(float(t), float(r)) for t, r in cfg["resolution_levels"]]
    _resolution_engine = ResolutionEngine(levels=levels)
    # Assert parity with frozen config (fail loudly on drift).
    assert _importance_engine.weights == {k: float(v) for k, v in cfg["importance_weights"].items()}
    assert [tuple(x) for x in _resolution_engine.levels] == sorted(levels, key=lambda p: p[0], reverse=True)
    return _importance_engine, _resolution_engine


def _nusc_dataset():
    global _nusc
    if _nusc is not None:
        return _nusc
    from nuscenes.nuscenes import NuScenes

    if not NUSC_ROOT.is_dir():
        raise PipelineStageError("loading", "DATA_LOAD_FAILED", f"nuScenes Mini dataset missing at {NUSC_ROOT}")
    try:
        _nusc = NuScenes(version="v1.0-mini", dataroot=str(NUSC_ROOT), verbose=False)
    except Exception as exc:
        raise PipelineStageError("loading", "DATA_LOAD_FAILED", f"Cannot open nuScenes Mini: {exc}") from exc
    return _nusc


def _safe_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def run_frame(frame_id: str, max_map_cells: int | None = None,
              semantic_mode: str = "annotation") -> Dict[str, Any]:
    """Execute the frozen pipeline for one frame; return a JSON-safe dict.

    ``semantic_mode="annotation"`` (default) uses nuScenes box annotations
    with heuristic fallback. ``semantic_mode="model"`` uses the trained
    MLP point classifier (models/segmentation/weights/mlp_model.pkl) with
    real softmax confidence; regions then carry semantic_source "model".
    """
    mode = str(semantic_mode or "annotation").lower()
    if mode not in ("annotation", "model"):
        raise PipelineStageError("input", "INVALID_REQUEST",
                                 f"semantic_mode must be 'annotation' or 'model', got {semantic_mode!r}")
    if mode == "model":
        from src.semantic_model import model_available

        if not model_available():
            raise PipelineStageError("perception", "MODEL_UNAVAILABLE",
                                     "Trained model weights missing; run scripts/train_segmentation_mlp.py first.")
    fid = str(frame_id)
    t_wall0 = time.perf_counter()
    try:
        raw_points, meta = replay_service.load_frame_points(fid)
    except FrameNotFoundError as exc:
        raise PipelineStageError("input", "FRAME_NOT_FOUND", str(exc)) from exc
    except DataLoadError as exc:
        raise PipelineStageError("loading", "DATA_LOAD_FAILED", str(exc)) from exc

    nusc = _nusc_dataset()
    try:
        sample = nusc.get("sample", fid)
    except Exception as exc:
        raise PipelineStageError("loading", "DATA_LOAD_FAILED", f"Sample {fid} not in dataset: {exc}") from exc
    return _execute(raw_points, sample, meta, fid, mode, max_map_cells, t_wall0)


def run_sample_token(sample_token: str, max_map_cells: int | None = None,
                     semantic_mode: str = "annotation") -> Dict[str, Any]:
    """Run the pipeline for any raw nuScenes sample (not only manifest frames).

    Loads the raw LIDAR_TOP .bin via src/lidar_loader and preprocesses it —
    the path used for in-scene consecutive-sample sequences (tracking).
    """
    from src.lidar_loader import load_sample_lidar

    mode = str(semantic_mode or "annotation").lower()
    if mode not in ("annotation", "model"):
        raise PipelineStageError("input", "INVALID_REQUEST",
                                 f"semantic_mode must be 'annotation' or 'model', got {semantic_mode!r}")
    tok = str(sample_token)
    t_wall0 = time.perf_counter()
    nusc = _nusc_dataset()
    try:
        sample = nusc.get("sample", tok)
    except Exception as exc:
        raise PipelineStageError("loading", "DATA_LOAD_FAILED", f"Sample {tok} not in dataset: {exc}") from exc
    try:
        raw_points, info = load_sample_lidar(nusc, tok)
    except (FileNotFoundError, ValueError) as exc:
        raise PipelineStageError("loading", "DATA_LOAD_FAILED", str(exc)) from exc
    try:
        scene = nusc.get("scene", sample["scene_token"])
        scene_id = scene.get("name")
    except Exception:
        scene_id = None
    meta = {"frame_id": tok, "scene_id": scene_id,
            "timestamp": float(info.get("timestamp", sample.get("timestamp", 0.0))),
            "source_path": info.get("source_file", "")}
    return _execute(raw_points, sample, meta, tok, mode, max_map_cells, t_wall0)


def _execute(raw_points, sample, meta, fid: str, mode: str,
             max_map_cells: int | None, t_wall0: float,
             live_input: bool = False) -> Dict[str, Any]:
    """Shared pipeline tail: engines -> condition -> serialize -> result."""
    from src.robustness import run_pipeline_condition

    importance_engine, resolution_engine = _engines()
    failsafe_cfg = load_failsafe_config()
    final_cfg = load_final_config()
    nusc = _nusc_dataset()

    frame_metadata = {"frame_id": fid}
    condition = {"condition": "original", "condition_type": "original", "condition_parameter": None, "seed": int(final_cfg.get("random_seed", 42))}
    # Model-first pipeline: the trained point classifier produces the
    # semantic labels consumed by Importance -> Resolution -> Mapper.
    # Annotation boxes are used ONLY for the separate evaluation branch
    # below (never fed into the model path as fake inference).
    try:
        rec = run_pipeline_condition(
            raw_points, frame_metadata, sample, nusc,
            importance_engine, resolution_engine, condition,
            cell_size=float(final_cfg["integration_cell_size_m"]),
            failsafe_config=failsafe_cfg,
            semantic_override={"mode": "model"} if mode == "model" else None,
        )
    except Exception as exc:
        if "semantic_model" in str(type(exc).__module__) or "ModelNotTrained" in type(exc).__name__:
            raise PipelineStageError("perception", "PERCEPTION_FAILED",
                                     f"Segmentation model unavailable for frame {fid}: {exc}"[:500]) from exc
        raise
    model_eval = _evaluate_against_annotation(fid, raw_points, sample, nusc)
    if not rec.get("success"):
        stage = str(rec.get("stage_of_failure") or "mapping")
        raise PipelineStageError(stage, _stage_code(stage), f"Pipeline failed for frame {fid}: {rec.get('failure_reason', '')}"[:500])

    map_df = rec["_map_df"]
    t_ser0 = time.perf_counter()
    cells = _serialize_cells(map_df, limit=max_map_cells)
    serialization_ms = (time.perf_counter() - t_ser0) * 1000.0
    wall_ms = (time.perf_counter() - t_wall0) * 1000.0

    res = map_df["resolution"].to_numpy(dtype=float)
    imp = map_df["importance"].to_numpy(dtype=float)
    source_counts: Dict[str, int] = {}
    if "semantic_source" in map_df.columns:
        for s, c in map_df["semantic_source"].value_counts().items():
            source_counts[str(s)] = int(c)

    result = {
        "frame_id": fid,
        "scene_id": meta.get("scene_id"),
        "timestamp": meta.get("timestamp"),
        "input_source": meta.get("source_path") or meta.get("replay_reference") or "live-stream",
        "status": "success",
        "input_point_count": int(rec.get("input_points", len(raw_points))),
        "processed_point_count": int(rec.get("processed_points", 0)),
        "map_cells": cells,
        "map_cell_count": int(len(map_df)),
        "importance": {
            "mean": _safe_float(float(imp.mean())) if len(imp) else None,
            "min": _safe_float(float(imp.min())) if len(imp) else None,
            "max": _safe_float(float(imp.max())) if len(imp) else None,
            "count": int(len(imp)),
        },
        "resolution": {
            "fine_cells": int((res == 0.05).sum()),
            "medium_cells": int((res == 0.10).sum()),
            "coarse_cells": int(((res == 0.20) | (res == 0.50)).sum()),
            "res_20cm_cells": int((res == 0.20).sum()),
            "res_50cm_cells": int((res == 0.50).sum()),
            "average_resolution": _safe_float(float(res.mean())) if len(res) else None,
            "distribution": {str(k): int((res == k).sum()) for k in (0.05, 0.10, 0.20, 0.50)},
        },
        "semantic": {
            "mode": str(rec.get("semantic_mode") or "annotation+fallback"),
            "source_counts": source_counts,
            "note": ("Trained point-classifier predictions with measured confidence; "
                     "see GET /model/info for held-out metrics."
                     if mode == "model" else
                     ("Live stream: no dataset boxes attached; annotation channel "
                      "is all-fallback by construction."
                      if live_input else
                      "Annotation-derived labels are evaluation references, never model predictions.")),
        },
        "timing": {
            "preprocessing_latency_ms": _safe_float(rec.get("preprocessing_latency_ms")),
            "perception_latency_ms": _safe_float(rec.get("perception_latency_ms")),
            "feature_extraction_latency_ms": _safe_float(rec.get("feature_extraction_latency_ms")),
            "importance_latency_ms": _safe_float(rec.get("importance_latency_ms")),
            "resolution_latency_ms": _safe_float(rec.get("resolution_latency_ms")),
            "mapping_latency_ms": _safe_float(rec.get("mapping_latency_ms")),
            "total_latency_ms": _safe_float(rec.get("total_latency_ms")),
            "serialization_latency_ms": round(serialization_ms, 3),
            "wall_clock_ms": round(wall_ms, 3),
            "fps": _safe_float(rec.get("fps")),
        },
        "model": _model_block(rec),
        "model_eval": model_eval,
    }
    logger.info(
        "frame=%s stage=mapping status=success cells=%d latency_ms=%s",
        fid, len(map_df), rec.get("total_latency_ms"),
    )
    return result


def _model_block(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Describe the segmentation model behind this frame (measured only)."""
    from src import semantic_model as sm

    info = sm.model_info()
    block: Dict[str, Any] = {
        "model_name": info.get("model_name"),
        "architecture": info.get("architecture"),
        "classes": info.get("classes"),
        "trained": bool(info.get("trained")),
        "weights_present": bool(info.get("weights_present")),
        "active": str(rec.get("semantic_mode", "")).startswith("model_prediction"),
        "inference_latency_ms": _safe_float(rec.get("perception_latency_ms"))
        if str(rec.get("semantic_mode", "")).startswith("model_prediction") else None,
        "note": ("Active: map semantics are genuine MLP outputs."
                 if str(rec.get("semantic_mode", "")).startswith("model_prediction")
                 else "Inactive for this frame: annotation reference drove the map; "
                      "see model_eval for the evaluation-only model comparison."),
    }
    return block


def _evaluate_against_annotation(fid: str, raw_points: np.ndarray,
                                 sample, nusc) -> Dict[str, Any]:
    """Evaluation-only branch: model predictions vs annotation reference.

    Runs on the same raw input for a fair per-frame comparison. Timed
    separately (``eval_only_ms``) and EXCLUDED from pipeline latency.
    Reference = box-derived weak labels (NOT human point labels).
    Never fails the frame: returns ``available: False`` with a reason.
    """
    t0 = time.perf_counter()
    try:
        from src.semantic_adapter import (
            annotations_to_sensor_frame,
            assign_annotation_semantics,
        )
        from src.semantic_model import MODEL_CLASSES, predict_points

        pts = np.asarray(raw_points, dtype=np.float64)
        anns = annotations_to_sensor_frame(nusc, sample)
        ref_labels, ref_sources, _ = assign_annotation_semantics(pts, anns)
        ref_labels = np.asarray(ref_labels, dtype=object)
        ref_sources = np.asarray(ref_sources, dtype=object)
        mask = (ref_sources == "annotation") & np.array(
            [str(s) in MODEL_CLASSES for s in ref_labels])
        labels, sources, conf, _ = predict_points(pts)
        labels = np.asarray(labels, dtype=object)
        n_ref = int(mask.sum())
        acc = float((labels[mask] == ref_labels[mask]).mean()) if n_ref else None
        return {
            "available": True,
            "accuracy_vs_annotation_reference": acc,
            "n_reference_points": n_ref,
            "mean_model_confidence": float(np.mean(conf)) if len(conf) else None,
            "eval_only_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "note": ("Per-frame model-vs-annotation agreement (box interiors only). "
                     "Reference is annotation-derived, not human point labels. "
                     "Eval-only overhead, excluded from pipeline latency."),
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": str(exc)[:300],
            "eval_only_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }


def _serialize_cells(map_df, limit: int | None = None) -> List[Dict[str, Any]]:
    df = map_df if limit is None else map_df.iloc[: int(limit)]
    cells: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        conf = row.get("confidence", None)
        try:
            conf_f = float(conf)
            conf = conf_f if math.isfinite(conf_f) else None
        except (TypeError, ValueError):
            conf = None
        cells.append(
            {
                "x": float(row["x"]),
                "y": float(row["y"]),
                "elevation": float(row["elevation"]),
                "occupancy": float(row["occupancy"]),
                "resolution": float(row["resolution"]),
                "importance": float(row["importance"]),
                "semantic_class": str(row["semantic_class"]),
                "semantic_source": str(row.get("semantic_source", "fallback")),
                "confidence": conf,
                "point_count": int(row.get("point_count", 0)),
                "region_id": int(row.get("region_id", -1)),
            }
        )
    return cells


def _stage_code(stage: str) -> str:
    s = (stage or "").lower()
    if "preprocess" in s:
        return "PREPROCESSING_FAILED"
    if "percept" in s:
        return "PERCEPTION_FAILED"
    if "aggregat" in s or "feature" in s:
        return "FEATURE_EXTRACTION_FAILED"
    if "importance" in s:
        return "IMPORTANCE_FAILED"
    if "resolution" in s:
        return "RESOLUTION_FAILED"
    if "failsafe" in s or "reject" in s:
        return "DATA_LOAD_FAILED"
    return "MAPPING_FAILED"
