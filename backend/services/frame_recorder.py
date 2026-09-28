"""Live frame recorder (Phase 1): one machine-readable frame per loop step.

Writes under ``results/recordings/<run_id>/``::

    metadata.json            run-level metadata
    frames/<frame_id>.json   per-frame metadata (relative portable paths)
    points/<frame_id>.npy    N x 4 float64 [x, y, z, intensity] (replay convention)
    maps/MAP_<seq>.npz       adaptive 2.5D map arrays (reconstructable)

Called once from ``pybullet_live._loop_step`` with the EXACT objects the
loop already produced — nothing is recomputed. Failures raise
``RecorderError``; the caller logs, exposes status, and continues
autonomy (persistence is never a hard fault for the control loop).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger("paradox.backend.recorder")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RECORDINGS_ROOT = PROJECT_ROOT / "results" / "recordings"

MAP_FIELDS = ("x", "y", "elevation", "occupancy", "resolution",
              "importance", "semantic_class", "confidence")


class RecorderError(RuntimeError):
    pass


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def validate_points(points: np.ndarray) -> np.ndarray:
    """Enforce the replay N x 4 XYZI convention (numeric, finite)."""
    try:
        arr = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RecorderError(f"points not numeric: {exc}") from exc
    if arr.dtype == object:
        raise RecorderError("points have object dtype")
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise RecorderError(f"points must be N x 4, got {arr.shape!r}")
    if arr.shape[0] == 0:
        raise RecorderError("points are empty")
    if not np.all(np.isfinite(arr)):
        raise RecorderError("points contain NaN/inf")
    return np.ascontiguousarray(arr)


def run_dir(run_id: str) -> Path:
    return RECORDINGS_ROOT / str(run_id)


def record_frame(run_id: str, sequence: int, loop_frame_id: str,
                 scenario: str | None, timestamp: float | None,
                 lidar_points: np.ndarray,
                 pipeline_result: Dict[str, Any],
                 state_vector: List[float],
                 jev: Dict[str, Any], safety: Dict[str, Any],
                 execution: Dict[str, Any],
                 vehicle: Dict[str, Any]) -> Dict[str, Any]:
    """Persist one complete frame. Returns {frame_id, paths...}.

    ``jev``/``safety``/``execution``/``vehicle`` are the loop's own
    dicts (decision record, safety verdict, executor info, vehicle
    state) — recorded verbatim, never reinterpreted. No secrets are
    written (callers never supply any).
    """
    pts = validate_points(lidar_points)
    frame_id = f"{run_id}_f{int(sequence):05d}"
    root = run_dir(run_id)
    for sub in ("frames", "points", "maps"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    points_rel = f"points/{frame_id}.npy"
    map_rel = f"maps/MAP_{int(sequence):05d}.npz"
    meta_rel = f"frames/{frame_id}.json"
    try:
        np.save(root / points_rel, pts)
        cells = list((pipeline_result or {}).get("map_cells", []))
        _save_map_npz(root / map_rel, cells)
        meta = {
            "frame_id": frame_id,
            "loop_frame_id": loop_frame_id,
            "run_id": run_id,
            "sequence": int(sequence),
            "timestamp": timestamp,
            "scenario": scenario,
            "created_utc": _utc(),
            "lidar": {"points_path": points_rel,
                      "point_count": int(pts.shape[0])},
            "vehicle": {k: vehicle.get(k) for k in
                        ("x", "y", "z", "yaw_deg", "speed_ms", "yaw_rate")
                        if isinstance(vehicle, dict)},
            "pipeline": {
                "map_cells": len(cells),
                "importance": (pipeline_result or {}).get("importance"),
                "resolution": (pipeline_result or {}).get("resolution"),
                "timing": (pipeline_result or {}).get("timing"),
            },
            "stage6": {"state_vector": [float(v) for v in (state_vector or [])]},
            "jev": {k: (jev or {}).get(k) for k in
                    ("source", "mode", "decision_model", "model",
                     "proposed_action", "confidence", "answer_confidence",
                     "engine_confidence_raw", "probabilities", "jev_status",
                     "engine_status", "policy_latency_ms", "gate",
                     "gate_metric", "gate_threshold",
                     "temperature_applied", "eligible_actions",
                     "ineligible_actions", "category",
                     "raw_laya_action", "raw_laya_probabilities",
                     "raw_laya_confidence", "raw_laya_status",
                     "constrained_action", "laya_skipped", "skip_reason",
                     "model_id", "checkpoint_revision", "device",
                     "runtime_version", "calibration_version",
                     "request_id", "error")
                    if isinstance(jev, dict)},
            "laya": {k: (jev or {}).get(k) for k in
                     ("mode", "model_id", "checkpoint_revision", "device",
                      "runtime_version", "calibration_version",
                      "eligible_actions", "raw_laya_action",
                      "constrained_action", "gate_metric", "gate_threshold")
                     if isinstance(jev, dict)},
            "safety": {
                "status": (safety or {}).get("verdict", (safety or {}).get("status")),
                "override": bool((safety or {}).get("safety_override",
                                  (safety or {}).get("verdict") not in (None, "SAFE_TO_EXECUTE"))),
                "reason": (safety or {}).get("reason", (safety or {}).get("safety_reason")),
                "emergency_flag": bool((safety or {}).get("emergency_stop", False)),
                "nearest_obstacle_m": (safety or {}).get("nearest_forward_obstacle_m"),
                "forward_clearance_m": (safety or {}).get("forward_clearance_m"),
                "left_clearance_m": (safety or {}).get("left_clearance_m"),
                "right_clearance_m": (safety or {}).get("right_clearance_m"),
            },
            "execution": {k: (execution or {}).get(k) for k in
                          ("action", "executed_action", "source", "collision",
                           "pose_before", "pose_after", "speed_ms", "frame")
                          if isinstance(execution, dict)},
        }
        (root / meta_rel).write_text(json.dumps(meta, indent=2, default=str))
    except (OSError, ValueError, TypeError) as exc:
        raise RecorderError(f"frame {frame_id} write failed: {exc}") from exc
    return {"frame_id": frame_id, "sequence": int(sequence),
            "points_path": points_rel, "map_path": map_rel,
            "metadata_path": meta_rel,
            "metadata": meta, "map_cells": cells}


def _save_map_npz(path: Path, cells: List[Dict[str, Any]]) -> None:
    arrays: Dict[str, Any] = {}
    for f in ("x", "y", "elevation", "occupancy", "resolution",
              "importance", "confidence"):
        vals = []
        for c in cells:
            try:
                v = c.get(f, None)
                vals.append(float(v) if v is not None else np.nan)
            except (TypeError, ValueError):
                vals.append(np.nan)
        arrays[f] = np.asarray(vals, dtype=np.float64)
    arrays["semantic_class"] = np.asarray(
        [str(c.get("semantic_class", "unknown")) for c in cells], dtype="U32")
    arrays["point_count"] = np.asarray(
        [int(c.get("point_count", 0) or 0) for c in cells], dtype=np.int64)
    np.savez_compressed(path, **arrays)


def load_map_npz(path: Path) -> Dict[str, Any]:
    """Reload a map artifact into dashboard-style cell dicts."""
    try:
        z = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise RecorderError(f"cannot read map {path}: {exc}") from exc
    n = len(z["x"])
    cells = []
    for i in range(n):
        cells.append({
            "x": float(z["x"][i]), "y": float(z["y"][i]),
            "elevation": float(z["elevation"][i]),
            "occupancy": float(z["occupancy"][i]),
            "resolution": float(z["resolution"][i]),
            "importance": float(z["importance"][i]),
            "semantic_class": str(z["semantic_class"][i]),
            "semantic_source": "recorded",
            "confidence": (None if float(z["confidence"][i]) != float(z["confidence"][i])
                           else float(z["confidence"][i])),
            "point_count": int(z["point_count"][i]),
        })
    return {"cells": cells, "count": n}


def write_run_metadata(run_id: str, metadata: Dict[str, Any]) -> Path:
    root = run_dir(run_id)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "metadata.json"
    path.write_text(json.dumps({"run_id": run_id, **metadata,
                                "written_utc": _utc()}, indent=2, default=str))
    return path


def recorder_status() -> Dict[str, Any]:
    """Recorder health (never raises into callers)."""
    try:
        RECORDINGS_ROOT.mkdir(parents=True, exist_ok=True)
        probe = RECORDINGS_ROOT / ".writetest"
        probe.write_text("ok")
        probe.unlink()
        writable = True
    except OSError:
        writable = False
    runs = 0
    try:
        runs = sum(1 for p in RECORDINGS_ROOT.iterdir() if p.is_dir())
    except OSError:
        pass
    return {"root": str(RECORDINGS_ROOT), "writable": writable,
            "recorded_runs": runs}
