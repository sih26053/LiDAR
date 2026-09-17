"""Interface validation for the common data contract.

Raises InterfaceError (a ValueError) with explicit messages instead of bare
asserts so producers get actionable feedback.
"""
from __future__ import annotations

import numpy as np

from .data_types import (
    POINT_DIM,
    LiDARFrame,
    PerceptionResult,
    RegionFeatures,
)


class InterfaceError(ValueError):
    """Raised when a data-contract object violates its schema."""


def _fail(msg: str) -> InterfaceError:
    return InterfaceError(f"Interface validation failed:\n{msg}")


def _check_01(value, name: str, context: str = "") -> None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise _fail(f"{name} must be a number.{context} Received: {value!r}")
    if not np.isfinite(v) or not (0.0 <= v <= 1.0):
        raise _fail(f"{name} must be within [0,1].{context} Received: {value!r}")


def validate_lidar_frame(frame: LiDARFrame) -> bool:
    """Validate LiDARFrame: points is finite Nx4."""
    if not isinstance(frame.frame_id, str) or not frame.frame_id:
        raise _fail("LiDARFrame.frame_id must be a non-empty str.")
    pts = frame.points
    if not isinstance(pts, np.ndarray):
        raise _fail(f"points must be numpy array. Received: {type(pts).__name__}")
    if pts.ndim != 2:
        raise _fail(f"points.ndim must be 2. Received: {pts.ndim}")
    if pts.shape[1] != POINT_DIM:
        raise _fail(
            f"points.shape[1] must be {POINT_DIM} [x, y, z, intensity]. "
            f"Received shape: {tuple(pts.shape)}"
        )
    if pts.shape[0] == 0:
        raise _fail("points must contain at least 1 point (N > 0).")
    if not np.all(np.isfinite(pts)):
        raise _fail("points must be all finite (NaN/inf not allowed).")
    return True


def validate_perception_result(pr: PerceptionResult) -> bool:
    """Validate point-level length consistency and normalized ranges."""
    validate_lidar_frame(
        LiDARFrame(frame_id=pr.frame_id, timestamp=0.0, points=pr.points)
    )
    n = int(pr.points.shape[0])
    arrays = {
        "semantic_labels": pr.semantic_labels,
        "confidence": pr.confidence,
        "distance": pr.distance,
        "elevation": pr.elevation,
        "roughness": pr.roughness,
        "point_density": pr.point_density,
    }
    for name, arr in arrays.items():
        if not isinstance(arr, np.ndarray):
            raise _fail(f"{name} must be numpy array. Received: {type(arr).__name__}")
        if arr.shape[0] != n:
            raise _fail(
                f"len({name}) must equal N={n}. Received: {arr.shape[0]}"
            )
    # 11-September backward-compatible extension: NaN confidence entries are
    # accepted as the documented "not applicable" representation for non-ML
    # semantic sources (lidarseg / annotation / fallback). Finite entries must
    # still lie within [0,1]; inf is always rejected. All previously valid
    # (all-finite [0,1]) arrays still pass unchanged.
    for i, v in enumerate(np.asarray(pr.confidence, dtype=float)):
        if np.isnan(v):
            continue  # documented N/A for non-ML sources (never ML confidence)
        if not np.isfinite(v) or not (0.0 <= v <= 1.0):
            raise _fail(
                f"confidence must be within [0,1] or NaN (not applicable for "
                f"non-ML sources). Received: {v!r} at index {i}"
            )
    for i, v in enumerate(np.asarray(pr.roughness, dtype=float)):
        if not np.isfinite(v) or not (0.0 <= v <= 1.0):
            raise _fail(
                f"roughness must be within [0,1]. Received: {v!r} at index {i}"
            )
    for i, v in enumerate(np.asarray(pr.distance, dtype=float)):
        if not np.isfinite(v) or v < 0:
            raise _fail(
                f"distance must be finite and >= 0 (metres). Received: {v!r} at index {i}"
            )
    for name in ("elevation", "point_density"):
        arr = np.asarray(getattr(pr, name), dtype=float)
        if not np.all(np.isfinite(arr)):
            raise _fail(f"{name} must be all finite. Received non-finite value.")
    return True


def validate_region_features(r: RegionFeatures) -> bool:
    """Validate region-level contract."""
    if not isinstance(r.region_id, (int, np.integer)) or int(r.region_id) < 0:
        raise _fail(f"region_id must be int >= 0. Received: {r.region_id!r}")
    if not np.isfinite(float(r.distance)) or float(r.distance) < 0:
        raise _fail(f"distance must be finite and >= 0 (metres). Received: {r.distance!r}")
    for name in (
        "semantic_importance",
        "confidence",
        "roughness",
        "dynamic_relevance",
        "uncertainty",
    ):
        _check_01(getattr(r, name), name, f" (region_id={r.region_id})")
    if not isinstance(r.point_count, (int, np.integer)) or int(r.point_count) < 0:
        raise _fail(f"point_count must be int >= 0. Received: {r.point_count!r}")
    if not isinstance(r.semantic_label, str) or not r.semantic_label:
        raise _fail("semantic_label must be a non-empty str.")
    for name in ("x", "y", "elevation", "point_density"):
        if not np.isfinite(float(getattr(r, name))):
            raise _fail(f"{name} must be finite. Received: {getattr(r, name)!r}")
    return True
