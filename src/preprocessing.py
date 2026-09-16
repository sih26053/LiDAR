"""LiDAR preprocessing (10 September pipeline, Rajashree side).

Raw M x 4 -> processed N x 4 ``[x, y, z, intensity]``:

1. NaN/Inf removal (any non-finite coordinate or intensity is dropped).
2. Invalid-point filtering (zero-range returns at the sensor origin are
   dropped: ``hypot(x, y, z) <= 0``).
3. ROI filtering (generous engineering defaults, configurable -- see
   ``ROI_X``, ``ROI_Y``, ``ROI_Z``; chosen to keep the full usable nuScenes
   sweep while cutting far-field noise, NOT to tune results).

Output per sample: ``<sample_token>_LIDAR_TOP_xyzi.npy`` (float64 N x 4,
all finite) + ``<sample_token>_metadata.csv`` with at minimum
``frame_id, sample_token, lidar_token, timestamp, source_file,
n_raw, n_processed``. Raises (never silently imputes) on empty results.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple
import numpy as np
import pandas as pd

# Engineering ROI defaults (metres, documented -- not tuned results).
ROI_X = (-80.0, 80.0)
ROI_Y = (-80.0, 80.0)
ROI_Z = (-10.0, 15.0)


def preprocess_points(
    points: np.ndarray,
    roi_x: Tuple[float, float] = ROI_X,
    roi_y: Tuple[float, float] = ROI_Y,
    roi_z: Tuple[float, float] = ROI_Z,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Filter raw M x 4 -> clean N x 4. Returns (clean, counts dict)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 4:
        raise ValueError(f"Raw points must be M x 4. Received shape {pts.shape!r}.")
    n_raw = int(pts.shape[0])
    if n_raw == 0:
        raise ValueError("Raw points are empty.")

    finite_mask = np.isfinite(pts).all(axis=1)
    n_nonfinite = int(n_raw - int(finite_mask.sum()))
    pts = pts[finite_mask]

    r = np.linalg.norm(pts[:, :3], axis=1)
    valid_mask = r > 0.0
    n_zero_range = int(len(pts) - int(valid_mask.sum()))
    pts = pts[valid_mask]

    roi_mask = (
        (pts[:, 0] >= roi_x[0]) & (pts[:, 0] <= roi_x[1])
        & (pts[:, 1] >= roi_y[0]) & (pts[:, 1] <= roi_y[1])
        & (pts[:, 2] >= roi_z[0]) & (pts[:, 2] <= roi_z[1])
    )
    n_outside_roi = int(len(pts) - int(roi_mask.sum()))
    pts = np.ascontiguousarray(pts[roi_mask])

    if pts.shape[0] == 0:
        raise ValueError("Preprocessing removed all points; nothing to save.")
    if not np.all(np.isfinite(pts)):
        raise ValueError("Preprocessed points must be all finite.")
    counts = {
        "n_raw": n_raw,
        "n_nonfinite_removed": n_nonfinite,
        "n_zero_range_removed": n_zero_range,
        "n_outside_roi_removed": n_outside_roi,
        "n_processed": int(pts.shape[0]),
    }
    return pts, counts


def save_processed_frame(
    points: np.ndarray,
    frame_metadata: dict,
    counts: dict,
    processed_root: str,
    sensor: str = "LIDAR_TOP",
) -> Tuple[str, str]:
    """Write ``<sample_token>_<sensor>_xyzi.npy`` + ``_metadata.csv``.

    ``frame_metadata`` must carry ``frame_id, sample_token, lidar_token,
    timestamp`` (plus ``source_file`` when known). Returns (npy_path, csv_path).
    """
    for key in ("frame_id", "sample_token", "timestamp"):
        if key not in frame_metadata:
            raise ValueError(f"frame_metadata missing key: {key!r}")
    os.makedirs(processed_root, exist_ok=True)
    token = str(frame_metadata["sample_token"])
    npy_path = os.path.join(processed_root, f"{token}_{sensor}_xyzi.npy")
    csv_path = os.path.join(processed_root, f"{token}_metadata.csv")
    np.save(npy_path, np.asarray(points, dtype=np.float64))
    row = {
        "frame_id": str(frame_metadata["frame_id"]),
        "sample_token": token,
        "lidar_token": str(frame_metadata.get("lidar_token", "")),
        "timestamp": float(frame_metadata["timestamp"]),
        "source_file": str(frame_metadata.get("source_file", "")),
        **{k: int(v) for k, v in counts.items()},
    }
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    return npy_path, csv_path
