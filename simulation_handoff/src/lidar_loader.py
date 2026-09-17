"""nuScenes LiDAR loading (10 September pipeline, Rajashree side).

Reads a raw nuScenes LIDAR_TOP ``.bin`` (x, y, z, intensity[, ring])
via the devkit and returns the project contract array: finite N x 4
``[x, y, z, intensity]``. No filtering here -- filtering lives in
``preprocessing.py``. No semantic claims.
"""
from __future__ import annotations

import os
from typing import Tuple
import numpy as np


def load_lidar_bin(bin_path: str) -> np.ndarray:
    """Load one raw nuScenes LiDAR .bin file -> (M, 4) float64 [x,y,z,intensity].

    nuScenes bins store float32 rows of (x, y, z, intensity, ring_index);
    only the first four columns are kept (contract: N x 4, no fifth field).
    Raises FileNotFoundError / ValueError on missing or malformed input.
    """
    if not os.path.isfile(bin_path):
        raise FileNotFoundError(f"LiDAR bin not found: {bin_path}")
    raw = np.fromfile(bin_path, dtype=np.float32)
    if raw.size == 0:
        raise ValueError(f"LiDAR bin is empty: {bin_path}")
    if raw.size % 5 != 0:
        raise ValueError(
            f"LiDAR bin size {raw.size} is not a multiple of 5: {bin_path}"
        )
    cloud = raw.reshape(-1, 5)[:, :4].astype(np.float64)
    return cloud


def load_sample_lidar(
    nusc, sample_token: str, sensor: str = "LIDAR_TOP"
) -> Tuple[np.ndarray, dict]:
    """Load the ``sensor`` point cloud for one nuScenes sample.

    Returns ``(points_Mx4, info)`` where info carries ``sample_token``,
    ``lidar_token`` (sample_data token), ``timestamp`` (microseconds),
    and ``source_file`` (relative filename inside the dataroot).
    """
    sample = nusc.get("sample", sample_token)
    try:
        sd_token = sample["data"][sensor]
    except KeyError:
        raise ValueError(f"Sample {sample_token} has no channel {sensor!r}.")
    sd = nusc.get("sample_data", sd_token)
    bin_path = os.path.join(nusc.dataroot, sd["filename"])
    points = load_lidar_bin(bin_path)
    info = {
        "sample_token": sample["token"],
        "lidar_token": sd["token"],
        "timestamp": float(sd["timestamp"]),
        "source_file": sd["filename"],
    }
    return points, info
