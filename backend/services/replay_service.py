"""Replay service (Steps 6-7). Thin wrapper over validated replay data.

Discovers frames from ``results/final/final_test_manifest.csv`` (the frozen
7-frame evaluation set) joined with ``data/processed/*_metadata.csv`` rows.
Loads bundled processed ``.npy`` frames (N x 4 float64). No second
incompatible loader: raw ``.bin`` reads still go through
``src/lidar_loader.py`` when needed.
"""

from __future__ import annotations

import csv
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger("paradox.backend.replay")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST = PROJECT_ROOT / "results" / "final" / "final_test_manifest.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SAMPLE_DATA_DIR = PROJECT_ROOT / "simulation_handoff" / "sample_data"


class FrameNotFoundError(KeyError):
    pass


class DataLoadError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _frame_index() -> List[Dict[str, Any]]:
    """Build the replay frame index. ``cache_hit`` is logged by callers."""
    if not MANIFEST.is_file():
        raise DataLoadError(f"Replay manifest missing: {MANIFEST}")
    frames: List[Dict[str, Any]] = []
    with open(MANIFEST, newline="") as fh:
        for row in csv.DictReader(fh):
            fid = row["frame_id"]
            meta = PROCESSED_DIR / f"{fid}_metadata.csv"
            npy = PROCESSED_DIR / f"{fid}_LIDAR_TOP_xyzi.npy"
            point_count: int | None = None
            source = f"data/processed/{fid}_LIDAR_TOP_xyzi.npy"
            if meta.is_file():
                try:
                    import pandas as pd

                    m = pd.read_csv(meta).iloc[0].to_dict()
                    point_count = int(m.get("n_processed", m.get("n_raw", 0)) or 0) or None
                    ts = m.get("timestamp", None)
                    timestamp = float(ts) if ts is not None else float(row["timestamp"])
                except Exception:
                    timestamp = float(row["timestamp"])
            else:
                timestamp = float(row["timestamp"])
                alt_npy = SAMPLE_DATA_DIR / f"{fid}_LIDAR_TOP_xyzi.npy"
                if alt_npy.is_file():
                    npy = alt_npy
                    source = f"simulation_handoff/sample_data/{fid}_LIDAR_TOP_xyzi.npy"
            frames.append(
                {
                    "frame_id": fid,
                    "scene_id": row.get("scene_id"),
                    "timestamp": timestamp,
                    "source": source,
                    "npy_path": str(npy),
                    "point_count": point_count,
                    "test_status": row.get("test_status"),
                }
            )
    return frames


def list_frames() -> List[Dict[str, Any]]:
    logger.debug("frame_index cache_hit=%s", _frame_index.cache_info().hits > 0)
    return [dict(f) for f in _frame_index()]


def get_frame_entry(frame_id: str) -> Dict[str, Any]:
    for entry in _frame_index():
        if entry["frame_id"] == str(frame_id):
            return dict(entry)
    raise FrameNotFoundError(f"Frame not found: {frame_id}")


def load_frame_points(frame_id: str) -> tuple[np.ndarray, Dict[str, Any]]:
    """Load one frame's processed points + metadata. Preserves id/timestamp."""
    entry = get_frame_entry(frame_id)
    path = Path(entry["npy_path"])
    if not path.is_file():
        raise DataLoadError(f"Replay data file missing for frame {frame_id}: {path}")
    try:
        points = np.load(path).astype(np.float64)
    except Exception as exc:
        raise DataLoadError(f"Cannot read replay frame {frame_id}: {exc}") from exc
    if points.ndim != 2 or points.shape[1] != 4:
        raise DataLoadError(f"Replay frame {frame_id} has bad shape {points.shape}, expected N x 4")
    if len(points) == 0:
        raise DataLoadError(f"Replay frame {frame_id} is empty")
    meta = {
        "frame_id": entry["frame_id"],
        "scene_id": entry.get("scene_id"),
        "timestamp": entry.get("timestamp"),
        "source_path": entry["npy_path"],
        "replay_reference": entry["source"],
    }
    logger.info("frame=%s stage=loading status=success points=%d", frame_id, len(points))
    return points, meta
