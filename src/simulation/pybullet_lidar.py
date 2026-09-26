"""Phase 6-7 — Ray-based simulated LiDAR + LiDARFrame adapter.

Every range comes from actual `rayTestBatch` queries against the live
PyBullet world (no fabricated points). Hit object ids are recorded ONLY
in the raw scan log for scenario analysis -- they NEVER enter perception
(the trained MLP sees only coordinates + neutral intensity).

Intensity: raycast has no intensity channel. Points carry the configured
neutral value (default 0.5) tagged `sim-default`; the adapter docstring
and every scan record state this explicitly (RULE: documented, not a
measurement).

Coordinates: hits are converted to the VEHICLE frame (ego at origin,
heading +X) to match the pipeline's sensor-frame assumption used with
nuScenes replay data.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List

import numpy as np

from src.data_types import LiDARFrame


class LidarUnavailable(RuntimeError):
    """Raised when a scan is requested without a live environment."""


class PyBulletLidar:
    """Configurable raycast LiDAR (fov / rays / rings / range / mount)."""

    def __init__(self, n_rays: int = 180, fov_deg: float = 360.0,
                 rings_deg: List[float] | None = None,
                 max_range_m: float = 50.0, height_m: float = 1.8,
                 neutral_intensity: float = 0.5):
        self.n_rays = int(n_rays)
        self.fov_deg = float(fov_deg)
        self.rings_deg = list(rings_deg if rings_deg is not None else [-6.0, 0.0, 6.0])
        self.max_range_m = float(max_range_m)
        self.height_m = float(height_m)
        self.neutral_intensity = float(neutral_intensity)

    def scan(self, env) -> Dict[str, Any]:
        """Fire all rays; return raw scan (genuine rayTest results)."""
        if env is None or getattr(env, "_p", None) is None:
            raise LidarUnavailable("no live PyBullet environment attached")
        st = env.get_vehicle_state()
        yaw = math.radians(st["yaw_deg"])
        ox, oy, oz = st["x"], st["y"], st["z"] + self.height_m - 0.7
        origins, endpoints, meta = [], [], []
        for ring in self.rings_deg:
            pitch = math.radians(ring)
            for i in range(self.n_rays):
                az = math.radians(self.fov_deg * i / self.n_rays)
                dx = math.cos(pitch) * math.cos(az) * self.max_range_m
                dy = math.cos(pitch) * math.sin(az) * self.max_range_m
                dz = math.sin(pitch) * self.max_range_m
                origins.append([ox, oy, oz])
                endpoints.append([ox + dx, oy + dy, oz + dz])
                meta.append((az, pitch))
        hits = env.raycast(origins, endpoints)
        return {"origins": origins, "hits": hits, "meta": meta,
                "vehicle": {"x": st["x"], "y": st["y"], "yaw_deg": st["yaw_deg"]},
                "n_rays": len(origins),
                "intensity_source": "sim-default (neutral, documented)"}

    def to_points(self, scan: Dict[str, Any]) -> np.ndarray:
        """Raw scan -> N x 4 vehicle-frame [x, y, z, intensity]."""
        veh = scan["vehicle"]
        yaw = math.radians(veh["yaw_deg"])
        c, s = math.cos(-yaw), math.sin(-yaw)
        pts = []
        for o, h in zip(scan["origins"], scan["hits"]):
            hit_id, hit_pos, hit_frac = h[0], h[3], h[2]
            if hit_id < 0:
                continue  # no return: dropped, never invented
            dx, dy, dz = hit_pos[0] - o[0], hit_pos[1] - o[1], hit_pos[2] - o[2]
            # world -> vehicle frame (ego at origin, heading +X)
            lx = dx * c - dy * s
            ly = dx * s + dy * c
            pts.append([lx, ly, dz, self.neutral_intensity])
        out = np.ascontiguousarray(np.array(pts, dtype=np.float64).reshape(-1, 4))
        if out.shape[0] == 0 or not np.all(np.isfinite(out)):
            raise ValueError("scan converted to empty/non-finite points")
        return out


def to_lidar_frame(points: np.ndarray, frame_id: str,
                   timestamp: float | None = None,
                   scene_id: str = "pybullet") -> LiDARFrame:
    """N x 4 array -> existing LiDARFrame (no new data structure).

    Verifies frame_id / timestamp / shape / dtype / finiteness.
    Intensity column must already carry the documented neutral value;
    this adapter does not invent or rescale it.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 4:
        raise ValueError(f"points must be N x 4, got {pts.shape!r}")
    if pts.shape[0] == 0 or not np.all(np.isfinite(pts)):
        raise ValueError("points must be non-empty and all-finite")
    return LiDARFrame(frame_id=str(frame_id),
                      timestamp=float(timestamp if timestamp is not None else time.time()),
                      points=np.ascontiguousarray(pts),
                      scene_id=scene_id, source="pybullet_sim")


def hit_object_ids(scan: Dict[str, Any]) -> List[int]:
    """Hit body ids for scenario analysis ONLY (never into perception)."""
    return sorted({int(h[0]) for h in scan["hits"] if int(h[0]) >= 0})
