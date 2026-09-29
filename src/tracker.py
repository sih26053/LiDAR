"""Multi-frame object tracker (greedy association, measured velocity).

Detections are spatial clusters of dynamic-class map cells (vehicle /
pedestrian_vru, whatever semantic source the map used) per frame.
Association across consecutive frames is greedy nearest-neighbor within a
class-matched gate; persistent integer track IDs are assigned; velocity is
measured from centroid displacement over real sample timestamps
(v = dx/dt, dt from nuScenes timestamps in seconds).

Association is REFUSED across scene boundaries or gaps larger than
MAX_DT_S (fresh tracks instead) — identity is never fabricated across
unrelated frames.

What this is NOT: not a learned tracker, not Kalman-filtered, not
ego-motion-compensated (positions are sensor-frame; ego motion leaks into
velocity when the ego vehicle moves — reported honestly in the output
note). Data association failures (ID switches in crowds) are possible and
are NOT hidden: every track reports hits/misses/gaps.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

DYNAMIC_CLASSES = ("vehicle", "pedestrian_vru")
CLUSTER_EPS_M = 3.0
GATE_BASE_M = 2.0
GATE_SPEED_M_S = 30.0
MAX_MISSES = 2
# Association is only attempted between consecutive frames of the SAME scene
# no more than this far apart. Larger gaps (scene change, replay jump) start
# fresh tracks instead of fabricating cross-scene identity.
MAX_DT_S = 3.0


def cluster_detections(cells: List[Dict[str, Any]],
                       eps: float = CLUSTER_EPS_M) -> List[Dict[str, Any]]:
    """Cluster dynamic-class cells into detections (one per object blob).

    BFS clustering on planar distance with ``eps`` linkage, per class.
    Returns detections sorted by (class, -cell_count).
    """
    dyn = [c for c in cells
           if str(c.get("semantic_class")) in DYNAMIC_CLASSES]
    if not dyn:
        return []
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None  # type: ignore
    dets: List[Dict[str, Any]] = []
    for cls in DYNAMIC_CLASSES:
        pts = np.array([[float(c["x"]), float(c["y"])] for c in dyn
                        if str(c.get("semantic_class")) == cls],
                       dtype=np.float64)
        if len(pts) == 0:
            continue
        if cKDTree is not None:
            tree = cKDTree(pts)
            pairs = tree.query_ball_point(pts, r=float(eps))
        else:
            d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
            pairs = [np.where(d2[i] <= float(eps) ** 2)[0].tolist()
                     for i in range(len(pts))]
        seen = np.zeros(len(pts), dtype=bool)
        for i in range(len(pts)):
            if seen[i]:
                continue
            stack, comp = [i], []
            seen[i] = True
            while stack:
                j = stack.pop()
                comp.append(j)
                for k in pairs[j]:
                    if not seen[k]:
                        seen[k] = True
                        stack.append(k)
            comp_pts = pts[comp]
            dets.append({
                "class": cls,
                "x": float(comp_pts[:, 0].mean()),
                "y": float(comp_pts[:, 1].mean()),
                "cells": int(len(comp)),
            })
    dets.sort(key=lambda d: (d["class"], -d["cells"]))
    return dets


def track_sequence(frames: List[Dict[str, Any]],
                   gate_base: float = GATE_BASE_M,
                   gate_speed: float = GATE_SPEED_M_S,
                   max_misses: int = MAX_MISSES) -> Dict[str, Any]:
    """Associate per-frame detections into tracks.

    ``frames``: [{"frame_id", "timestamp" (us), "scene_id", "detections": [...]}].
    Greedy global-nearest unmatched pairing within gate
    ``gate_base + gate_speed * dt``. Association is refused across scene
    boundaries or gaps larger than MAX_DT_S (fresh tracks instead).
    Returns tracks + association stats.
    """
    tracks: List[Dict[str, Any]] = []
    next_id = 1
    total_assoc = 0
    total_new = 0
    n_scene_breaks = 0
    for fi, fr in enumerate(frames):
        ts = fr.get("timestamp")
        dt_s = None
        link_ok = fi == 0
        if fi > 0 and ts is not None and frames[fi - 1].get("timestamp") is not None:
            dt_s = (float(ts) - float(frames[fi - 1]["timestamp"])) / 1e6
            same_scene = (fr.get("scene_id") is not None
                          and fr.get("scene_id") == frames[fi - 1].get("scene_id"))
            link_ok = bool(same_scene and 0 < dt_s <= MAX_DT_S)
            if not link_ok:
                n_scene_breaks += 1
                # Identity does not carry across scenes/gaps: retire open
                # tracks so the next frame starts fresh identities.
                for t in tracks:
                    t["closed"] = True
        dets = [dict(d) for d in fr.get("detections", [])]
        # Candidates: live tracks of matching class, only while the temporal
        # link to the previous frame is valid.
        scored: List[tuple] = []
        if link_ok:
            for di, d in enumerate(dets):
                for t in tracks:
                    if t["closed"]:
                        continue
                    if t["misses"] > max_misses:
                        continue
                    if t["class"] != d["class"]:
                        continue
                    last = t["states"][-1]
                    gate = gate_base + gate_speed * (dt_s if dt_s and dt_s > 0 else 0.5)
                    dist = math.hypot(d["x"] - last["x"], d["y"] - last["y"])
                    if dist <= gate:
                        scored.append((dist, di, t["id"]))
        scored.sort()
        used_dets, used_tracks = set(), set()
        for dist, di, tid in scored:
            if di in used_dets or tid in used_tracks:
                continue
            used_dets.add(di)
            used_tracks.add(tid)
            t = next(t for t in tracks if t["id"] == tid)
            last = t["states"][-1]
            vx = vy = None
            if link_ok and dt_s and dt_s > 0:
                vx = (dets[di]["x"] - last["x"]) / dt_s
                vy = (dets[di]["y"] - last["y"]) / dt_s
            t["states"].append({
                "frame_id": fr["frame_id"], "x": dets[di]["x"], "y": dets[di]["y"],
                "vx_m_s": vx, "vy_m_s": vy,
                "speed_m_s": math.hypot(vx, vy) if vx is not None else None,
                "cells": dets[di]["cells"],
            })
            t["misses"] = 0
            t["hits"] += 1
            total_assoc += 1
        for di, d in enumerate(dets):
            if di in used_dets:
                continue
            tracks.append({
                "id": next_id, "class": d["class"], "closed": False,
                "misses": 0, "hits": 1,
                "states": [{
                    "frame_id": fr["frame_id"], "x": d["x"], "y": d["y"],
                    "vx_m_s": None, "vy_m_s": None, "speed_m_s": None,
                    "cells": d["cells"],
                }],
            })
            next_id += 1
            total_new += 1
        for t in tracks:
            if t["id"] not in used_tracks and t["states"][-1]["frame_id"] != fr["frame_id"]:
                t["misses"] += 1
                if t["misses"] > max_misses:
                    t["closed"] = True
    for t in tracks:
        t["age_frames"] = len(t["states"])
    return {
        "tracks": tracks,
        "n_tracks": len(tracks),
        "n_associations": total_assoc,
        "n_new_tracks": total_new,
        "n_scene_breaks": n_scene_breaks,
        "note": ("Greedy nearest-neighbor association on dynamic-cell clusters; "
                 "velocity from sensor-frame centroid displacement over sample "
                 "timestamps (NOT ego-motion-compensated; ego motion leaks into "
                 "velocity). No learned motion model; ID switches possible. "
                 "Association refused across scenes/gaps (fresh tracks instead)."),
    }
