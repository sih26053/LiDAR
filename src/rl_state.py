"""Stage 6 — RL State Generation (methodology flow box 6).

Converts the local adaptive 2.5D map into a compact state vector for the
DQN agent. All quantities are computed from backend map cells; nothing
is invented.

State vector (13 dims): 8 ego-centric sector ranges (normalized by
``radius_m``) + obstacle density + moving share + static share +
terrain share + mean importance + mean uncertainty.

Conventions (documented assumptions, not measurements):
- ego heading = +X sensor axis (replay frames carry no heading signal);
- occupied = occupancy >= 0.5;
- a sector range = distance to the nearest occupied cell in that sector,
  or ``radius_m`` when the sector is clear.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.semantic_model import MANDATORY_CATEGORY

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_configs() -> Dict[str, Any]:
    """Backend-owned RL/safety configs (frontend must never modify)."""
    cfg: Dict[str, Any] = {}
    for name in ("rl_state_config.json", "safety_config.json"):
        try:
            cfg.update(json.loads((PROJECT_ROOT / "config" / name).read_text()))
        except (OSError, ValueError):
            pass
    return cfg


_CFG = _load_configs()
N_SECTORS = int(_CFG.get("n_sectors", 8))
DEFAULT_RADIUS_M = float(_CFG.get("radius_m", 30.0))
SAFETY_RADIUS_M = float(_CFG.get("safety_radius_m", 5.0))
SAFETY_HALF_ANGLE_DEG = float(_CFG.get("safety_half_angle_deg", 30.0))
OCCUPIED_THRESHOLD = float(_CFG.get("occupied_threshold", 0.5))


def _cells_to_arrays(cells: List[Dict[str, Any]]):
    xs = np.array([float(c["x"]) for c in cells], dtype=np.float64)
    ys = np.array([float(c["y"]) for c in cells], dtype=np.float64)
    occ = np.array([float(c.get("occupancy", 0.0)) for c in cells], dtype=np.float64)
    imp = np.array([float(c.get("importance", 0.0)) for c in cells], dtype=np.float64)
    unc = np.array(
        [1.0 - float(c["confidence"]) if c.get("confidence") is not None
         and np.isfinite(float(c["confidence"])) else 0.5 for c in cells],
        dtype=np.float64)
    cats = np.array(
        [MANDATORY_CATEGORY.get(str(c.get("semantic_class", "unknown")), "unknown")
         for c in cells], dtype=object)
    return xs, ys, occ, imp, unc, cats


def build_state(cells: List[Dict[str, Any]],
                radius_m: float = DEFAULT_RADIUS_M) -> Dict[str, Any]:
    """Map cells -> RL state dict (genuine computation, timed by caller)."""
    if not cells:
        raise ValueError("build_state received no map cells.")
    xs, ys, occ, imp, unc, cats = _cells_to_arrays(cells)
    rng = np.hypot(xs, ys)
    inside = rng <= float(radius_m)
    occ_mask = (occ >= OCCUPIED_THRESHOLD) & inside

    angles = (np.degrees(np.arctan2(ys, xs)) + 360.0) % 360.0
    sector = np.floor(angles / (360.0 / N_SECTORS)).astype(int)
    sector_ranges = []
    for s in range(N_SECTORS):
        m = occ_mask & (sector == s)
        sector_ranges.append(float(rng[m].min()) if np.any(m) else float(radius_m))

    n_in = int(inside.sum())
    obstacle_density = float(occ_mask.sum() / n_in) if n_in else 0.0
    moving_share = float((((cats == "moving_object")) & inside).sum() / n_in) if n_in else 0.0
    static_share = float((((cats == "static_obstacle")) & inside).sum() / n_in) if n_in else 0.0
    terrain_share = float((((cats == "terrain")) & inside).sum() / n_in) if n_in else 0.0

    # Stage 8 safety check (genuine): occupied cell inside the safety
    # radius within +/- half-angle of the +X heading assumption.
    fwd = (np.abs(((angles + 180.0) % 360.0) - 180.0) <= SAFETY_HALF_ANGLE_DEG)
    danger = occ >= OCCUPIED_THRESHOLD
    near = rng <= SAFETY_RADIUS_M
    blocked = bool(np.any(danger & near & fwd))
    nearest = float(rng[danger & fwd].min()) if np.any(danger & fwd) else None

    state_vector = np.array(
        [v / float(radius_m) for v in sector_ranges]
        + [obstacle_density, moving_share, static_share,
           float(imp[inside].mean()) if n_in else 0.0,
           float(unc[inside].mean()) if n_in else 0.5],
        dtype=np.float64)

    return {
        "state_dim": int(len(state_vector)),
        "state_vector": [round(float(v), 4) for v in state_vector],
        "sector_ranges_m": [round(v, 2) for v in sector_ranges],
        "n_sectors": N_SECTORS,
        "radius_m": float(radius_m),
        "cells_in_radius": n_in,
        "cells_total": int(len(cells)),
        "obstacle_density": round(obstacle_density, 4),
        "moving_share": round(moving_share, 4),
        "static_share": round(static_share, 4),
        "terrain_share": round(terrain_share, 4),
        "safety": {
            "emergency_stop": blocked,
            "nearest_forward_obstacle_m": round(nearest, 2) if nearest is not None else None,
            "safety_radius_m": SAFETY_RADIUS_M,
            "rule": "occupied cell (occ>=0.5) within 5 m and +/-30 deg of +X heading assumption",
        },
        "conventions": ("ego heading assumed +X (no heading signal in replay); "
                        "occupied = occupancy >= 0.5"),
    }
