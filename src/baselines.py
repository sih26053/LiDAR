"""Benchmark baselines (13 September 2026 task, Anik workstream).

ADDITIVE module. Does not modify:

- ``src/data_types.py`` (9 September contract)
- ``src/importance_engine.py`` (v1, unchanged)
- ``src/resolution_engine.py`` (v1, unchanged)
- ``src/feature_adapter.py`` ``run_first_handoff`` (10 September, unchanged)
- ``src/semantic_mapping.py`` / ``src/semantic_adapter.py`` (11 Sept, unchanged)
- ``src/mapper_2_5d.py`` (12 September, unchanged)

Two genuine baselines sharing the SAME common inputs (same frame, same
preprocessing, same ROI, same evaluation area) as the proposed method.
Only the map-resolution strategy differs:

Baseline 1 -- uniform 5 cm
    Every spatial cell in the evaluation area has resolution 0.05 m.
    A true uniform grid over the common evaluation area (vectorized NumPy),
    NOT a relabelled adaptive region list.

Baseline 2 -- distance adaptive
    Resolution from planar distance ONLY::

        0-10 m   -> 0.05 m
        10-30 m  -> 0.10 m
        30-60 m  -> 0.20 m
        60-100 m -> 0.50 m

    Must never call ``ImportanceEngine`` and never use semantic class,
    terrain complexity, dynamic relevance, or uncertainty.

Method separation (visible in code and docs)::

    Proposed:          semantic+distance+terrain+dynamic+uncertainty
                       -> importance -> resolution
    Uniform 5 cm:      resolution = 0.05 m (everywhere)
    Distance adaptive: distance -> resolution
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Frozen benchmark policy (mirrors results/benchmark/benchmark_config.json)
# ---------------------------------------------------------------------------

#: Baseline 1: every cell has this resolution (metres).
UNIFORM_RESOLUTION = 0.05

#: Baseline 2: (exclusive upper distance bound [m], resolution [m]).
#: Configuration-driven; the defaults below are the frozen benchmark values.
DISTANCE_THRESHOLDS: Tuple[Tuple[float, float], ...] = (
    (10.0, 0.05),
    (30.0, 0.10),
    (60.0, 0.20),
    (float("inf"), 0.50),
)

#: Resolutions shared with the proposed method (data_types / resolution_config).
VALID_RESOLUTIONS = (0.05, 0.10, 0.20, 0.50)


# ---------------------------------------------------------------------------
# Baseline 2: distance-only resolution
# ---------------------------------------------------------------------------

def distance_based_resolution(
    distance: float,
    thresholds: Sequence[Tuple[float, float]] = DISTANCE_THRESHOLDS,
) -> float:
    """Map a planar distance [m] to a cell size [m] using distance ONLY.

    Raises ``ValueError`` on negative / non-finite distance. Thresholds are
    ``(exclusive_upper_bound_m, resolution_m)`` pairs evaluated in order.
    """
    try:
        d = float(distance)
    except (TypeError, ValueError):
        raise ValueError(
            f"distance must be a number (metres). Received: {distance!r}"
        )
    if not np.isfinite(d) or d < 0:
        raise ValueError(
            f"distance must be finite and >= 0 (metres). Received: {distance!r}"
        )
    for upper, resolution in thresholds:
        if d < float(upper):
            return float(resolution)
    raise ValueError(f"No distance bin matched distance={d!r}.")


def build_distance_adaptive_map(
    region_features: Sequence[Any],
    thresholds: Sequence[Tuple[float, float]] = DISTANCE_THRESHOLDS,
) -> pd.DataFrame:
    """Build the Baseline 2 map: one cell per region, resolution by distance.

    Uses ONLY ``region.distance`` (planar centroid range, metres). Never
    uses the importance machinery, semantic labels, terrain, dynamics, or
    uncertainty for the resolution decision -- the semantic columns below
    are carried through as provenance labels from the same source evidence,
    never as inputs to the resolution decision.
    """
    regions = list(region_features)
    if not regions:
        raise ValueError("region_features must contain at least 1 region.")
    rows: List[Dict[str, Any]] = []
    for region in regions:
        resolution = distance_based_resolution(float(region.distance),
                                               thresholds)
        rows.append({
            "x": float(region.x),
            "y": float(region.y),
            "elevation": float(region.elevation),
            "occupancy": 1.0,  # evidence-presence indicator (same convention)
            "semantic_class": str(region.semantic_label),
            "confidence": float(region.confidence),
            "resolution": resolution,
            "resolution_m": resolution,
            "distance": float(region.distance),
            "point_count": int(region.point_count),
            "region_id": int(region.region_id),
            "semantic_source": str(
                getattr(region, "semantic_source", "fallback") or "fallback"
            ),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Common evaluation area (shared verbatim by all three methods)
# ---------------------------------------------------------------------------

def evaluation_area_from_regions(
    region_features: Sequence[Any],
    pad_m: float = 1.0,
) -> Dict[str, float]:
    """Bounding box of region centroids, expanded by ``pad_m`` (half the
    2.0 m integration cell, so the area encloses the integration cells).

    The SAME dict is passed to the proposed, uniform, and distance methods;
    coverage equivalence is asserted downstream (``coverage_area`` must
    match within tolerance before any cell-count comparison).
    """
    regions = list(region_features)
    if not regions:
        raise ValueError("region_features must contain at least 1 region.")
    xs = np.array([float(r.x) for r in regions], dtype=np.float64)
    ys = np.array([float(r.y) for r in regions], dtype=np.float64)
    x_min, x_max = float(xs.min()) - pad_m, float(xs.max()) + pad_m
    y_min, y_max = float(ys.min()) - pad_m, float(ys.max()) + pad_m
    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "coverage_area": (x_max - x_min) * (y_max - y_min),
    }


# ---------------------------------------------------------------------------
# Baseline 1: true uniform 5 cm grid
# ---------------------------------------------------------------------------

@dataclass
class UniformMapResult:
    """Result of :func:`build_uniform_map`.

    ``exact_cells`` (``nx * ny``) is the authoritative count. Full per-cell
    arrays are materialized ONLY when ``return_arrays=True`` (the
    representative frame); aggregate runs keep the lightweight metrics so a
    ~8M-cell grid never becomes an enormous retained object.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    coverage_area: float
    resolution_m: float
    nx: int
    ny: int
    exact_cells: int
    theoretical_cells: float
    occupied_cells: int
    grid_x: np.ndarray | None = None  # (ny, nx) float32, optional
    grid_y: np.ndarray | None = None  # (ny, nx) float32, optional
    elevation: np.ndarray | None = None  # nearest-region evidence, optional
    sem_code: np.ndarray | None = None  # nearest-region class code, optional


def build_uniform_map(
    evaluation_area: Dict[str, float],
    region_features: Sequence[Any] | None = None,
    resolution_m: float = UNIFORM_RESOLUTION,
    return_arrays: bool = False,
) -> UniformMapResult:
    """Generate a TRUE uniform grid at ``resolution_m`` over the SAME
    evaluation area used by the other methods.

    Grid nodes ``arange(x_min, x_max + r/2, r)`` (inclusive); the exact
    count ``nx * ny`` is authoritative, ``A / r^2`` is reported alongside
    as the theoretical approximation. Where ``region_features`` are given,
    each grid node aggregates the SAME source LiDAR evidence (nearest
    region centroid via ``scipy.spatial.cKDTree``) for elevation/semantic
    columns, so the baseline is never starved of source information.

    Vectorized NumPy throughout; no Python per-cell loop.
    """
    x_min = float(evaluation_area["x_min"])
    x_max = float(evaluation_area["x_max"])
    y_min = float(evaluation_area["y_min"])
    y_max = float(evaluation_area["y_max"])
    r = float(resolution_m)
    if not np.isfinite(r) or r <= 0:
        raise ValueError(f"resolution_m must be finite and > 0. Got {r!r}")
    if not (x_max > x_min and y_max > y_min):
        raise ValueError(f"Invalid evaluation area: {evaluation_area!r}")

    xs = np.arange(x_min, x_max + r * 0.5, r, dtype=np.float64)
    ys = np.arange(y_min, y_max + r * 0.5, r, dtype=np.float64)
    nx, ny = int(len(xs)), int(len(ys))
    exact = int(nx * ny)
    coverage = (x_max - x_min) * (y_max - y_min)
    theoretical = float(coverage / (r ** 2))

    grid_x = grid_y = elev = sem = None
    if return_arrays:
        gx, gy = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
        grid_x, grid_y = gx, gy
        if region_features is not None:
            from scipy.spatial import cKDTree

            regions = list(region_features)
            centroids = np.array(
                [[float(x) for x in (_.x, _.y)] for _ in regions],
                dtype=np.float64,
            )
            tree = cKDTree(centroids)
            query = np.column_stack(
                [gx.ravel().astype(np.float64), gy.ravel().astype(np.float64)]
            )
            _, nearest = tree.query(query, k=1)
            elev = np.array(
                [float(regions[i].elevation) for i in nearest],
                dtype=np.float32,
            ).reshape(gx.shape)
            codes, _ = _semantic_codes(
                [str(_.semantic_label) for _ in regions]
            )
            sem = np.asarray(codes, dtype=np.int16)[np.asarray(nearest)].reshape(
                gx.shape
            )
    return UniformMapResult(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        coverage_area=coverage,
        resolution_m=r,
        nx=nx,
        ny=ny,
        exact_cells=exact,
        theoretical_cells=theoretical,
        occupied_cells=exact,  # dense grid: every cell covers evaluation area
        grid_x=grid_x,
        grid_y=grid_y,
        elevation=elev,
        sem_code=sem,
    )


#: Project taxonomy codes shared with the uniform-grid evidence aggregation.
PROJECT_CLASS_CODES: Dict[str, int] = {
    "vehicle": 0,
    "pedestrian_vru": 1,
    "static_manmade": 2,
    "vegetation": 3,
    "road_driveable": 4,
    "unknown": 5,
}
PROJECT_CODE_NAMES: Dict[int, str] = {
    v: k for k, v in PROJECT_CLASS_CODES.items()
}


def _semantic_codes(
    labels: Sequence[str],
) -> Tuple[List[int], Dict[str, int]]:
    """Map project-class labels to stable int codes (unknown for other)."""
    codes = [
        PROJECT_CLASS_CODES.get(str(s), PROJECT_CLASS_CODES["unknown"])
        for s in labels
    ]
    return codes, dict(PROJECT_CLASS_CODES)


# ---------------------------------------------------------------------------
# Comparative metric: resolution-density proxy (NOT measured compute cost)
# ---------------------------------------------------------------------------

def resolution_density_proxy(resolutions_m: Sequence[float]) -> float:
    """``RDP = sum_i 1 / r_i^2`` over map cells.

    A comparative spatial-resolution metric only. Never described as
    measured compute cost or runtime.
    """
    r = np.asarray(list(resolutions_m), dtype=np.float64)
    if r.size == 0:
        raise ValueError("resolutions_m must be non-empty.")
    if not np.all(np.isfinite(r)) or np.any(r <= 0):
        raise ValueError("All resolutions must be finite and > 0.")
    return float(np.sum(1.0 / (r ** 2)))
