"""Reusable Stage-1 core (importable reuse, NOT a redesign).

Faithful copy of the logic in 01_Model_Development.ipynb + stage1_stage2_combined.py:
I_base = wd*D + ws*S + wt*T + wm*M + wu*U, I_safe = min(1, I_base + lam*U),
resolution policy 0.75/0.50/0.25 -> 0.05/0.10/0.20/0.50 m, per-region XY
binning mapper. Canonical Stage-1 LAMBDA = 0.5.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from .data_types import (
    LAMBDA_UNCERTAINTY,
    MAX_RANGE_M,
    RESOLUTION_LEVELS,
    RESOLUTION_THRESHOLDS,
    WEIGHTS,
)


@dataclass
class ImportanceResult:
    region_id: int
    distance_score: float
    semantic_score: float
    terrain_score: float
    dynamic_score: float
    uncertainty_score: float
    base_importance: float
    safe_importance: float
    selected_resolution_m: float = -1.0
    resolution_level: str = ""


@dataclass
class MapCell:
    x: float
    y: float
    elevation: float
    elevation_std: float
    occupancy: int
    semantic_class: str
    confidence: float
    importance: float
    resolution: float
    region_id: int
    point_count: int


@dataclass
class MapResult:
    cells: List[MapCell]
    importance: List[ImportanceResult]
    elapsed_s: float
    n_points: int


class ImportanceEngine:
    """Existing Stage-1 engine. Consumes objects with distance_m etc. (see compat)."""

    def __init__(
        self,
        weights: Dict[str, float] | None = None,
        max_range_m: float = MAX_RANGE_M,
        lambda_uncertainty: float = LAMBDA_UNCERTAINTY,
    ):
        weights = dict(weights or WEIGHTS)
        if max_range_m <= 0:
            raise ValueError("max_range_m must be > 0")
        if any(v < 0 for v in weights.values()):
            raise ValueError("weights must be non-negative")
        if abs(sum(weights.values()) - 1.0) > 1e-9:
            raise ValueError("weights must sum to 1")
        self.w, self.max_range_m, self.lam = weights, max_range_m, lambda_uncertainty

    @staticmethod
    def _clip01(v: float) -> float:
        return float(min(1.0, max(0.0, v)))

    def distance_score(self, d: float) -> float:
        if not np.isfinite(d) or d < 0:
            raise ValueError(f"invalid distance: {d!r}")
        return self._clip01(1.0 - d / self.max_range_m)

    def score_region(self, r) -> ImportanceResult:
        # Duck-typed: accepts legacy SyntheticRegion OR RegionFeatures.to_legacy_region().
        d_m = getattr(r, "distance_m", getattr(r, "distance", None))
        terr = getattr(r, "terrain_complexity", getattr(r, "roughness", None))
        sem_c = getattr(r, "semantic_class", getattr(r, "semantic_label", "unknown"))
        d = self.distance_score(float(d_m))
        s = self._clip01(float(r.semantic_importance))
        t = self._clip01(float(terr))
        m = self._clip01(float(r.dynamic_relevance))
        u = self._clip01(float(r.uncertainty))
        base = (
            self.w["distance"] * d
            + self.w["semantic"] * s
            + self.w["terrain"] * t
            + self.w["dynamic"] * m
            + self.w["uncertainty"] * u
        )
        return ImportanceResult(
            int(r.region_id), d, s, t, m, u,
            self._clip01(base), float(min(1.0, base + self.lam * u)),
        )


class ResolutionEngine:
    """Existing Stage-1 policy: boundaries inclusive of the finer bin."""

    def __init__(
        self,
        t_fine: float = RESOLUTION_THRESHOLDS[0],
        t_med_fine: float = RESOLUTION_THRESHOLDS[1],
        t_med_coarse: float = RESOLUTION_THRESHOLDS[2],
        levels: Dict[str, float] | None = None,
    ):
        if not (t_fine > t_med_fine > t_med_coarse > 0):
            raise ValueError("thresholds must satisfy fine > med_fine > med_coarse > 0")
        self.t_fine, self.t_med_fine, self.t_med_coarse = t_fine, t_med_fine, t_med_coarse
        self.levels = dict(levels or RESOLUTION_LEVELS)

    def select(self, importance: float) -> Tuple[float, str]:
        if not np.isfinite(importance) or not (0.0 <= importance <= 1.0):
            raise ValueError(f"importance must be in [0,1]. Received: {importance!r}")
        if importance >= self.t_fine:
            return self.levels["fine"], "fine (5 cm)"
        if importance >= self.t_med_fine:
            return self.levels["medium_fine"], "medium_fine (10 cm)"
        if importance >= self.t_med_coarse:
            return self.levels["medium_coarse"], "medium_coarse (20 cm)"
        return self.levels["coarse"], "coarse (50 cm)"


class AdaptiveMapper2_5D:
    """Existing Stage-1 mapper: per-region XY binning at region resolution."""

    def __init__(self, imp: ImportanceEngine, res: ResolutionEngine):
        self.imp, self.res = imp, res

    def map_regions_legacy(self, legacy_regions) -> MapResult:
        t0 = time.perf_counter()
        cells: List[MapCell] = []
        scored: List[ImportanceResult] = []
        n = 0
        for r in legacy_regions:
            s = self.imp.score_region(r)
            res_m, lvl = self.res.select(s.safe_importance)
            s.selected_resolution_m, s.resolution_level = res_m, lvl
            pts = np.asarray(r.points, dtype=np.float64)
            ix = np.floor(pts[:, 0] / res_m).astype(np.int64)
            iy = np.floor(pts[:, 1] / res_m).astype(np.int64)
            ukeys, inv = np.unique(np.column_stack([ix, iy]), axis=0, return_inverse=True)
            for k, (cx, cy) in enumerate(ukeys):
                z = pts[inv == k, 2]
                cells.append(
                    MapCell(
                        float((cx + 0.5) * res_m), float((cy + 0.5) * res_m),
                        float(z.mean()), float(z.std() if len(z) > 1 else 0.0), 1,
                        str(getattr(r, "semantic_class", getattr(r, "semantic_label", "unknown"))),
                        float(r.confidence), float(s.safe_importance),
                        float(res_m), int(r.region_id), int((inv == k).sum()),
                    )
                )
            scored.append(s)
            n += len(pts)
        return MapResult(cells, scored, time.perf_counter() - t0, n)
