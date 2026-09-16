"""Common data contract: LiDAR <-> perception <-> Importance Engine.

Stable bridge between the DATA/PERCEPTION side and the existing Stage-1
core (Importance Engine -> Resolution Engine -> Adaptive 2.5D Mapper).

Contract expects LiDAR points as N x 4 [x, y, z, intensity]. Do NOT assume
a fifth field.

Provenance: field names/values reuse Stage-1 conventions from
01_Model_Development.ipynb and stage1_stage2_combined.py
(SyntheticRegion, SEMANTIC_IMPORTANCE, W, LAMBDA=0.5). No engine redesign.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np

# ---------------------------------------------------------------------------
# Shared constants (reused Stage-1 values, not new tuning)
# ---------------------------------------------------------------------------

POINT_DIM = 4  # contract: N x 4 [x, y, z, intensity]

MAX_RANGE_M = 100.0
TERRAIN_NORM_M = 0.5  # roughness = min(1, std(z) / 0.5), matches Stage-1

WEIGHTS: Dict[str, float] = {
    "distance": 0.30,
    "semantic": 0.30,
    "terrain": 0.15,
    "dynamic": 0.15,
    "uncertainty": 0.10,
}
LAMBDA_UNCERTAINTY = 0.5  # canonical Stage-1 value (01 notebook + combined file)

RESOLUTION_LEVELS: Dict[str, float] = {
    "fine": 0.05,
    "medium_fine": 0.10,
    "medium_coarse": 0.20,
    "coarse": 0.50,
}
RESOLUTION_THRESHOLDS = (0.75, 0.50, 0.25)  # fine, medium_fine, medium_coarse

# Union of Stage-1 vocabularies (01 notebook + combined file) + interface "unknown".
SEMANTIC_IMPORTANCE: Dict[str, float] = {
    "pedestrian": 1.0,
    "bicycle": 1.0,
    "motorcycle": 1.0,
    "vehicle": 0.90,
    "traffic_cone": 0.80,
    "barrier": 0.80,
    "unknown_obstacle": 0.80,
    "building": 0.50,
    "rough_terrain": 0.45,
    "vegetation": 0.30,
    "road": 0.10,
    "unknown": 0.0,  # missing-data default; never invent a class
}

# Documented prototype proxy (NOT measured velocity). Matches Stage-1 DYNAMIC_PROXY.
DYNAMIC_PROXY: Dict[str, float] = {
    "pedestrian": 0.95,
    "bicycle": 0.95,
    "motorcycle": 0.90,
    "vehicle": 0.80,
    "traffic_cone": 0.40,
    "barrier": 0.30,
    "unknown_obstacle": 0.40,
    "building": 0.05,
    "rough_terrain": 0.05,
    "vegetation": 0.05,
    "road": 0.0,
    "unknown": 0.20,  # conservative documented default for missing/unknown
}

VALID_SEMANTIC_LABELS = frozenset(SEMANTIC_IMPORTANCE.keys())

DEFAULT_UNKNOWN_LABEL = "unknown"
DEFAULT_UNKNOWN_CONFIDENCE = 0.0
DEFAULT_UNKNOWN_UNCERTAINTY = 0.5  # documented proxy, NOT sensor uncertainty


# ---------------------------------------------------------------------------
# Data structures (exact field names required by the task spec)
# ---------------------------------------------------------------------------

@dataclass
class LiDARFrame:
    """Raw LiDAR input. points.shape == (N, 4), columns [x, y, z, intensity].

    x/y/z in metres. intensity is the sensor intensity value (no unit
    assumption, no fifth field).
    """

    frame_id: str
    timestamp: float
    points: np.ndarray


@dataclass
class PerceptionResult:
    """Point-level perception/geometric output for one frame.

    Point-level (length N each): semantic_labels, confidence, distance,
    elevation, roughness, point_density. Region aggregation happens later
    in feature_adapter.PerceptionResult -> List[RegionFeatures].
    """

    frame_id: str
    points: np.ndarray            # (N, 4)
    semantic_labels: np.ndarray   # (N,) str
    confidence: np.ndarray        # (N,) [0,1]; NaN = not applicable (non-ML
                                  # source: lidarseg/annotation/fallback, see
                                  # docs/semantic_perception_integration.md)
    distance: np.ndarray          # (N,) metres, >= 0
    elevation: np.ndarray         # (N,) metres (= z)
    roughness: np.ndarray         # (N,) [0,1] normalized terrain variation
    point_density: np.ndarray     # (N,) documented numeric unit (pts/region broadcast)


@dataclass
class RegionFeatures:
    """Region-level features. Main structure consumed by the Importance Engine.

    Already normalized [0,1]: semantic_importance, confidence, roughness,
    dynamic_relevance, uncertainty. Physical/raw: distance (m), elevation (m),
    point_density (points/region), x/y (m centroid).
    """

    region_id: int
    x: float
    y: float
    distance: float
    elevation: float
    roughness: float
    point_density: float
    semantic_label: str
    semantic_importance: float
    confidence: float
    dynamic_relevance: float
    uncertainty: float
    point_count: int

    def to_legacy_region(self, points: np.ndarray):
        """Compat view matching the existing Stage-1 engine signature.

        Existing engines (01_Model_Development.ipynb SyntheticRegion and
        stage1_stage2_combined.py RegionFeatures) expect attributes:
        region_id, semantic_class, points (Nx4), distance_m,
        semantic_importance, terrain_complexity, dynamic_relevance,
        uncertainty, confidence.
        """
        from types import SimpleNamespace

        return SimpleNamespace(
            region_id=int(self.region_id),
            semantic_class=str(self.semantic_label),
            points=np.asarray(points, dtype=np.float64),
            distance_m=float(self.distance),
            semantic_importance=float(self.semantic_importance),
            terrain_complexity=float(self.roughness),
            dynamic_relevance=float(self.dynamic_relevance),
            uncertainty=float(self.uncertainty),
            confidence=float(self.confidence),
        )
