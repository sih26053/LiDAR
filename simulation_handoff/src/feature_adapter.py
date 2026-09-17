"""Feature adapter: PerceptionResult -> List[RegionFeatures].

Deterministic, replaceable interface adapter (NOT a perception model).
Strategy mirrors Stage-1 (4 m XY grid bins, centroid range, std(z) terrain
proxy) so synthetic Stage-1 data flows unchanged. A future real-data adapter
only needs to produce the same RegionFeatures list.

Also hosts ``run_first_handoff`` (10 September Together task): the reusable,
notebook-independent first-handoff pipeline
``processed N×4 -> LiDARFrame -> regions -> RegionFeatures -> Importance v1
-> Resolution v1 -> row dicts``. It consumes the production engines
unchanged (default configuration) and applies the task-mandated conservative
fallbacks (unknown / 0.0 / 0.0 / 0.0 / 0.5) wherever real perception output
is unavailable. No semantic classes, motion, or confidence are invented.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .data_types import (
    DEFAULT_UNKNOWN_CONFIDENCE,
    DEFAULT_UNKNOWN_LABEL,
    DEFAULT_UNKNOWN_UNCERTAINTY,
    DYNAMIC_PROXY,
    SEMANTIC_IMPORTANCE,
    TERRAIN_NORM_M,
    LiDARFrame,
    PerceptionResult,
    RegionFeatures,
)
from .interface_validator import (
    validate_lidar_frame,
    validate_perception_result,
    validate_region_features,
)

STAGE1_BIN_M = 4.0

# ---------------------------------------------------------------------------
# First-handoff constants (10 September Together task, Sections 17-19).
#
# Conservative fallbacks used ONLY because the first handoff has no trained
# perception model. They differ deliberately from the perception-pipeline
# proxies in data_types.py (DYNAMIC_PROXY unknown -> 0.20, uncertainty
# 1 - confidence): here dynamic/semantic/confidence are forced to zero and
# uncertainty to the documented 0.5 default, so nothing resembles measured
# perception. Labeled TEMPORARY FALLBACK, never measured data.
# ---------------------------------------------------------------------------
FIRST_HANDOFF_CELL_M = 2.0
FIRST_HANDOFF_SEMANTIC_LABEL = DEFAULT_UNKNOWN_LABEL  # "unknown"
FIRST_HANDOFF_SEMANTIC_IMPORTANCE = 0.0
FIRST_HANDOFF_DYNAMIC_RELEVANCE = 0.0
FIRST_HANDOFF_CONFIDENCE = 0.0
FIRST_HANDOFF_UNCERTAINTY = 0.5
FIRST_HANDOFF_MIN_HEIGHT_VARIATION = 1e-6


def build_perception_from_labeled_points(
    frame_id: str,
    points: np.ndarray,
    point_labels: List[str] | np.ndarray,
    point_confidence: List[float] | np.ndarray | None = None,
) -> PerceptionResult:
    """Build a PerceptionResult from synthetic labeled points.

    Geometric fields are derived deterministically:
    distance=hypot(x,y) (m), elevation=z (m), roughness=zeros (region step
    computes real roughness), point_density=ones (region step overwrites).
    """
    pts = np.asarray(points, dtype=np.float64)
    labels = np.asarray(list(point_labels), dtype=str)
    n = pts.shape[0]
    if point_confidence is None:
        conf = np.full(n, 0.9, dtype=np.float64)
    else:
        conf = np.asarray(list(point_confidence), dtype=np.float64)
    distance = np.hypot(pts[:, 0], pts[:, 1]).astype(np.float64)
    elevation = pts[:, 2].astype(np.float64)
    roughness = np.zeros(n, dtype=np.float64)
    density = np.ones(n, dtype=np.float64)
    pr = PerceptionResult(
        frame_id=frame_id,
        points=pts,
        semantic_labels=labels,
        confidence=conf,
        distance=distance,
        elevation=elevation,
        roughness=roughness,
        point_density=density,
    )
    validate_perception_result(pr)
    return pr


def _aggregate_semantic(labels: np.ndarray, conf: np.ndarray):
    """Majority vote over valid labels; mean confidence of winners.

    No fake probabilities. Empty/blank -> ("unknown", 0.0).
    """
    clean = [str(s).strip() for s in labels.tolist()]
    valid = [s for s in clean if s]
    if not valid:
        return DEFAULT_UNKNOWN_LABEL, DEFAULT_UNKNOWN_CONFIDENCE
    winner = Counter(valid).most_common(1)[0][0]
    mask = np.array([s == winner for s in clean])
    if mask.sum() == 0:
        return winner, DEFAULT_UNKNOWN_CONFIDENCE
    return winner, float(np.mean(conf[mask]))


def adapt_perception_to_regions(
    pr: PerceptionResult,
    bin_m: float = STAGE1_BIN_M,
    min_points: int = 5,
    semantic_importance_map: Optional[Dict[str, float]] = None,
    dynamic_proxy_map: Optional[Dict[str, float]] = None,
    default_uncertainty: float = DEFAULT_UNKNOWN_UNCERTAINTY,
) -> List[RegionFeatures]:
    """Convert point-level PerceptionResult to region-level RegionFeatures."""
    validate_perception_result(pr)
    if bin_m <= 0:
        raise ValueError(f"bin_m must be > 0. Received: {bin_m!r}")
    sem_map = dict(semantic_importance_map or SEMANTIC_IMPORTANCE)
    dyn_map = dict(dynamic_proxy_map or DYNAMIC_PROXY)

    pts = pr.points
    ix = np.floor(pts[:, 0] / bin_m).astype(np.int64)
    iy = np.floor(pts[:, 1] / bin_m).astype(np.int64)
    keys = np.column_stack([ix, iy])
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)

    regions: List[RegionFeatures] = []
    for rid in range(len(uniq)):
        idx = np.where(inv == rid)[0]
        if len(idx) < min_points:
            continue
        cxyz = pts[idx][:, :3]
        cx, cy = float(cxyz[:, 0].mean()), float(cxyz[:, 1].mean())
        # Missing semantic prediction -> "unknown" + confidence 0.0 (no invention).
        label, conf = _aggregate_semantic(pr.semantic_labels[idx], pr.confidence[idx])
        if label not in sem_map:
            label = DEFAULT_UNKNOWN_LABEL
            conf = DEFAULT_UNKNOWN_CONFIDENCE
        sem_imp = float(sem_map.get(label, 0.0))
        # Missing dynamic info -> documented semantic proxy (NOT velocity).
        dyn = float(dyn_map.get(label, dyn_map.get("unknown", 0.2)))
        # Missing uncertainty -> documented proxy: 1 - confidence.
        # Explicitly a prototype proxy, not sensor uncertainty.
        unc = float(max(0.0, min(1.0, 1.0 - conf)))
        if label == DEFAULT_UNKNOWN_LABEL and conf == 0.0:
            unc = float(default_uncertainty)
        elev = float(cxyz[:, 2].mean())
        rough = float(min(1.0, float(cxyz[:, 2].std()) / TERRAIN_NORM_M))
        region = RegionFeatures(
            region_id=int(rid),
            x=cx,
            y=cy,
            distance=float(np.hypot(cx, cy)),
            elevation=elev,
            roughness=rough,
            point_density=float(len(idx)),
            semantic_label=str(label),
            semantic_importance=sem_imp,
            confidence=float(conf),
            dynamic_relevance=dyn,
            uncertainty=unc,
            point_count=int(len(idx)),
        )
        validate_region_features(region)
        regions.append(region)
    return regions


# ---------------------------------------------------------------------------
# First-handoff pipeline (10 September Together task, Section 30).
# ---------------------------------------------------------------------------

def _require_frame_metadata(frame_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Check the handoff metadata dict carries identity + timestamp."""
    if not isinstance(frame_metadata, dict):
        raise ValueError(
            "frame_metadata must be a dict with at least "
            f"'frame_id', 'sample_token', 'timestamp'. Received: {type(frame_metadata).__name__}"
        )
    missing = [k for k in ("frame_id", "sample_token", "timestamp") if k not in frame_metadata]
    if missing:
        raise ValueError(f"frame_metadata missing keys: {missing}")
    try:
        float(frame_metadata["timestamp"])
    except (TypeError, ValueError):
        raise ValueError(
            f"frame_metadata['timestamp'] must be numeric. "
            f"Received: {frame_metadata['timestamp']!r}"
        )
    if not str(frame_metadata["frame_id"]) or not str(frame_metadata["sample_token"]):
        raise ValueError("frame_metadata['frame_id'/'sample_token'] must be non-empty strings.")
    return frame_metadata


def run_first_handoff(
    points: np.ndarray,
    frame_metadata: dict,
    cell_size: float = FIRST_HANDOFF_CELL_M,
) -> Tuple[LiDARFrame, List[RegionFeatures], List[Dict[str, float]]]:
    """Convert one processed N×4 LiDAR frame into engine outputs.

    Reusable, notebook-independent integration path::

        processed points (N×4 [x, y, z, intensity])
            ↓ basic geometric features (3D range + XY grid regions)
            ↓ terrain proxy (max-normalized height variation, temporary)
            ↓ RegionFeatures (task-mandated conservative fallbacks)
            ↓ interface validation (existing validator, never weakened)
            ↓ Importance Engine v1 (default config, unchanged)
            ↓ Resolution Engine v1 (default config, unchanged)
            ↓ row dicts (caller builds the DataFrame)

    Provenance per field (see docs/first_handoff_integration.md):
    REAL MEASURED: x, y, z, intensity, point counts, distances, elevation.
    GEOMETRIC DERIVED: region centroids, height variation, terrain proxy.
    TEMPORARY FALLBACK: semantic "unknown"/0.0, dynamic 0.0, confidence 0.0,
    uncertainty 0.5 (NOT measured perception; never presented as such).

    Returns ``(lidar_frame, region_features, handoff_rows)`` where each row
    carries region geometry + fallbacks + ``importance`` (final_importance)
    + ``resolution_m``. Deterministic. Raises ValueError on invalid input.
    """
    from .importance_engine import ImportanceEngine
    from .resolution_engine import ResolutionEngine

    _require_frame_metadata(frame_metadata)
    if not isinstance(cell_size, (int, float)) or not np.isfinite(float(cell_size)) or float(cell_size) <= 0:
        raise ValueError(f"cell_size must be finite and > 0 (metres). Received: {cell_size!r}")
    cell = float(cell_size)

    pts = np.asarray(points, dtype=np.float64)
    lidar_frame = LiDARFrame(
        frame_id=str(frame_metadata["frame_id"]),
        timestamp=float(frame_metadata["timestamp"]),
        points=pts,
    )
    validate_lidar_frame(lidar_frame)  # N×4, finite, N > 0 (raises otherwise)

    # Basic geometric features: 3D Euclidean range per point (diagnostic only;
    # region distance below uses the planar centroid range per the contract).
    _point_distance = np.linalg.norm(pts[:, :3], axis=1)

    # Spatial regions: simple XY grid at cell_size (engineering integration
    # grid, NOT the final adaptive mapping representation).
    grid_x = np.floor(pts[:, 0] / cell).astype(np.int64)
    grid_y = np.floor(pts[:, 1] / cell).astype(np.int64)
    unique_regions = np.unique(np.column_stack([grid_x, grid_y]), axis=0)

    records: List[Dict[str, float]] = []
    for region_id, (gx, gy) in enumerate(unique_regions):
        mask = (grid_x == int(gx)) & (grid_y == int(gy))
        region_points = pts[mask]
        if len(region_points) == 0:
            continue
        xyz_region = region_points[:, :3]
        x_mean = float(xyz_region[:, 0].mean())
        y_mean = float(xyz_region[:, 1].mean())
        z_values = xyz_region[:, 2]
        records.append({
            "region_id": int(region_id),
            "x": x_mean,
            "y": y_mean,
            "distance": float(np.sqrt(x_mean ** 2 + y_mean ** 2)),
            "elevation": float(z_values.mean()),
            "height_variation": float(z_values.std()),
            "point_density": float(len(region_points) / (cell * cell)),
            "point_count": int(len(region_points)),
        })
    if not records:
        raise ValueError("No spatial regions generated from the input points.")

    # Terrain-complexity proxy: max-normalized height variation, clipped to
    # [0, 1]. Temporary geometric proxy for integration testing — NOT a
    # trained terrain classifier, NOT semantic terrain understanding.
    max_height_variation = max(
        float(max(r["height_variation"] for r in records)),
        FIRST_HANDOFF_MIN_HEIGHT_VARIATION,
    )
    for r in records:
        r["terrain_complexity"] = float(
            max(0.0, min(1.0, r["height_variation"] / max_height_variation))
        )

    importance_engine = ImportanceEngine()  # default repo config, unchanged
    resolution_engine = ResolutionEngine()  # default repo config, unchanged

    region_features: List[RegionFeatures] = []
    handoff_rows: List[Dict[str, float]] = []
    for r in records:
        feature = RegionFeatures(
            region_id=int(r["region_id"]),
            x=float(r["x"]),
            y=float(r["y"]),
            distance=float(r["distance"]),
            elevation=float(r["elevation"]),
            roughness=float(r["terrain_complexity"]),
            point_density=float(r["point_density"]),
            semantic_label=FIRST_HANDOFF_SEMANTIC_LABEL,
            semantic_importance=FIRST_HANDOFF_SEMANTIC_IMPORTANCE,
            confidence=FIRST_HANDOFF_CONFIDENCE,
            dynamic_relevance=FIRST_HANDOFF_DYNAMIC_RELEVANCE,
            uncertainty=FIRST_HANDOFF_UNCERTAINTY,
            point_count=int(r["point_count"]),
        )
        validate_region_features(feature)  # mandatory checkpoint; raises on violation
        region_features.append(feature)

        importance = float(importance_engine.calculate(feature).final_importance)
        resolution_m = float(resolution_engine.assign_resolution(importance))
        handoff_rows.append({
            "region_id": feature.region_id,
            "x": feature.x,
            "y": feature.y,
            "distance": feature.distance,
            "elevation": feature.elevation,
            "terrain_complexity": feature.roughness,
            "semantic_label": feature.semantic_label,
            "semantic_importance": feature.semantic_importance,
            "dynamic_relevance": feature.dynamic_relevance,
            "uncertainty": feature.uncertainty,
            "confidence": feature.confidence,
            "point_count": feature.point_count,
            "importance": importance,
            "resolution_m": resolution_m,
        })
    return lidar_frame, region_features, handoff_rows
