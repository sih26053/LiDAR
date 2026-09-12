"""Feature adapter: PerceptionResult -> List[RegionFeatures].

Deterministic, replaceable interface adapter (NOT a perception model).
Strategy mirrors Stage-1 (4 m XY grid bins, centroid range, std(z) terrain
proxy) so synthetic Stage-1 data flows unchanged. A future real-data adapter
only needs to produce the same RegionFeatures list.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional
import numpy as np

from .data_types import (
    DEFAULT_UNKNOWN_CONFIDENCE,
    DEFAULT_UNKNOWN_LABEL,
    DEFAULT_UNKNOWN_UNCERTAINTY,
    DYNAMIC_PROXY,
    SEMANTIC_IMPORTANCE,
    TERRAIN_NORM_M,
    PerceptionResult,
    RegionFeatures,
)
from .interface_validator import validate_perception_result, validate_region_features

STAGE1_BIN_M = 4.0


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
