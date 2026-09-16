"""Adaptive 2.5D Mapper (12 September 2026 task, Anik workstream).

ADDITIVE module. Does not modify:
- src/data_types.py (9 September contract)
- src/importance_engine.py (v1, unchanged)
- src/resolution_engine.py (v1, unchanged)
- src/feature_adapter.py run_first_handoff (10 September, unchanged)
- src/semantic_mapping.py / src/semantic_adapter.py (11 September, unchanged)

Pipeline stage owned here::

    RegionFeatures
        -> Importance Engine  -> importance [0,1]
        -> Resolution Engine  -> resolution_m in {0.05, 0.10, 0.20, 0.50}
        -> AdaptiveMapCell    (one region -> one cell for this prototype)

Occupancy rule (documented prototype simplification):
    occupancy = 1.0 for every region containing valid LiDAR evidence.
    Current prototype occupancy is an evidence-presence indicator; a
    probabilistic occupancy model is a later extension. It is NOT described
    as probabilistic occupancy anywhere in outputs.

Resolution policy is NEVER retuned here: the mapper consumes the injected
ResolutionEngine with its shipped default configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd

from .interface_validator import validate_region_features

REQUIRED_MAP_FIELDS = (
    "x", "y", "elevation", "occupancy", "semantic_class",
    "confidence", "importance", "resolution",
)

VALID_RESOLUTIONS = (0.05, 0.10, 0.20, 0.50)

PROTOTYPE_OCCUPANCY = 1.0


@dataclass
class AdaptiveMapCell:
    """One adaptive 2.5D map cell (first prototype: one region -> one cell)."""

    x: float
    y: float
    elevation: float
    occupancy: float
    semantic_class: str
    confidence: float | None
    importance: float
    resolution: float
    point_count: int
    region_id: int
    semantic_source: str


def _final_importance(result: Any) -> float:
    """Extract final importance from ImportanceResult object or dict."""
    if isinstance(result, dict):
        if "final_importance" in result:
            return float(result["final_importance"])
        if "safe_importance" in result:
            return float(result["safe_importance"])
        raise ValueError(f"Importance dict missing final_importance: {result!r}")
    for attr in ("final_importance", "safe_importance"):
        if hasattr(result, attr):
            return float(getattr(result, attr))
    raise ValueError(f"Importance result has no final importance: {result!r}")


def _resolve_region_semantic_source(region: Any, detail_lookup: Dict[int, str]) -> str:
    """Return audit-tracked semantic source when available, else fallback."""
    if int(getattr(region, "region_id", -1)) in detail_lookup:
        return str(detail_lookup[int(region.region_id)])
    return str(getattr(region, "semantic_source", "fallback") or "fallback")


def build_adaptive_map(
    region_features: Sequence[Any],
    importance_engine: Any,
    resolution_engine: Any,
    region_details: Sequence[Dict[str, Any]] | None = None,
    occupancy: float = PROTOTYPE_OCCUPANCY,
) -> Tuple[List[AdaptiveMapCell], pd.DataFrame]:
    """Map RegionFeatures -> AdaptiveMapCells -> adaptive-map DataFrame.

    For each region: validate contract -> ImportanceEngine.calculate ->
    ResolutionEngine.assign_resolution -> AdaptiveMapCell (occupancy =
    evidence-presence indicator 1.0). Returns (cells, DataFrame) with one
    row per region in input order. Raises ValueError on empty input.
    """
    regions = list(region_features)
    if not regions:
        raise ValueError("region_features must contain at least 1 region.")
    detail_lookup: Dict[int, str] = {}
    for row in (region_details or []):
        try:
            detail_lookup[int(row["region_id"])] = str(row.get("semantic_source", "fallback"))
        except (KeyError, TypeError, ValueError):
            continue

    cells: List[AdaptiveMapCell] = []
    for region in regions:
        validate_region_features(region)
        importance = float(_final_importance(importance_engine.calculate(region)))
        if not np.isfinite(importance) or not (0.0 <= importance <= 1.0):
            raise ValueError(
                f"Importance Engine returned out-of-range value {importance!r} "
                f"(region_id={getattr(region, 'region_id', '?')})."
            )
        resolution = float(resolution_engine.assign_resolution(importance))
        cells.append(
            AdaptiveMapCell(
                x=float(region.x),
                y=float(region.y),
                elevation=float(region.elevation),
                occupancy=float(occupancy),
                semantic_class=str(region.semantic_label),
                confidence=float(region.confidence),
                importance=importance,
                resolution=resolution,
                point_count=int(region.point_count),
                region_id=int(region.region_id),
                semantic_source=_resolve_region_semantic_source(region, detail_lookup),
            )
        )
    df = pd.DataFrame([cell.__dict__ for cell in cells])
    return cells, df


def validate_adaptive_map_df(adaptive_map_df: pd.DataFrame) -> bool:
    """Validate the adaptive-map DataFrame contract. Returns True or raises."""
    for field in REQUIRED_MAP_FIELDS:
        if field not in adaptive_map_df.columns:
            raise ValueError(f"Adaptive map missing required field: {field!r}")
    if len(adaptive_map_df) == 0:
        raise ValueError("Adaptive map is empty.")
    imp = pd.to_numeric(adaptive_map_df["importance"], errors="coerce").to_numpy()
    if not np.all(np.isfinite(imp)) or np.any((imp < 0.0) | (imp > 1.0)):
        raise ValueError("Adaptive map 'importance' must be finite within [0,1].")
    if not adaptive_map_df["resolution"].isin(list(VALID_RESOLUTIONS)).all():
        raise ValueError(
            "Adaptive map 'resolution' must be one of "
            f"{list(VALID_RESOLUTIONS)}. Received: "
            f"{sorted(adaptive_map_df['resolution'].unique().tolist())!r}"
        )
    occ = pd.to_numeric(adaptive_map_df["occupancy"], errors="coerce").to_numpy()
    if not np.all(np.isfinite(occ)) or np.any(occ < 0):
        raise ValueError("Adaptive map 'occupancy' must be finite and >= 0.")
    return True


def map_statistics(adaptive_map_df: pd.DataFrame) -> Dict[str, Any]:
    """Descriptive prototype statistics (NOT final performance benchmarks)."""
    validate_adaptive_map_df(adaptive_map_df)
    resolution_counts = adaptive_map_df["resolution"].value_counts().sort_index()
    return {
        "num_cells": int(len(adaptive_map_df)),
        "occupied_cells": int((adaptive_map_df["occupancy"] > 0).sum()),
        "mean_importance": float(adaptive_map_df["importance"].mean()),
        "min_importance": float(adaptive_map_df["importance"].min()),
        "max_importance": float(adaptive_map_df["importance"].max()),
        "resolution_counts": {float(k): int(v) for k, v in resolution_counts.items()},
        "resolution_percentage": {
            float(k): float(v / len(adaptive_map_df) * 100.0)
            for k, v in resolution_counts.items()
        },
    }
