"""Stage 3 — Scene Analysis (methodology flow box 3).

Computes the six scene factors per region, exactly the vocabulary of the
flow diagram:

  1. distance_from_lidar ......... planar range of the region centroid (m)
  2. point_density ................ points per m^2, frame-normalized to [0,1]
  3. object_density ............... moving-object share proxy in [0,1]
  4. object_importance ............ project semantic-importance lookup (alias)
  5. elevation_variation .......... terrain complexity 0..1 (alias)
  6. prediction_uncertainty ....... region uncertainty + estimator provenance

Provenance per factor (measured / annotation-derived / heuristic) is
carried alongside every value. Feed mapping into the Importance Engine
(which is UNCHANGED) is documented in ``IMPORTANCE_FEED``: closeness,
object importance, elevation variation, dynamic relevance and
uncertainty enter the weighted formula; point density and object
density are diagnostic outputs consumed by RL state generation.

``object_density`` proxy (documented, not a tracked count): the dominant
class vote ratio when the dominant label is a moving-object class
(vehicle / pedestrian_vru), else 0. Region member point labels are not
retained by ``aggregate_to_regions``, so a dominant-vote proxy is the
honest computable quantity at this stage.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from src.importance_engine import normalize_distance
from src.semantic_mapping import PROJECT_SEMANTIC_IMPORTANCE

MOVING_OBJECT_CLASSES = frozenset({"vehicle", "pedestrian_vru"})

#: Which scene factors enter the frozen importance formula, and how.
IMPORTANCE_FEED: Dict[str, str] = {
    "distance_from_lidar": "D = clip01(1 - distance / max_distance), weight 0.30 (measured)",
    "object_importance": "S = semantic-importance lookup, weight 0.30 (annotation- or model-derived)",
    "elevation_variation": "T = frame-normalized height std, weight 0.15 (measured)",
    "dynamic_relevance": "M = class prior heuristic, weight 0.15 (heuristic, kept outside this module)",
    "prediction_uncertainty": "U = 1 - model confidence, or 0.5 fallback, weight 0.10 + 0.15 boost",
    "point_density": "diagnostic only (RL state input, not an importance term)",
    "object_density": "diagnostic only (RL state input, not an importance term)",
}

FACTOR_PROVENANCE: Dict[str, str] = {
    "distance_from_lidar": "measured (LiDAR geometry)",
    "point_density": "measured (point counts)",
    "object_density": "derived proxy (dominant-vote share, documented)",
    "object_importance": "annotation- or model-derived lookup",
    "elevation_variation": "measured (height variation)",
    "prediction_uncertainty": "model-derived (1 - confidence) or documented 0.5 fallback",
}


def analyze_frame(
    detail_rows: Sequence[Dict[str, Any]],
    max_distance: float = 100.0,
) -> pd.DataFrame:
    """Region detail rows -> six-factor scene-analysis table (one row/region)."""
    rows = list(detail_rows)
    if not rows:
        raise ValueError("analyze_frame received no regions.")
    densities = np.array([float(r.get("point_density", 0.0)) for r in rows], dtype=np.float64)
    dmax = float(densities.max()) if float(densities.max()) > 0 else 1.0
    out: List[Dict[str, Any]] = []
    for r in rows:
        dist = float(r.get("distance", 0.0))
        label = str(r.get("semantic_label", "unknown"))
        ratio = float(r.get("dominant_class_ratio", 0.0))
        out.append({
            "region_id": int(r.get("region_id", -1)),
            "distance_from_lidar_m": dist,
            "distance_closeness": normalize_distance(dist, max_distance),
            "point_density_raw": float(r.get("point_density", 0.0)),
            "point_density": float(np.clip(densities[len(out)] / dmax, 0.0, 1.0)),
            "object_density": float(np.clip(ratio, 0.0, 1.0)) if label in MOVING_OBJECT_CLASSES else 0.0,
            "object_density_basis": ("dominant-vote share of " + label) if label in MOVING_OBJECT_CLASSES else "dominant label not a moving-object class",
            "object_importance": float(PROJECT_SEMANTIC_IMPORTANCE.get(label, 0.5)),
            "elevation_variation": float(r.get("terrain_complexity", 0.0)),
            "prediction_uncertainty": float(r.get("uncertainty", 0.5)),
            "uncertainty_source": str(r.get("uncertainty_source", "documented fallback 0.5 (no estimator)")),
            "semantic_label": label,
            "semantic_source": str(r.get("semantic_source", "unknown")),
        })
    return pd.DataFrame(out)


def scene_summary(scene_df: pd.DataFrame) -> Dict[str, Any]:
    """Frame-level roll-up of the six factors (measured aggregations only)."""
    return {
        "n_regions": int(len(scene_df)),
        "mean_distance_m": float(scene_df["distance_from_lidar_m"].mean()),
        "mean_point_density": float(scene_df["point_density"].mean()),
        "mean_object_density": float(scene_df["object_density"].mean()),
        "mean_object_importance": float(scene_df["object_importance"].mean()),
        "mean_elevation_variation": float(scene_df["elevation_variation"].mean()),
        "mean_prediction_uncertainty": float(scene_df["prediction_uncertainty"].mean()),
        "model_sourced_share": float((scene_df["semantic_source"] == "model").mean()),
        "factor_provenance": dict(FACTOR_PROVENANCE),
        "importance_feed": dict(IMPORTANCE_FEED),
    }
