"""Phase 9 — 13-D state -> named structured Jev state (adapter only).

Verified index mapping of the frozen Stage-6 contract (src/rl_state.py,
config/rl_state_config.json); dtype float64, all [0,1]-normalized,
missing sectors saturate to 1.0. This module renames, never recomputes.

Exact mapping (index | feature          | meaning                                  | normalization      | range):
  0-7    | sector_{0..7}_range | ego-centric sector range to nearest occupied | value / radius_m   | [0,1]
         |                     | cell; sectors are 45-deg bins from +X heading  | (radius_m=30.0)    |
         |                     | (7,0 fwd; 1,2 left; 5,6 right; 3,4 rear);      |                    |
         |                     | clear sector saturates to 1.0                  |                    |
  8      | obstacle_density    | occupied cells / cells in radius               | already a share    | [0,1]
  9      | moving_share        | moving_object cells / cells in radius          | already a share    | [0,1]
  10     | static_share        | static_obstacle cells / cells in radius        | already a share    | [0,1]
  11     | mean_importance     | mean cell importance in radius                 | native [0,1]       | [0,1]
  12     | mean_uncertainty    | mean (1 - confidence) in radius                | native [0,1]       | [0,1]

Note: Stage 6 also computes terrain_share + sector_ranges_m + safety, but
only the 13 normalized values above enter the Jev state (plus the raw
sector_ranges_m list for reference). Vehicle speed / road state are NOT
in the contract and are reported unavailable, never estimated.
"""

from __future__ import annotations

from typing import Any, Dict, List

FIELD_NAMES = [
    "sector_0_range", "sector_1_range", "sector_2_range", "sector_3_range",
    "sector_4_range", "sector_5_range", "sector_6_range", "sector_7_range",
    "obstacle_density", "moving_share", "static_share",
    "mean_importance", "mean_uncertainty",
]
STATE_DIM = 13

# Fallback radius (m) for meter derivation when sector_ranges_m is absent.
# Must match config/rl_state_config.json radius_m (documented, not tuned).
FALLBACK_RADIUS_M = 30.0

DENSITY_SEMANTICS = (
    "obstacle_density is LiDAR-coverage evidence presence "
    "(fraction of in-radius map cells holding sensor evidence); "
    "1.0 is normal full-scan coverage and is NOT by itself equivalent "
    "to 'all surrounding directions are blocked'. Judge blockage from "
    "the directional clearances and nearest_obstacle_m."
)


def verify_vector(vec: List[float]) -> List[float]:
    """Assert dimension/order/dtype/bounds (raises, never coerces silently)."""
    import math
    v = [float(x) for x in list(vec)]
    if len(v) != STATE_DIM:
        raise ValueError(f"state must have dim {STATE_DIM}, got {len(v)}")
    for i, x in enumerate(v):
        if not math.isfinite(x) or not (0.0 <= x <= 1.0):
            raise ValueError(f"state[{i}]={x!r} outside [0,1] or non-finite")
    return v


def to_named_state(state_vector: List[float],
                   sector_ranges_m: List[float] | None = None) -> Dict[str, Any]:
    """13-D vector -> named INPUT STATE for Jev (RULE 4: no raw cloud)."""
    v = verify_vector(state_vector)
    named = dict(zip(FIELD_NAMES, v))
    named["sector_ranges_m"] = list(sector_ranges_m or [])
    named["state_dim"] = STATE_DIM
    return named


def decision_criteria(named: Dict[str, Any]) -> Dict[str, Any]:
    """Derive only quantities computable from verified fields (no invention).

    clearances: sectors 0 (front, +X .. +/-22.5deg approx via sector map),
    left/right aggregates; nearest obstacle from min sector range.
    Vehicle speed / road state are NOT in the 13-D contract and are
    reported as unavailable (never estimated).
    """
    secs = [named[f"sector_{i}_range"] for i in range(8)]
    # Sectors are 45 deg bins from +X: 7,0 = forward; 1,2 = left;
    # 5,6 = right; 3,4 = rear. Documented approximation.
    return {
        "forward_clearance": min(secs[7], secs[0]),
        "left_clearance": min(secs[1], secs[2]),
        "right_clearance": min(secs[5], secs[6]),
        "rear_clearance": min(secs[3], secs[4]),
        "nearest_obstacle_norm": min(secs),
        "obstacle_density": named["obstacle_density"],
        "moving_share": named["moving_share"],
        "mean_importance": named["mean_importance"],
        "mean_uncertainty": named["mean_uncertainty"],
        "vehicle_speed_ms": None,
        "road_state": None,
        "unavailable_note": "speed/road not in 13-D contract; Jev must decide without them",
    }


def enrich_for_jev(named: Dict[str, Any],
                   safety: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Named 13-D state -> enriched Jev INPUT STATE (adapter layer only).

    The 13 Stage-6 fields are preserved exactly (not added/removed/
    reordered). The enrichment adds named fields DERIVED from existing
    runtime/state data so the decision criteria can reference them:

    - forward_clearance_m: min(sector_ranges_m[7], [0]) in meters
    - left_clearance_m:    min(sector_ranges_m[1], [2]) in meters
    - right_clearance_m:   min(sector_ranges_m[5], [6]) in meters
    - nearest_obstacle_m:  min(sector_ranges_m) in meters
      (meter source: Stage-6 sector_ranges_m; when absent, falls back
      to normalized value x FALLBACK_RADIUS_M (30.0 m, documented))
    - emergency_flag:      bool(safety.emergency_stop), False if unknown
    - obstacle_density:    copied Stage-6 share (semantics unchanged)
    - obstacle_density_semantics: explicit evidence-presence note
      (DENSITY_SEMANTICS); Stage-6 semantics are NOT redefined
    - nearest_forward_obstacle_m: Stage-6 safety distance (may be None)

    No new sensor measurement is invented; nothing here alters
    Stage 1-6 algorithms or contracts.
    """
    enriched = dict(named)
    ranges_m = list(named.get("sector_ranges_m") or [])

    def meters(idx: int) -> float:
        if len(ranges_m) == 8:
            try:
                return round(float(ranges_m[idx]), 2)
            except (TypeError, ValueError):
                pass
        return round(float(named[f"sector_{idx}_range"]) * FALLBACK_RADIUS_M, 2)

    fwd = min(meters(7), meters(0))
    left = min(meters(1), meters(2))
    right = min(meters(5), meters(6))
    enriched["forward_clearance_m"] = fwd
    enriched["left_clearance_m"] = left
    enriched["right_clearance_m"] = right
    enriched["nearest_obstacle_m"] = min(meters(i) for i in range(8))
    enriched["emergency_flag"] = bool((safety or {}).get("emergency_stop", False))
    enriched["obstacle_density"] = float(named["obstacle_density"])
    enriched["obstacle_density_semantics"] = DENSITY_SEMANTICS
    enriched["nearest_forward_obstacle_m"] = (safety or {}).get(
        "nearest_forward_obstacle_m", None)
    return enriched
