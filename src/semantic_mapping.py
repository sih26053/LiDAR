"""Project semantic taxonomy + nuScenes mapping (11 September 2026 task).

11-September Semantic Perception + Geometric Feature Integration.

This module is ADDITIVE. It does not modify:
- src/data_types.py (9 September contract, Stage-1 vocab)
- src/importance_engine.py (v1, unchanged)
- src/resolution_engine.py (v1, unchanged)
- src/feature_adapter.py run_first_handoff (10 September, unchanged)

Project taxonomy (exactly these six broad classes):
    vehicle, pedestrian_vru, static_manmade, vegetation,
    road_driveable, unknown

Source vocabulary (mandatory provenance tracking):
    "lidarseg" | "annotation" | "model" | "fallback"

Confidence rule (never fabricate ML confidence):
- model      -> actual model confidence in [0,1]
- lidarseg / annotation / fallback -> NOT APPLICABLE, represented as
  NaN in point-level PerceptionResult.confidence (see interface_validator).
  RegionFeatures.confidence keeps the numeric 0.0 interface placeholder
  (documented as "no ML confidence", never presented as a measurement)
  because the existing Importance Engine requires [0,1].

Importance values below are initial engineering parameters, NOT measured
importance, accuracy, or learned weights. Dynamic values are prior /
heuristic defaults, NOT measured velocity.
"""
from __future__ import annotations

from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Project taxonomy (spec Section 12)
# ---------------------------------------------------------------------------

PROJECT_CLASSES: Dict[str, int] = {
    "vehicle": 0,
    "pedestrian_vru": 1,
    "static_manmade": 2,
    "vegetation": 3,
    "road_driveable": 4,
    "unknown": 5,
}

PROJECT_CLASS_NAMES: Dict[int, str] = {v: k for k, v in PROJECT_CLASSES.items()}

VALID_PROJECT_LABELS = frozenset(PROJECT_CLASSES.keys())

VALID_SEMANTIC_SOURCES = frozenset({"lidarseg", "annotation", "model", "fallback"})

# ---------------------------------------------------------------------------
# Semantic importance (spec Section 24): initial engineering parameters.
# Deliberately separate from src/data_types.SEMANTIC_IMPORTANCE (Stage-1
# vocab: pedestrian/vehicle/road/...). The engine consumes the scalar, so
# either map is consumable without engine modification.
# ---------------------------------------------------------------------------

PROJECT_SEMANTIC_IMPORTANCE: Dict[str, float] = {
    "vehicle": 1.0,
    "pedestrian_vru": 1.0,
    "static_manmade": 0.7,
    "vegetation": 0.3,
    "road_driveable": 0.1,
    "unknown": 0.5,
}

# ---------------------------------------------------------------------------
# Dynamic relevance prior (spec Section 25): heuristic prior ONLY.
# vehicle != measured moving vehicle. Documented fallback when no temporal
# tracking / ego-motion-compensated velocity exists.
# ---------------------------------------------------------------------------

PROJECT_DYNAMIC_PRIOR: Dict[str, float] = {
    "vehicle": 0.80,
    "pedestrian_vru": 0.95,
    "static_manmade": 0.05,
    "vegetation": 0.05,
    "road_driveable": 0.0,
    "unknown": 0.20,
}

DEFAULT_UNCERTAINTY_FALLBACK = 0.5  # documented proxy, NOT calibrated uncertainty
DEFAULT_UNKNOWN_LABEL = "unknown"


def map_nuscenes_label_to_project(label_name: str) -> str:
    """Map a nuScenes category/label name to one of the six project classes.

    Only returns: vehicle | pedestrian_vru | static_manmade | vegetation |
    road_driveable | unknown. Unmapped / ambiguous input -> "unknown".
    Matching is prefix/substring based and documented below; no unsupported
    mapping is invented (e.g. sidewalk/terrain default to unknown unless
    explicitly a driveable surface).
    """
    name = str(label_name or "").strip().lower()
    if not name or name in ("unknown", "noise", "ignore"):
        return "unknown"

    # Pedestrian / vulnerable road users (human.* + animal + bicycle/motorcycle
    # riders map to VRU because the carrier is a VRU-scale dynamic object).
    if name.startswith("human.pedestrian") or name.startswith("human"):
        return "pedestrian_vru"
    if name == "animal":
        return "pedestrian_vru"
    if name in ("vehicle.bicycle", "vehicle.motorcycle"):
        return "pedestrian_vru"

    # Vehicles (excluding the two VRU-scale carriers above).
    if name.startswith("vehicle."):
        return "vehicle"

    # Vegetation.
    if "vegetation" in name:
        return "vegetation"

    # Road / driveable surface.
    if name in ("flat.driveable_surface", "driveable_surface", "road",
                "flat.road", "drivable_surface"):
        return "road_driveable"

    # Static / manmade structures.
    if name.startswith("static."):
        return "static_manmade"
    if name.startswith("movable_object."):
        # barriers / cones / debris are manmade scene furniture.
        return "static_manmade"
    if name.startswith("static_object."):
        return "static_manmade"
    if name in ("flat.sidewalk", "flat.other", "flat.sidewalk".lower()):
        return "static_manmade"

    return "unknown"


# nuScenes lidarseg numeric ids (v1.0) -> official label names.
# Reference: nuscenes lidarseg overview (32 labels, 0..31).
LIDARSEG_ID_TO_NAME: Dict[int, str] = {
    0: "noise",
    1: "animal",
    2: "human.pedestrian.adult",
    3: "human.pedestrian.child",
    4: "human.pedestrian.construction_worker",
    5: "human.pedestrian.personal_mobility",
    6: "human.pedestrian.police_officer",
    7: "human.pedestrian.stroller",
    8: "human.pedestrian.wheelchair",
    9: "movable_object.barrier",
    10: "movable_object.debris",
    11: "movable_object.pushable_pullable",
    12: "movable_object.trafficcone",
    13: "static_object.bicycle_rack",
    14: "vehicle.bicycle",
    15: "vehicle.bus.bendy",
    16: "vehicle.bus.rigid",
    17: "vehicle.car",
    18: "vehicle.construction",
    19: "vehicle.emergency.ambulance",
    20: "vehicle.emergency.police",
    21: "vehicle.motorcycle",
    22: "vehicle.trailer",
    23: "vehicle.truck",
    24: "flat.driveable_surface",
    25: "flat.other",
    26: "flat.sidewalk",
    27: "flat.terrain",
    28: "static.manmade",
    29: "static.other",
    30: "static.vegetation",
    31: "vehicle.ego",
}

# Precomputed id -> project class (derived via map_nuscenes_label_to_project,
# with two documented overrides: flat.terrain -> unknown (not confidently
# vegetation without a classifier), vehicle.ego -> unknown (ego vehicle mask,
# not a perception target)).
LIDARSEG_ID_TO_PROJECT: Dict[int, str] = {}
for _lid, _lname in LIDARSEG_ID_TO_NAME.items():
    LIDARSEG_ID_TO_PROJECT[_lid] = map_nuscenes_label_to_project(_lname)
LIDARSEG_ID_TO_PROJECT[27] = "unknown"  # flat.terrain: ambiguous ground
LIDARSEG_ID_TO_PROJECT[31] = "unknown"  # vehicle.ego: ego mask, not a target
LIDARSEG_ID_TO_PROJECT[0] = "unknown"  # noise


def map_lidarseg_id_to_project(lidarseg_id: int) -> Tuple[str, str]:
    """Map a numeric lidarseg id -> (original_label_name, project_label)."""
    try:
        lid = int(lidarseg_id)
    except (TypeError, ValueError):
        return "unknown", "unknown"
    original = LIDARSEG_ID_TO_NAME.get(lid, "unknown")
    project = LIDARSEG_ID_TO_PROJECT.get(lid, "unknown")
    return original, project
