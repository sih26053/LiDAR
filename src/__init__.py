"""Interface integration package (SIH/DRDO adaptive 2.5D mapping)."""
from .data_types import LiDARFrame, PerceptionResult, RegionFeatures
from .interface_validator import InterfaceError, validate_lidar_frame, validate_perception_result, validate_region_features
from .feature_adapter import adapt_perception_to_regions, build_perception_from_labeled_points, run_first_handoff
from .stage1_engines import AdaptiveMapper2_5D, ImportanceEngine, ImportanceResult, MapCell, MapResult, ResolutionEngine
from .importance_engine import (
    ImportanceEngine as ImportanceEngineV1,
    ImportanceResult as ImportanceResultV1,
    apply_uncertainty_modifier,
    calculate_base_importance,
    clip01,
    extract_features,
    normalize_distance,
    validate_inputs,
)
from .resolution_engine import ResolutionEngine as ResolutionEngineV1
from .resolution_engine import ResolutionResult, assign_resolution, validate_importance

__all__ = [
    "LiDARFrame", "PerceptionResult", "RegionFeatures",
    "InterfaceError", "validate_lidar_frame", "validate_perception_result",
    "validate_region_features", "adapt_perception_to_regions",
    "build_perception_from_labeled_points", "run_first_handoff", "ImportanceEngine",
    "ResolutionEngine", "AdaptiveMapper2_5D", "ImportanceResult",
    "MapCell", "MapResult", "ImportanceEngineV1", "ResolutionEngineV1",
    "ImportanceResultV1", "ResolutionResult", "assign_resolution",
    "clip01", "normalize_distance", "validate_inputs",
    "validate_importance", "calculate_base_importance",
    "apply_uncertainty_modifier", "extract_features",
]
