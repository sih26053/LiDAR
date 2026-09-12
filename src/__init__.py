"""Interface integration package (SIH/DRDO adaptive 2.5D mapping)."""
from .data_types import LiDARFrame, PerceptionResult, RegionFeatures
from .interface_validator import InterfaceError, validate_lidar_frame, validate_perception_result, validate_region_features
from .feature_adapter import adapt_perception_to_regions, build_perception_from_labeled_points
from .stage1_engines import AdaptiveMapper2_5D, ImportanceEngine, ImportanceResult, MapCell, MapResult, ResolutionEngine

__all__ = [
    "LiDARFrame", "PerceptionResult", "RegionFeatures",
    "InterfaceError", "validate_lidar_frame", "validate_perception_result",
    "validate_region_features", "adapt_perception_to_regions",
    "build_perception_from_labeled_points", "ImportanceEngine",
    "ResolutionEngine", "AdaptiveMapper2_5D", "ImportanceResult",
    "MapCell", "MapResult",
]
