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
from .semantic_mapping import (
    PROJECT_CLASSES,
    PROJECT_CLASS_NAMES,
    PROJECT_DYNAMIC_PRIOR,
    PROJECT_SEMANTIC_IMPORTANCE,
    VALID_SEMANTIC_SOURCES,
    map_lidarseg_id_to_project,
    map_nuscenes_label_to_project,
)
from .semantic_adapter import (
    aggregate_to_regions,
    annotations_to_sensor_frame,
    assign_annotation_semantics,
    build_perception_from_semantics,
    compute_point_geometry,
    points_in_box,
)
from .lidar_loader import load_lidar_bin, load_sample_lidar
from .preprocessing import preprocess_points, save_processed_frame
from .mapper_2_5d import (
    AdaptiveMapCell,
    PROTOTYPE_OCCUPANCY,
    REQUIRED_MAP_FIELDS,
    VALID_RESOLUTIONS,
    build_adaptive_map,
    map_statistics,
    validate_adaptive_map_df,
)
from .visualization import (
    plot_combined_demo,
    plot_elevation_map,
    plot_importance_map,
    plot_lidar_xy,
    plot_resolution_map,
    plot_semantic_regions,
)

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
    "PROJECT_CLASSES", "PROJECT_CLASS_NAMES", "PROJECT_DYNAMIC_PRIOR",
    "PROJECT_SEMANTIC_IMPORTANCE", "VALID_SEMANTIC_SOURCES",
    "map_lidarseg_id_to_project", "map_nuscenes_label_to_project",
    "aggregate_to_regions", "annotations_to_sensor_frame",
    "assign_annotation_semantics",
    "build_perception_from_semantics", "compute_point_geometry",
    "points_in_box", "load_lidar_bin", "load_sample_lidar",
    "preprocess_points", "save_processed_frame",
    "AdaptiveMapCell", "PROTOTYPE_OCCUPANCY", "REQUIRED_MAP_FIELDS",
    "VALID_RESOLUTIONS", "build_adaptive_map", "map_statistics",
    "validate_adaptive_map_df", "plot_combined_demo", "plot_elevation_map",
    "plot_importance_map", "plot_lidar_xy", "plot_resolution_map",
    "plot_semantic_regions",
]
