from src.data_types import RegionFeatures

def validate_region_features(region: RegionFeatures) -> bool:
    """
    Validates boundary conditions, value constraints, and data types of a RegionFeatures instance.
    Raises ValueError or TypeError with explicit diagnostic messages on violations.
    """
    if not isinstance(region.region_id, int) or isinstance(region.region_id, bool):
        raise TypeError(f"region_id must be an integer, got {type(region.region_id).__name__} (value: {region.region_id})")

    if not isinstance(region.x, (float, int)) or not isinstance(region.y, (float, int)):
        raise TypeError(f"Coordinates (x, y) must be numeric floats. Got x={type(region.x).__name__}, y={type(region.y).__name__}")

    # Ensure canonical float type representation
    if not isinstance(region.x, float) or not isinstance(region.y, float):
        raise TypeError(f"Coordinates (x, y) must strictly be floats. Found x:{type(region.x)}, y:{type(region.y)}")

    if not isinstance(region.point_count, int) or isinstance(region.point_count, bool) or region.point_count < 0:
        raise ValueError(f"point_count must be a non-negative integer, got {region.point_count}")

    if region.distance < 0.0:
        raise ValueError(f"distance must be non-negative (>= 0.0), got {region.distance}")

    unit_interval_fields = {
        "semantic_importance": region.semantic_importance,
        "confidence": region.confidence,
        "dynamic_relevance": region.dynamic_relevance,
        "uncertainty": region.uncertainty,
        "terrain_complexity": region.terrain_complexity,
    }

    for name, val in unit_interval_fields.items():
        if not isinstance(val, (float, int)):
            raise TypeError(f"Field '{name}' must be a float in [0.0, 1.0], got {type(val).__name__}")
        if not (0.0 <= float(val) <= 1.0):
            raise ValueError(f"Field '{name}' must be bounded in [0.0, 1.0], got {val}")

    return True
