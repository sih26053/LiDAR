from dataclasses import dataclass
from typing import Optional

@dataclass
class RegionFeatures:
    """
    Standardized, dataset-agnostic intermediate representation for spatial regions.
    Decoupled from specific dataset formats (e.g., SemanticKITTI, nuScenes, Kaggle).
    """
    region_id: int
    x: float
    y: float
    distance: float
    terrain_complexity: float
    point_count: int
    semantic_label: Optional[str]
    semantic_importance: float
    confidence: float
    dynamic_relevance: float
    uncertainty: float
