"""Stable external OUTPUT contracts (Steps 0.2 / 0.3)."""

from __future__ import annotations

from typing import Dict, List, Literal
from pydantic import BaseModel, Field

SemanticSource = Literal[
    "model_prediction",
    "lidarseg_annotation",
    "lidarseg",
    "annotation",
    "object_annotation",
    "fallback",
    "unknown",
    "model",
]


class MapCell(BaseModel):
    x: float
    y: float
    elevation: float
    occupancy: float
    resolution: float
    importance: float
    semantic_class: str
    semantic_source: str
    confidence: float | None = None
    point_count: int | None = None
    region_id: int | None = None


class ImportanceSummary(BaseModel):
    mean: float | None = None
    min: float | None = None
    max: float | None = None
    count: int = 0


class ResolutionSummary(BaseModel):
    fine_cells: int = 0
    medium_cells: int = 0
    coarse_cells: int = 0
    res_20cm_cells: int = 0
    res_50cm_cells: int = 0
    average_resolution: float | None = None
    distribution: Dict[str, int] = Field(default_factory=dict)


class SemanticSummary(BaseModel):
    mode: str = "annotation+fallback (no trained model)"
    source_counts: Dict[str, int] = Field(default_factory=dict)
    note: str = (
        "Annotation-derived labels are evaluation references, "
        "never model predictions."
    )


class TimingInfo(BaseModel):
    preprocessing_latency_ms: float | None = None
    perception_latency_ms: float | None = None
    feature_extraction_latency_ms: float | None = None
    importance_latency_ms: float | None = None
    resolution_latency_ms: float | None = None
    mapping_latency_ms: float | None = None
    total_latency_ms: float | None = None
    serialization_latency_ms: float | None = None
    wall_clock_ms: float | None = None
    fps: float | None = None


class PipelineResult(BaseModel):
    frame_id: str
    scene_id: str | None = None
    timestamp: float | None = None
    status: str = "success"
    input_point_count: int | None = None
    processed_point_count: int | None = None
    map_cells: List[MapCell] = Field(default_factory=list)
    map_cell_count: int = 0
    importance: ImportanceSummary = Field(default_factory=ImportanceSummary)
    resolution: ResolutionSummary = Field(default_factory=ResolutionSummary)
    semantic: SemanticSummary = Field(default_factory=SemanticSummary)
    timing: TimingInfo = Field(default_factory=TimingInfo)


class FrameMetadata(BaseModel):
    frame_id: str
    scene_id: str | None = None
    timestamp: float | None = None
    source: str | None = None
    point_count: int | None = None


class FrameListResponse(BaseModel):
    frames: List[FrameMetadata]
    count: int


class ReplayLoadResponse(BaseModel):
    frame_id: str
    scene_id: str | None = None
    timestamp: float | None = None
    point_count: int | None = None
    load_status: str = "loaded"


class ReplayRunRequest(BaseModel):
    frame_id: str = Field(..., min_length=1)
    save_output: bool = False
    include_visualization: bool = False
    max_map_cells: int | None = Field(default=None, ge=1, le=5000)


class ReplayRunResponse(BaseModel):
    result: PipelineResult
    cache_hit: bool = False


class DemoMetrics(BaseModel):
    frame_id: str
    input_point_count: int | None = None
    processed_point_count: int | None = None
    map_cell_count: int | None = None
    fine_cells: int | None = None
    medium_cells: int | None = None
    coarse_cells: int | None = None
    average_resolution: float | None = None
    latency_ms: float | None = None
    fps: float | None = None


class DemoStatus(BaseModel):
    backend: str = "running"
    configuration_loaded: bool = False
    replay_available: bool = False
    available_frames: int = 0
    last_frame_id: str | None = None
    last_status: str | None = None
