"""Tests for the 12-September Adaptive 2.5D Mapper + visualization utilities.

Contract checks only: no engine retuning, no threshold changes.
"""
import numpy as np
import pandas as pd
import pytest

from src.data_types import RegionFeatures
from src.importance_engine import ImportanceEngine
from src.resolution_engine import ResolutionEngine
from src.mapper_2_5d import (
    AdaptiveMapCell,
    build_adaptive_map,
    map_statistics,
    validate_adaptive_map_df,
)
from src import visualization as viz


def _region(rid=0, **kw):
    base = dict(region_id=rid, x=1.0, y=2.0, distance=10.0, elevation=0.5,
                roughness=0.2, point_density=25.0, semantic_label="vehicle",
                semantic_importance=1.0, confidence=0.0, dynamic_relevance=0.8,
                uncertainty=0.5, point_count=100)
    base.update(kw)
    return RegionFeatures(**base)


def test_build_adaptive_map_one_region_one_cell():
    regions = [_region()]
    cells, df = build_adaptive_map(regions, ImportanceEngine(), ResolutionEngine())
    assert len(cells) == 1 and len(df) == 1
    assert isinstance(cells[0], AdaptiveMapCell)
    assert validate_adaptive_map_df(df) is True


def test_map_cell_carries_all_required_fields():
    _, df = build_adaptive_map([_region(), _region(1)],
                               ImportanceEngine(), ResolutionEngine(),
                               [{"region_id": 0, "semantic_source": "annotation"},
                                {"region_id": 1, "semantic_source": "fallback"}])
    for field in ("x", "y", "elevation", "occupancy", "semantic_class",
                  "confidence", "importance", "resolution"):
        assert field in df.columns
    assert df["importance"].between(0.0, 1.0).all()
    assert df["resolution"].isin([0.05, 0.10, 0.20, 0.50]).all()
    assert (df["occupancy"] == 1.0).all()  # evidence-presence indicator
    assert df["semantic_source"].tolist() == ["annotation", "fallback"]


def test_empty_regions_raise():
    with pytest.raises(ValueError):
        build_adaptive_map([], ImportanceEngine(), ResolutionEngine())


def test_resolution_boundaries_frozen():
    eng = ResolutionEngine()
    assert eng.assign_resolution(0.24) == 0.50
    assert eng.assign_resolution(0.25) == 0.20
    assert eng.assign_resolution(0.50) == 0.10
    assert eng.assign_resolution(0.75) == 0.05


def test_map_statistics_descriptive():
    _, df = build_adaptive_map([_region(i) for i in range(3)],
                               ImportanceEngine(), ResolutionEngine())
    stats = map_statistics(df)
    assert stats["num_cells"] == 3
    assert stats["occupied_cells"] == 3
    assert abs(sum(stats["resolution_percentage"].values()) - 100.0) < 1e-6


def test_visualization_helpers_reject_bad_input():
    with pytest.raises(ValueError):
        viz.plot_lidar_xy(np.zeros((10, 3)))
    df = pd.DataFrame([{"x": 0.0, "y": 0.0, "elevation": 0.0,
                        "semantic_class": "vehicle", "importance": 0.5,
                        "resolution": 0.1}])
    import matplotlib
    matplotlib.use("Agg")
    for fn in (viz.plot_semantic_regions, viz.plot_importance_map,
               viz.plot_resolution_map, viz.plot_elevation_map):
        ax = fn(df)
        assert ax is not None
