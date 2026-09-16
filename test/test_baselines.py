"""Tests for src.baselines (13 September benchmark baselines).

Fast unit tests only (small synthetic evaluation areas; no nuScenes I/O).
"""
import numpy as np
import pandas as pd
import pytest

from src.baselines import (
    UNIFORM_RESOLUTION,
    build_distance_adaptive_map,
    build_uniform_map,
    distance_based_resolution,
    evaluation_area_from_regions,
    resolution_density_proxy,
)
from src.data_types import RegionFeatures


def _region(rid=0, x=5.0, y=0.0, label="vehicle"):
    return RegionFeatures(
        region_id=rid, x=x, y=y,
        distance=float(np.hypot(x, y)), elevation=1.0, roughness=0.2,
        point_density=10.0, semantic_label=label, semantic_importance=0.9,
        confidence=0.0, dynamic_relevance=0.8, uncertainty=0.5,
        point_count=40,
    )


def test_distance_bins_frozen_policy():
    assert distance_based_resolution(0.0) == 0.05
    assert distance_based_resolution(9.99) == 0.05
    assert distance_based_resolution(10.0) == 0.10
    assert distance_based_resolution(29.99) == 0.10
    assert distance_based_resolution(30.0) == 0.20
    assert distance_based_resolution(59.99) == 0.20
    assert distance_based_resolution(60.0) == 0.50
    assert distance_based_resolution(100.0) == 0.50


def test_distance_rejects_bad_input():
    with pytest.raises(ValueError):
        distance_based_resolution(-1.0)
    with pytest.raises(ValueError):
        distance_based_resolution(float("nan"))


def test_distance_map_never_uses_importance_engine():
    regions = [_region(0, 5, 0), _region(1, 50, 0, "road")]
    df = build_distance_adaptive_map(regions)
    assert list(df["resolution"]) == [0.05, 0.20]
    assert "ImportanceEngine" not in build_distance_adaptive_map.__code__.co_names
    assert len(df) == len(regions)


def test_uniform_grid_exact_count_small_area():
    area = {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0,
            "coverage_area": 1.0}
    res = build_uniform_map(area, resolution_m=0.05)
    assert res.nx == 21 and res.ny == 21  # inclusive arange
    assert res.exact_cells == 21 * 21
    assert res.occupied_cells == res.exact_cells
    assert res.resolution_m == UNIFORM_RESOLUTION


def test_uniform_grid_uses_same_evaluation_area():
    regions = [_region(0, 5, 0), _region(1, -3, 4)]
    area = evaluation_area_from_regions(regions, pad_m=1.0)
    res = build_uniform_map(area)
    assert res.x_min == area["x_min"] and res.y_max == area["y_max"]
    assert res.coverage_area == area["coverage_area"]


def test_resolution_density_proxy_label():
    assert resolution_density_proxy([0.05]) == pytest.approx(400.0)
    assert resolution_density_proxy([0.05, 0.10]) == pytest.approx(500.0)
    with pytest.raises(ValueError):
        resolution_density_proxy([])
