"""Unit + integration tests for the common data interface.

Synthetic-only. No dataset download, no Kaggle credentials, no fabricated
real-world metrics. Run: py -m pytest tests/test_data_interface.py -q
(or: py -m unittest discover -s tests -v)
"""
import numpy as np
import pytest

from src.data_types import LiDARFrame, PerceptionResult, RegionFeatures
from src.feature_adapter import (
    adapt_perception_to_regions,
    build_perception_from_labeled_points,
)
from src.interface_validator import (
    InterfaceError,
    validate_lidar_frame,
    validate_perception_result,
    validate_region_features,
)
from src.stage1_engines import (
    AdaptiveMapper2_5D,
    ImportanceEngine,
    ResolutionEngine,
)


def _frame(n=50, seed=0):
    rng = np.random.default_rng(seed)
    pts = np.column_stack([
        rng.uniform(-10, 10, n),
        rng.uniform(-10, 10, n),
        rng.uniform(-1, 2, n),
        rng.uniform(0, 1, n),
    ])
    return LiDARFrame(frame_id="synth-001", timestamp=0.0, points=pts)


def _perception(n=60, seed=1):
    rng = np.random.default_rng(seed)
    pts = np.column_stack([
        rng.uniform(-8, 8, n),
        rng.uniform(-8, 8, n),
        rng.uniform(0, 1, n),
        rng.uniform(0, 1, n),
    ])
    labels = np.array(["road"] * n, dtype=str)
    return build_perception_from_labeled_points("synth-002", pts, labels, [0.9] * n)


def _region(**kw):
    base = dict(
        region_id=0, x=1.0, y=2.0, distance=2.236, elevation=0.5,
        roughness=0.1, point_density=20.0, semantic_label="road",
        semantic_importance=0.1, confidence=0.9, dynamic_relevance=0.0,
        uncertainty=0.1, point_count=20,
    )
    base.update(kw)
    return RegionFeatures(**base)


# --- LiDARFrame ---
def test_valid_lidar_frame_passes():
    assert validate_lidar_frame(_frame()) is True


def test_invalid_lidar_shape_fails():
    bad = _frame()
    bad.points = bad.points[:, :3]  # (N,3)
    with pytest.raises(InterfaceError):
        validate_lidar_frame(bad)


def test_nan_inf_lidar_fails():
    bad = _frame()
    bad.points[0, 0] = np.nan
    with pytest.raises(InterfaceError):
        validate_lidar_frame(bad)
    bad2 = _frame()
    bad2.points[1, 2] = np.inf
    with pytest.raises(InterfaceError):
        validate_lidar_frame(bad2)


# --- PerceptionResult ---
def test_correct_point_level_lengths_pass():
    assert validate_perception_result(_perception()) is True


def test_mismatched_array_lengths_fail():
    pr = _perception()
    pr.confidence = pr.confidence[:-1]
    with pytest.raises(InterfaceError):
        validate_perception_result(pr)


def test_confidence_outside_01_fails():
    pr = _perception()
    pr.confidence[0] = 1.37
    with pytest.raises(InterfaceError, match="confidence"):
        validate_perception_result(pr)


# --- RegionFeatures ---
def test_invalid_importance_fails():
    with pytest.raises(InterfaceError):
        validate_region_features(_region(semantic_importance=1.5))


def test_negative_distance_fails():
    with pytest.raises(InterfaceError):
        validate_region_features(_region(distance=-2.0))


def test_negative_point_count_fails():
    with pytest.raises(InterfaceError):
        validate_region_features(_region(point_count=-1))


def test_unknown_semantic_label_accepted_when_confidence_zero():
    r = _region(semantic_label="unknown", semantic_importance=0.0,
                confidence=0.0, uncertainty=0.5)
    assert validate_region_features(r) is True


# --- Adapter ---
def test_perception_to_regions_produces_valid_regions():
    pr = _perception(n=120)
    regions = adapt_perception_to_regions(pr, bin_m=4.0, min_points=5)
    assert len(regions) > 0
    for r in regions:
        assert validate_region_features(r) is True


def test_missing_semantic_maps_to_unknown():
    rng = np.random.default_rng(7)
    pts = np.column_stack([rng.uniform(0, 2, 40), rng.uniform(0, 2, 40),
                           rng.uniform(0, 1, 40), rng.uniform(0, 1, 40)])
    pr = PerceptionResult(
        frame_id="s", points=pts,
        semantic_labels=np.array([""] * 40, dtype=str),
        confidence=np.zeros(40), distance=np.hypot(pts[:, 0], pts[:, 1]),
        elevation=pts[:, 2], roughness=np.zeros(40), point_density=np.ones(40),
    )
    regions = adapt_perception_to_regions(pr, bin_m=100.0, min_points=5)
    assert regions and regions[0].semantic_label == "unknown"
    assert regions[0].semantic_importance == 0.0


# --- Full integration: synthetic -> engine -> resolution -> mapper ---
def test_full_integration_synthetic_reaches_mapper():
    rng = np.random.default_rng(42)
    # Pedestrian-like cluster near (70, 2), road-like plane near (70, -6).
    ped = np.column_stack([70 + rng.normal(0, 0.2, 200), 2 + rng.normal(0, 0.2, 200),
                           rng.normal(0.9, 0.1, 200), rng.uniform(0.4, 0.7, 200)])
    road = np.column_stack([70 + rng.uniform(-4, 4, 300), -6 + rng.uniform(-2, 2, 300),
                            rng.normal(0, 0.02, 300), rng.uniform(0.2, 0.5, 300)])
    pts = np.vstack([ped, road])
    labels = np.array(["pedestrian"] * len(ped) + ["road"] * len(road), dtype=str)
    conf = np.array([0.85] * len(ped) + [0.95] * len(road), dtype=float)
    pr = build_perception_from_labeled_points("usp-70m", pts, labels, conf)
    regions = adapt_perception_to_regions(pr, bin_m=4.0, min_points=15)
    assert len(regions) >= 2

    imp, res = ImportanceEngine(), ResolutionEngine()
    # Scoring needs no point cloud (engine uses scalar features only);
    # pass one dummy Nx4 row per region to satisfy the legacy points contract.
    legacy = [
        r.to_legacy_region(np.array([[r.x, r.y, r.elevation, 0.5]]))
        for r in regions
    ]
    # Score + resolve (proves consumer accepts standardized RegionFeatures).
    scored = [imp.score_region(lr) for lr in legacy]
    for s in scored:
        res_m, lvl = res.select(s.safe_importance)
        s.selected_resolution_m, s.resolution_level = res_m, lvl
        assert res_m in (0.05, 0.10, 0.20, 0.50)

    # Mapper on real region point clouds (group by same 4 m grid):
    ix = np.floor(pts[:, 0] / 4.0).astype(int)
    iy = np.floor(pts[:, 1] / 4.0).astype(int)
    _, inv = np.unique(np.column_stack([ix, iy]), axis=0, return_inverse=True)
    full_legacy, by_label = [], {}
    for r in regions:
        mask = inv == r.region_id
        if mask.sum() == 0:
            continue
        full_legacy.append(r.to_legacy_region(pts[mask]))
        by_label.setdefault(r.semantic_label, []).append(r)
    mapper = AdaptiveMapper2_5D(imp, res)
    mres = mapper.map_regions_legacy(full_legacy)
    assert len(mres.cells) > 0 and len(mres.importance) > 0
    # Semantic-vs-distance USP on measured outputs (both ~70 m):
    assert "pedestrian" in by_label and "road" in by_label
