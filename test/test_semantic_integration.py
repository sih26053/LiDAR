"""Tests for 11-September semantic + geometric integration.

All inputs are SYNTHETIC N x 4 stand-ins (never real nuScenes data, never
presented as measured perception). They exercise the mapping, geometry,
aggregation, and engine-handoff machinery only. Run: python -m pytest -q
"""
import numpy as np
import pytest

from config.importance_config import WEIGHTS
from config.resolution_config import RESOLUTION_LEVELS
from src.data_types import PerceptionResult
from src.interface_validator import (
    InterfaceError,
    validate_perception_result,
    validate_region_features,
)
from src.semantic_adapter import (
    aggregate_to_regions,
    assign_annotation_semantics,
    build_perception_from_semantics,
    compute_point_geometry,
    points_in_box,
)
from src.semantic_mapping import (
    PROJECT_CLASSES,
    PROJECT_SEMANTIC_IMPORTANCE,
    map_lidarseg_id_to_project,
    map_nuscenes_label_to_project,
)
from src.importance_engine import ImportanceEngine
from src.resolution_engine import ResolutionEngine


def _synthetic_frame(seed=0, n_ground=800, n_box=200):
    rng = np.random.default_rng(seed)
    ground = np.column_stack([
        rng.uniform(-20, 20, n_ground),
        rng.uniform(-20, 20, n_ground),
        rng.normal(0.0, 0.03, n_ground),
        rng.uniform(0.2, 0.6, n_ground),
    ])
    box = np.column_stack([
        8 + rng.normal(0, 0.5, n_box),
        4 + rng.normal(0, 0.5, n_box),
        rng.normal(1.0, 0.15, n_box),
        rng.uniform(0.4, 0.8, n_box),
    ])
    return np.vstack([ground, box]).astype(np.float64)


# --- Taxonomy / mapping ---

def test_project_taxonomy_has_exactly_six_classes():
    assert set(PROJECT_CLASSES) == {
        "vehicle", "pedestrian_vru", "static_manmade",
        "vegetation", "road_driveable", "unknown",
    }


def test_nuscenes_label_mapping_returns_only_project_labels():
    for raw in ["vehicle.car", "vehicle.truck", "human.pedestrian.adult",
                "movable_object.barrier", "static.manmade", "static.vegetation",
                "flat.driveable_surface", "noise", "", "something.weird"]:
        assert map_nuscenes_label_to_project(raw) in PROJECT_CLASSES
    assert map_nuscenes_label_to_project("vehicle.car") == "vehicle"
    assert map_nuscenes_label_to_project("human.pedestrian.adult") == "pedestrian_vru"
    assert map_nuscenes_label_to_project("flat.driveable_surface") == "road_driveable"
    assert map_nuscenes_label_to_project("static.vegetation") == "vegetation"
    assert map_nuscenes_label_to_project("bogus.label") == "unknown"


def test_lidarseg_id_mapping_covers_0_to_31():
    for lid in range(32):
        original, project = map_lidarseg_id_to_project(lid)
        assert project in PROJECT_CLASSES
        assert isinstance(original, str) and original
    _, proj_car = map_lidarseg_id_to_project(17)  # vehicle.car
    assert proj_car == "vehicle"
    _, proj_road = map_lidarseg_id_to_project(24)  # flat.driveable_surface
    assert proj_road == "road_driveable"


def test_project_semantic_importance_config():
    assert PROJECT_SEMANTIC_IMPORTANCE == {
        "vehicle": 1.0, "pedestrian_vru": 1.0, "static_manmade": 0.7,
        "vegetation": 0.3, "road_driveable": 0.1, "unknown": 0.5,
    }


# --- Geometry ---

def test_point_geometry_distance_and_elevation():
    pts = np.array([[3.0, 4.0, 2.0, 0.5], [0.0, 0.0, -1.0, 0.1]])
    dist, elev = compute_point_geometry(pts)
    assert dist == pytest.approx([np.sqrt(9 + 16 + 4), 1.0])
    assert elev == pytest.approx([2.0, -1.0])


def test_point_geometry_rejects_bad_input():
    with pytest.raises(ValueError):
        compute_point_geometry(np.zeros((0, 4)))
    bad = _synthetic_frame()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        compute_point_geometry(bad)
    with pytest.raises(ValueError):
        compute_point_geometry(np.zeros((10, 3)))


# --- Annotation fallback honesty ---

def test_annotation_fallback_assigns_inside_only():
    pts = _synthetic_frame()
    anns = [{
        "category_name": "vehicle.car",
        "translation": [8.0, 4.0, 1.0],
        "size": [2.0, 4.0, 2.0],
        "rotation": [1.0, 0.0, 0.0, 0.0],
    }]
    labels, sources, conf = assign_annotation_semantics(pts, anns)
    assert len(labels) == len(pts)
    assert set(np.unique(sources)).issubset({"annotation", "fallback"})
    assert (sources == "annotation").sum() > 0
    assert (sources == "fallback").sum() > 0
    # Outside points are unknown, never a specific object.
    assert set(labels[sources == "fallback"]) == {"unknown"}
    # No ML confidence is fabricated.
    assert np.all(~np.isfinite(conf))


def test_points_in_box_axis_aligned():
    xyz = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    mask = points_in_box(xyz, [0, 0, 0], [2, 2, 2], None)
    assert mask.tolist() == [True, False]


def test_points_in_box_uses_length_along_x():
    # size (w=2, l=4, h=2): x extent = l/2 = 2, y extent = w/2 = 1.
    xyz = np.array([[1.9, 0.0, 0.0], [0.0, 1.1, 0.0]])
    mask = points_in_box(xyz, [0, 0, 0], [2, 4, 2], None)
    assert mask.tolist() == [True, False]


class _FakeNuscFrames:
    """Minimal stub: identity ego pose, translated calibrated sensor."""

    def __init__(self, ego_t, cs_t):
        self._ego_t = list(ego_t)
        self._cs_t = list(cs_t)

    def get(self, table, token):
        if table == "sample":
            return {"token": "S", "data": {"LIDAR_TOP": "SD"},
                    "anns": ["A1"]}
        if table == "sample_data":
            return {"token": "SD", "calibrated_sensor_token": "CS",
                    "ego_pose_token": "EP"}
        if table == "calibrated_sensor":
            return {"translation": self._cs_t, "rotation": [1, 0, 0, 0]}
        if table == "ego_pose":
            return {"translation": self._ego_t, "rotation": [1, 0, 0, 0]}
        assert table == "sample_annotation"
        return {"category_name": "vehicle.car", "translation": [10, 0, 0],
                "size": [2, 4, 2], "rotation": [1, 0, 0, 0],
                "instance_token": "I1"}


def test_annotations_to_sensor_frame_identity():
    from src.semantic_adapter import annotations_to_sensor_frame
    nusc = _FakeNuscFrames([0, 0, 0], [0, 0, 0])
    out = annotations_to_sensor_frame(nusc, nusc.get("sample", "S"))
    assert len(out) == 1
    assert np.allclose(out[0]["translation"], [10, 0, 0])
    assert out[0]["category_name"] == "vehicle.car"


def test_annotations_to_sensor_frame_translates():
    from src.semantic_adapter import annotations_to_sensor_frame
    nusc = _FakeNuscFrames([8, 0, 0], [2, 0, 0])  # global 10 -> sensor 0
    out = annotations_to_sensor_frame(nusc, nusc.get("sample", "S"))
    assert np.allclose(out[0]["translation"], [0, 0, 0])


# --- PerceptionResult + confidence rule ---

def test_non_model_perception_uses_nan_confidence_and_validates():
    pts = _synthetic_frame(n_ground=200, n_box=50)
    labels = np.array(["road_driveable"] * len(pts), dtype=str)
    pr, sources, _ = build_perception_from_semantics(
        "synth-001", pts, labels, "annotation")
    assert validate_perception_result(pr) is True
    assert np.all(~np.isfinite(pr.confidence))  # not applicable, not 0.95
    assert set(np.unique(sources)) == {"annotation"}


def test_model_perception_requires_real_confidence():
    pts = _synthetic_frame(n_ground=200, n_box=50)
    labels = np.array(["vehicle"] * len(pts), dtype=str)
    conf = np.full(len(pts), 0.82)
    pr, _, _ = build_perception_from_semantics(
        "synth-002", pts, labels, "model", confidence=conf)
    assert validate_perception_result(pr) is True
    assert np.all((pr.confidence >= 0.0) & (pr.confidence <= 1.0))


def test_fabricated_confidence_for_non_model_is_rejected():
    pts = _synthetic_frame(n_ground=200, n_box=50)
    labels = np.array(["vehicle"] * len(pts), dtype=str)
    with pytest.raises(ValueError):
        build_perception_from_semantics(
            "synth-003", pts, labels, "annotation",
            confidence=np.full(len(pts), 0.95))


def test_model_source_without_confidence_is_rejected():
    pts = _synthetic_frame(n_ground=200, n_box=50)
    labels = np.array(["vehicle"] * len(pts), dtype=str)
    with pytest.raises(ValueError):
        build_perception_from_semantics("synth-004", pts, labels, "model")


def test_validator_still_rejects_inf_and_out_of_range():
    pts = _synthetic_frame(n_ground=200, n_box=50)
    labels = np.array(["vehicle"] * len(pts), dtype=str)
    pr, _, _ = build_perception_from_semantics(
        "synth-005", pts, labels, "model",
        confidence=np.full(len(pts), 0.5))
    pr.confidence[0] = 1.37
    with pytest.raises(InterfaceError):
        validate_perception_result(pr)
    pr.confidence[0] = np.inf
    with pytest.raises(InterfaceError):
        validate_perception_result(pr)


# --- Region aggregation + engine handoff ---

def test_aggregation_produces_valid_regions_with_audit():
    pts = _synthetic_frame()
    anns = [{
        "category_name": "vehicle.car",
        "translation": [8.0, 4.0, 1.0],
        "size": [2.0, 4.0, 2.0],
        "rotation": [1.0, 0.0, 0.0, 0.0],
    }]
    labels, sources, _ = assign_annotation_semantics(pts, anns)
    pr, src_arr, _ = build_perception_from_semantics(
        "synth-006", pts, labels, sources)
    regions, details = aggregate_to_regions(pr, src_arr, cell_size=2.0)
    assert len(regions) > 0
    for r in regions:
        assert validate_region_features(r) is True
    assert len(details) == len(regions)
    for d in details:
        assert d["semantic_source"] in ("annotation", "fallback")
        assert 0.0 <= d["dominant_class_ratio"] <= 1.0
        assert 0.0 <= d["terrain_complexity"] <= 1.0
        assert "uncertainty_source" in d and "dynamic_source" in d


def test_semantic_regions_reach_engines_unchanged():
    pts = _synthetic_frame()
    n = len(pts)
    labels = np.array(["unknown"] * n, dtype=object)
    labels[800:] = "vehicle"  # box cluster votes vehicle
    sources = np.array(["fallback"] * n, dtype=object)
    sources[800:] = "annotation"
    pr, src_arr, _ = build_perception_from_semantics(
        "synth-007", pts, labels, sources)
    regions, _ = aggregate_to_regions(pr, src_arr, cell_size=2.0)
    imp, res = ImportanceEngine(), ResolutionEngine()
    for r in regions:
        out = imp.calculate(r)
        assert 0.0 <= out.final_importance <= 1.0
        assert res.assign_resolution(out.final_importance) in (0.05, 0.10, 0.20, 0.50)
    # Engine configs untouched by the semantic path.
    assert WEIGHTS == {"distance": 0.30, "semantic": 0.30, "terrain": 0.15,
                       "dynamic": 0.15, "uncertainty": 0.10}
    assert RESOLUTION_LEVELS == [(0.75, 0.05), (0.50, 0.10), (0.25, 0.20), (0.00, 0.50)]


def test_vehicle_region_outranks_road_region_at_same_distance():
    # Two compact clusters at ~same range with different semantics.
    rng = np.random.default_rng(11)
    veh = np.column_stack([
        20 + rng.normal(0, 0.15, 300), rng.normal(0, 0.15, 300),
        rng.normal(1.0, 0.05, 300), rng.uniform(0.4, 0.7, 300)])
    road = np.column_stack([
        20 + rng.normal(0, 0.15, 300), 10 + rng.normal(0, 0.15, 300),
        rng.normal(0.0, 0.02, 300), rng.uniform(0.2, 0.5, 300)])
    pts = np.vstack([veh, road])
    labels = np.array(["vehicle"] * len(veh) + ["road_driveable"] * len(road))
    sources = np.array(["annotation"] * len(pts))
    pr, src_arr, _ = build_perception_from_semantics(
        "synth-008", pts, labels, sources)
    regions, _ = aggregate_to_regions(pr, src_arr, cell_size=2.0)
    imp = ImportanceEngine()
    scored = [(r.semantic_label, imp.calculate(r).final_importance) for r in regions]
    veh_scores = [s for lab, s in scored if lab == "vehicle"]
    road_scores = [s for lab, s in scored if lab == "road_driveable"]
    assert veh_scores and road_scores
    assert max(veh_scores) > max(road_scores)


def test_perception_result_point_level_contract():
    pts = _synthetic_frame(n_ground=200, n_box=50)
    labels = np.array(["vegetation"] * len(pts), dtype=str)
    pr, _, _ = build_perception_from_semantics(
        "synth-009", pts, labels, "lidarseg")
    assert pr.points.ndim == 2 and pr.points.shape[1] == 4
    assert np.isfinite(pr.points).all()
    for arr in (pr.semantic_labels, pr.confidence, pr.distance, pr.elevation):
        assert len(arr) == len(pts)
    assert isinstance(pr, PerceptionResult)
