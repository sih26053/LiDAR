"""Unit tests for Importance Engine v1 (10 September 2026 task).

Synthetic-only: no Kaggle/nuScenes dependency. Expected numbers below follow
directly from the implemented formula (hand-computed in comments), they are
not external scientific claims. Run: py -m pytest tests/ -q
"""
import math

import pytest

from config.importance_config import (
    MAX_DISTANCE,
    UNCERTAINTY_LAMBDA,
    WEIGHTS,
    validate_importance_config,
)
from src.data_types import RegionFeatures
from src.importance_engine import (
    ImportanceEngine,
    apply_uncertainty_modifier,
    calculate_base_importance,
    clip01,
    extract_features,
    normalize_distance,
    validate_inputs,
)
from src.resolution_engine import ResolutionEngine


def _region_dict(
    distance=70.0,
    semantic=1.0,
    terrain=0.1,
    dynamic=1.0,
    uncertainty=0.2,
    confidence=0.8,
    region_id=0,
):
    return {
        "region_id": region_id,
        "distance": distance,
        "semantic_importance": semantic,
        "terrain_complexity": terrain,
        "dynamic_relevance": dynamic,
        "uncertainty": uncertainty,
        "confidence": confidence,
    }


def _region_features(**kw):
    base = dict(
        region_id=0, x=1.0, y=2.0, distance=70.0, elevation=0.5,
        roughness=0.1, point_density=20.0, semantic_label="pedestrian",
        semantic_importance=1.0, confidence=0.8, dynamic_relevance=1.0,
        uncertainty=0.2, point_count=20,
    )
    base.update(kw)
    return RegionFeatures(**base)


# --- clip01 ---
def test_clip01_behavior():
    assert clip01(0.5) == pytest.approx(0.5)
    assert clip01(-0.3) == 0.0
    assert clip01(1.4) == 1.0
    assert clip01(0.0) == 0.0
    assert clip01(1.0) == 1.0


def test_clip01_rejects_nan_inf():
    with pytest.raises(ValueError):
        clip01(float("nan"))
    with pytest.raises(ValueError):
        clip01(float("inf"))


# --- distance normalization ---
@pytest.mark.parametrize(
    "distance,expected",
    [(0, 1.00), (10, 0.90), (25, 0.75), (50, 0.50),
     (75, 0.25), (100, 0.00), (150, 0.00)],
)
def test_distance_normalization(distance, expected):
    assert normalize_distance(distance) == pytest.approx(expected)


def test_distance_at_0m():
    assert normalize_distance(0.0) == pytest.approx(1.0)


def test_distance_at_100m():
    assert normalize_distance(100.0) == pytest.approx(0.0)


def test_distance_beyond_100m_clips_to_zero():
    assert normalize_distance(101.0) == 0.0
    assert normalize_distance(500.0) == 0.0


def test_invalid_distance_rejected():
    with pytest.raises(ValueError):
        normalize_distance(-1.0)
    with pytest.raises(ValueError):
        normalize_distance(float("nan"))
    with pytest.raises(ValueError):
        normalize_distance(float("inf"))


def test_zero_max_distance_avoids_division_by_zero():
    with pytest.raises(ValueError):
        normalize_distance(10.0, max_distance=0.0)
    with pytest.raises(ValueError):
        normalize_distance(10.0, max_distance=-5.0)


# --- input validation ---
def test_normalized_input_validation_accepts_valid():
    assert validate_inputs(70.0, 1.0, 0.1, 1.0, 0.2, 0.8) is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"distance": -2.0},
        {"semantic_importance": 1.5},
        {"semantic_importance": -0.1},
        {"terrain_complexity": 2.0},
        {"dynamic_relevance": -0.01},
        {"uncertainty": 1.01},
        {"confidence": 1.2},
        {"uncertainty": float("nan")},
        {"distance": float("inf")},
    ],
)
def test_normalized_input_validation_rejects_invalid(kwargs):
    good = dict(distance=70.0, semantic_importance=1.0, terrain_complexity=0.1,
                dynamic_relevance=1.0, uncertainty=0.2, confidence=0.8)
    good.update(kwargs)
    with pytest.raises(ValueError):
        validate_inputs(**good)


# --- weights config ---
def test_weights_sum_validation():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6
    assert validate_importance_config() is True
    with pytest.raises(ValueError):
        validate_importance_config(
            weights={"distance": 0.5, "semantic": 0.5, "terrain": 0.5,
                     "dynamic": 0.5, "uncertainty": 0.5}
        )
    with pytest.raises(ValueError):
        validate_importance_config(
            weights={"distance": -0.1, "semantic": 0.7, "terrain": 0.15,
                     "dynamic": 0.15, "uncertainty": 0.1}
        )
    assert MAX_DISTANCE == pytest.approx(100.0)
    assert UNCERTAINTY_LAMBDA == pytest.approx(0.15)


# --- base importance ---
def test_base_importance_calculation():
    # Pedestrian @70m: D=0.3. 0.3*0.3+0.3*1.0+0.15*0.1+0.15*1.0+0.1*0.2 = 0.575
    base = calculate_base_importance(0.3, 1.0, 0.1, 1.0, 0.2)
    assert base == pytest.approx(0.575)


def test_base_importance_remains_01():
    assert calculate_base_importance(1, 1, 1, 1, 1) == pytest.approx(1.0)
    assert calculate_base_importance(0, 0, 0, 0, 0) == pytest.approx(0.0)
    for d in (0.0, 0.5, 1.0):
        for u in (0.0, 0.5, 1.0):
            out = calculate_base_importance(d, u, d, u, d)
            assert 0.0 <= out <= 1.0


# --- uncertainty modifier ---
def test_uncertainty_modifier():
    # min(1, 0.575 + 0.15*0.2) = 0.605
    assert apply_uncertainty_modifier(0.575, 0.2) == pytest.approx(0.605)
    assert apply_uncertainty_modifier(0.99, 0.9) == pytest.approx(1.0)  # clipped
    assert apply_uncertainty_modifier(0.4, 0.0) == pytest.approx(0.4)  # no-op at U=0


def test_uncertainty_cannot_reduce_final_importance():
    for base in (0.0, 0.2, 0.5, 0.9):
        assert apply_uncertainty_modifier(base, 0.7) >= base


def test_final_importance_remains_01():
    engine = ImportanceEngine()
    for dist in (0.0, 10.0, 70.0, 100.0, 250.0):
        for u in (0.0, 0.5, 1.0):
            r = engine.calculate(_region_dict(distance=dist, uncertainty=u,
                                              confidence=1.0 - u if u < 1 else 0.0))
            assert 0.0 <= r.base_importance <= 1.0
            assert 0.0 <= r.final_importance <= 1.0


# --- engine API: RegionFeatures + dict + legacy naming ---
def test_engine_accepts_region_features_directly():
    engine = ImportanceEngine()
    result = engine.calculate(_region_features())
    # Same hand-computed values as the dict form (roughness -> terrain).
    assert result.base_importance == pytest.approx(0.575)
    assert result.final_importance == pytest.approx(0.605)
    assert result.safe_importance == pytest.approx(result.final_importance)
    assert set(result.to_dict()) >= {"base_importance", "final_importance"}


def test_engine_accepts_legacy_terrain_naming():
    engine = ImportanceEngine()
    legacy = {
        "region_id": 3, "distance_m": 70.0, "semantic_importance": 1.0,
        "terrain_complexity": 0.1, "dynamic_relevance": 1.0,
        "uncertainty": 0.2, "confidence": 0.8,
    }
    result = engine.calculate(legacy)
    assert result.final_importance == pytest.approx(0.605)


def test_engine_rejects_invalid_region():
    engine = ImportanceEngine()
    with pytest.raises(ValueError):
        engine.calculate(_region_dict(distance=-5.0))
    with pytest.raises(ValueError):
        engine.calculate(_region_dict(semantic=1.5))


def test_extract_features_rejects_missing_fields():
    with pytest.raises(ValueError):
        extract_features({"distance": 10.0})  # missing the rest


# --- same-distance pedestrian vs road (key novelty demo) ---
def test_same_distance_pedestrian_vs_road():
    engine = ImportanceEngine()
    resolutions = ResolutionEngine()
    ped = engine.calculate(_region_dict(distance=70.0, semantic=1.0, terrain=0.1,
                                        dynamic=1.0, uncertainty=0.2, confidence=0.8))
    road = engine.calculate(_region_dict(distance=70.0, semantic=0.1, terrain=0.1,
                                         dynamic=0.0, uncertainty=0.1, confidence=0.9))
    # Hand-computed: ped base 0.575/final 0.605; road base 0.145/final 0.16.
    assert ped.base_importance == pytest.approx(0.575)
    assert road.base_importance == pytest.approx(0.145)
    assert ped.final_importance > road.final_importance  # not distance-only
    ped_res = resolutions.select_resolution(ped.final_importance)
    road_res = resolutions.select_resolution(road.final_importance)
    assert ped_res <= road_res  # smaller cell = finer detail for pedestrian


# --- monotonicity / sanity ---
def test_monotonic_distance_behavior():
    engine = ImportanceEngine()
    finals = [
        engine.calculate(_region_dict(distance=d)).final_importance
        for d in (10.0, 30.0, 60.0, 100.0)
    ]
    assert finals[0] > finals[1] > finals[2] > finals[3]


def test_monotonic_uncertainty_behavior():
    engine = ImportanceEngine()
    finals = [
        engine.calculate(_region_dict(uncertainty=u, confidence=1.0 - u)).final_importance
        for u in (0.0, 0.3, 0.6, 0.9)
    ]
    assert finals[0] <= finals[1] <= finals[2] <= finals[3]


def test_semantic_ordering_at_same_distance():
    engine = ImportanceEngine()
    ped = engine.calculate(_region_dict(semantic=1.0, dynamic=1.0)).final_importance
    veh = engine.calculate(_region_dict(semantic=0.9, dynamic=0.8)).final_importance
    road = engine.calculate(_region_dict(semantic=0.1, dynamic=0.0)).final_importance
    assert ped > road
    assert veh > road


# --- synthetic test matrix (8 scenarios, computed from the actual engine) ---
SYNTHETIC_MATRIX = [
    # (name, distance, semantic, terrain, dynamic, uncertainty, confidence)
    ("nearby pedestrian", 10.0, 1.00, 0.10, 1.00, 0.20, 0.80),
    ("distant pedestrian", 70.0, 1.00, 0.10, 1.00, 0.20, 0.80),
    ("nearby vehicle", 15.0, 0.90, 0.10, 0.80, 0.15, 0.85),
    ("distant vehicle", 80.0, 0.90, 0.10, 0.80, 0.15, 0.85),
    ("smooth empty road", 40.0, 0.10, 0.05, 0.00, 0.10, 0.90),
    ("rough terrain", 30.0, 0.45, 0.90, 0.05, 0.20, 0.80),
    ("unknown obstacle, high uncertainty", 50.0, 0.80, 0.30, 0.40, 0.90, 0.10),
    ("low-confidence region", 25.0, 0.30, 0.20, 0.20, 0.80, 0.20),
]


def test_synthetic_matrix_computed_from_engine():
    engine = ImportanceEngine()
    resolutions = ResolutionEngine()
    print("\n| Region | Dist | Sem | Terr | Dyn | Unc | Base | Final | Res |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, dist, sem, terr, dyn, unc, conf in SYNTHETIC_MATRIX:
        r = engine.calculate(_region_dict(distance=dist, semantic=sem, terrain=terr,
                                          dynamic=dyn, uncertainty=unc, confidence=conf))
        res = resolutions.select_resolution(r.final_importance)
        print(f"| {name} | {dist} | {sem} | {terr} | {dyn} | {unc} "
              f"| {r.base_importance:.3f} | {r.final_importance:.3f} | {res} |")
        assert 0.0 <= r.base_importance <= 1.0
        assert 0.0 <= r.final_importance <= 1.0
        assert res in (0.05, 0.10, 0.20, 0.50)
        assert r.final_importance >= r.base_importance  # safety term never reduces
    # Spot ordering checks on computed outputs:
    finals = {
        name: engine.calculate(_region_dict(
            distance=d, semantic=s, terrain=t, dynamic=m,
            uncertainty=u, confidence=c)).final_importance
        for name, d, s, t, m, u, c in SYNTHETIC_MATRIX
    }
    assert finals["nearby pedestrian"] > finals["distant pedestrian"]
    assert finals["nearby vehicle"] > finals["distant vehicle"]
    assert finals["nearby pedestrian"] > finals["smooth empty road"]
    assert finals["rough terrain"] > finals["smooth empty road"]
    assert math.isfinite(finals["unknown obstacle, high uncertainty"])


def test_engine_is_deterministic():
    engine = ImportanceEngine()
    region = _region_dict()
    assert engine.calculate(region).final_importance == engine.calculate(region).final_importance
