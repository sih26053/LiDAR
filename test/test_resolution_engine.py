"""Unit tests for Resolution Engine v1 (10 September 2026 task).

Synthetic-only. Boundary expectations follow directly from the configured
policy (0.75/0.50/0.25 -> 0.05/0.10/0.20/0.50 m). Run: py -m pytest tests/ -q
"""
import pytest

from config.resolution_config import RESOLUTION_LEVELS, validate_resolution_config
from src.resolution_engine import ResolutionEngine, validate_importance


def test_resolution_config_valid():
    assert validate_resolution_config() is True
    assert RESOLUTION_LEVELS == [(0.75, 0.05), (0.50, 0.10), (0.25, 0.20), (0.00, 0.50)]
    with pytest.raises(ValueError):
        validate_resolution_config([(0.5, 0.05), (0.75, 0.10), (0.25, 0.20), (0.0, 0.50)])


def test_high_importance_gives_finest_resolution():
    assert ResolutionEngine().select_resolution(1.00) == pytest.approx(0.05)
    assert ResolutionEngine().select_resolution(0.90) == pytest.approx(0.05)


def test_low_importance_gives_coarsest_resolution():
    assert ResolutionEngine().select_resolution(0.00) == pytest.approx(0.50)
    assert ResolutionEngine().select_resolution(0.10) == pytest.approx(0.50)


def test_threshold_075_boundary():
    engine = ResolutionEngine()
    assert engine.select_resolution(0.75) == pytest.approx(0.05)
    assert engine.select_resolution(0.7499) == pytest.approx(0.10)


def test_threshold_050_boundary():
    engine = ResolutionEngine()
    assert engine.select_resolution(0.50) == pytest.approx(0.10)
    assert engine.select_resolution(0.4999) == pytest.approx(0.20)


def test_threshold_025_boundary():
    engine = ResolutionEngine()
    assert engine.select_resolution(0.25) == pytest.approx(0.20)
    assert engine.select_resolution(0.2499) == pytest.approx(0.50)


def test_threshold_000_boundary():
    assert ResolutionEngine().select_resolution(0.00) == pytest.approx(0.50)


@pytest.mark.parametrize(
    "importance,expected",
    [
        (1.00, 0.05), (0.80, 0.05), (0.75, 0.05),
        (0.70, 0.10), (0.50, 0.10),
        (0.30, 0.20), (0.25, 0.20),
        (0.20, 0.50), (0.00, 0.50),
    ],
)
def test_exact_boundary_table(importance, expected):
    assert ResolutionEngine().select_resolution(importance) == pytest.approx(expected)


@pytest.mark.parametrize(
    "importance,expected",
    [
        (0.00, 0.50),
        (0.24, 0.50),
        (0.25, 0.20),
        (0.49, 0.20),
        (0.50, 0.10),
        (0.74, 0.10),
        (0.75, 0.05),
        (1.00, 0.05),
    ],
)
def test_spec_boundary_table(importance, expected):
    """Task-spec Section 16 boundary table (both API names, identical)."""
    engine = ResolutionEngine()
    assert engine.select_resolution(importance) == pytest.approx(expected)
    assert engine.assign_resolution(importance) == pytest.approx(expected)


@pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0, float("nan"), float("inf"), "high", None])
def test_invalid_importance_handling(bad):
    with pytest.raises(ValueError):
        ResolutionEngine().select_resolution(bad)
    with pytest.raises(ValueError):
        validate_importance(bad)


def test_higher_importance_maps_to_equal_or_finer_resolution():
    engine = ResolutionEngine()
    importances = [0.0, 0.1, 0.24, 0.25, 0.4, 0.49, 0.5, 0.6, 0.74, 0.75, 0.9, 1.0]
    resolutions = [engine.select_resolution(i) for i in importances]
    for higher_res, lower_res in zip(resolutions, resolutions[1:]):
        # importances ascend -> cell sizes must not grow (finer or equal).
        assert lower_res <= higher_res
    assert resolutions == sorted(resolutions, reverse=True)


def test_select_alias_returns_name():
    res_m, level = ResolutionEngine().select(0.8)
    assert res_m == pytest.approx(0.05)
    assert isinstance(level, str) and level


def test_thresholds_configurable():
    custom = ResolutionEngine(levels=[(0.9, 0.05), (0.6, 0.10), (0.3, 0.20), (0.0, 0.50)])
    assert custom.select_resolution(0.8) == pytest.approx(0.10)
    assert custom.select_resolution(0.95) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        ResolutionEngine(levels=[(0.5, 0.05), (0.75, 0.10), (0.25, 0.20), (0.0, 0.50)])
