"""Tests for the first-handoff pipeline (10 September Together task).

All inputs are SYNTHETIC N×4 arrays in the exact processed-loader format
(finite float [x, y, z, intensity]) used ONLY to exercise the adapter and
handoff machinery. Nothing here is real nuScenes data and nothing here
validates real-world performance. Run: python -m pytest tests/ -q
"""
from typing import cast

import numpy as np
import pytest

from config.importance_config import WEIGHTS
from config.resolution_config import RESOLUTION_LEVELS
from src.feature_adapter import (
    FIRST_HANDOFF_CONFIDENCE,
    FIRST_HANDOFF_DYNAMIC_RELEVANCE,
    FIRST_HANDOFF_SEMANTIC_IMPORTANCE,
    FIRST_HANDOFF_SEMANTIC_LABEL,
    FIRST_HANDOFF_UNCERTAINTY,
    run_first_handoff,
)
from src.importance_engine import ImportanceEngine
from src.resolution_engine import ResolutionEngine


def _synthetic_loader_format_frame(seed=0, n_ground=1500, n_box=300):
    """Format-identical stand-in for a processed N×4 frame (NOT real data).

    Flat ground plane (z≈0) plus one raised box cluster, mimicking the
    numeric layout Rajashree's loader emits: finite float N×4.
    """
    rng = np.random.default_rng(seed)
    ground = np.column_stack([
        rng.uniform(-30, 30, n_ground),
        rng.uniform(-30, 30, n_ground),
        rng.normal(0.0, 0.03, n_ground),
        rng.uniform(0.2, 0.6, n_ground),
    ])
    box = np.column_stack([
        10 + rng.normal(0, 0.5, n_box),
        5 + rng.normal(0, 0.5, n_box),
        rng.normal(1.0, 0.15, n_box),
        rng.uniform(0.4, 0.8, n_box),
    ])
    return np.vstack([ground, box]).astype(np.float64)


def _meta(frame_id="frame-A", sample_token="sample-A"):
    return {
        "frame_id": frame_id,
        "sample_token": sample_token,
        "lidar_token": "lidar-A",
        "timestamp": 1532402927647958.0,
    }


def test_handoff_runs_on_loader_format_input():
    pts = _synthetic_loader_format_frame()
    frame, regions, rows = run_first_handoff(pts, _meta())
    assert frame.points.shape[1] == 4
    assert len(regions) > 0
    assert len(rows) == len(regions)
    for row in rows:
        assert 0.0 <= row["importance"] <= 1.0
        assert row["resolution_m"] in (0.05, 0.10, 0.20, 0.50)
        assert 0.0 <= row["terrain_complexity"] <= 1.0


def test_handoff_identity_preserved():
    pts = _synthetic_loader_format_frame()
    meta = _meta(frame_id="F1", sample_token="S1")
    frame, _, _ = run_first_handoff(pts, meta)
    assert frame.frame_id == "F1"
    assert frame.timestamp == pytest.approx(1532402927647958.0)


def test_handoff_uses_mandated_fallbacks_only():
    _, regions, rows = run_first_handoff(_synthetic_loader_format_frame(), _meta())
    for r, row in zip(regions, rows):
        assert r.semantic_label == FIRST_HANDOFF_SEMANTIC_LABEL == "unknown"
        assert r.semantic_importance == FIRST_HANDOFF_SEMANTIC_IMPORTANCE == 0.0
        assert r.dynamic_relevance == FIRST_HANDOFF_DYNAMIC_RELEVANCE == 0.0
        assert r.confidence == FIRST_HANDOFF_CONFIDENCE == 0.0
        assert r.uncertainty == FIRST_HANDOFF_UNCERTAINTY == 0.5
        assert row["semantic_label"] == "unknown"
        # No invented pedestrian/vehicle evidence anywhere.
        assert r.semantic_label not in ("pedestrian", "vehicle")


def test_handoff_matches_direct_engine_calls():
    """The handoff must add no notebook-specific intermediate format."""
    pts = _synthetic_loader_format_frame()
    _, regions, rows = run_first_handoff(pts, _meta())
    imp, res = ImportanceEngine(), ResolutionEngine()
    for r, row in zip(regions, rows):
        assert row["importance"] == pytest.approx(imp.calculate(r).final_importance)
        assert row["resolution_m"] == pytest.approx(res.assign_resolution(row["importance"]))


def test_handoff_does_not_retune_engines():
    """Core configs must be untouched by the integration path."""
    run_first_handoff(_synthetic_loader_format_frame(), _meta())
    assert WEIGHTS == {"distance": 0.30, "semantic": 0.30, "terrain": 0.15,
                       "dynamic": 0.15, "uncertainty": 0.10}
    assert RESOLUTION_LEVELS == [(0.75, 0.05), (0.50, 0.10), (0.25, 0.20), (0.00, 0.50)]
    assert ImportanceEngine().lambda_uncertainty == pytest.approx(0.15)


def test_handoff_is_deterministic_and_frame_independent():
    pts_a = _synthetic_loader_format_frame(seed=0)
    pts_b = _synthetic_loader_format_frame(seed=1)
    _, _, rows_a1 = run_first_handoff(pts_a, _meta(frame_id="A", sample_token="SA"))
    _, _, rows_a2 = run_first_handoff(pts_a, _meta(frame_id="A", sample_token="SA"))
    _, regions_b, rows_b = run_first_handoff(pts_b, _meta(frame_id="B", sample_token="SB"))
    # Same input -> identical output, no code change between frames.
    assert [r["importance"] for r in rows_a1] == pytest.approx([r["importance"] for r in rows_a2])
    assert len(rows_b) == len(regions_b) > 0


def test_handoff_rejects_bad_points():
    good_meta = _meta()
    with pytest.raises(ValueError):
        run_first_handoff(np.zeros((0, 4)), good_meta)  # empty
    with pytest.raises(ValueError):
        bad = _synthetic_loader_format_frame()
        bad[0, 0] = np.nan  # NaN point
        run_first_handoff(bad, good_meta)
    with pytest.raises(ValueError):
        run_first_handoff(np.zeros((10, 5)), good_meta)  # not N×4
    with pytest.raises(ValueError):
        run_first_handoff(np.zeros((10, 3)), good_meta)  # not N×4


def test_handoff_rejects_bad_metadata_and_cell():
    pts = _synthetic_loader_format_frame()
    with pytest.raises(ValueError):
        run_first_handoff(pts, {"frame_id": "x"})  # missing keys
    with pytest.raises(ValueError):
        run_first_handoff(pts, cast(dict, "not-a-dict"))  # not a dict
    with pytest.raises(ValueError):
        run_first_handoff(pts, _meta(), cell_size=0.0)
    with pytest.raises(ValueError):
        run_first_handoff(pts, _meta(), cell_size=-2.0)
