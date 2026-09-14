"""End-to-end Stage-1 integration test (10 September 2026 task).

Proves the new v1 engines plug into the existing 9-September Stage-1
architecture without redesign, using synthetic data only:

    Synthetic Region
        -> existing data/interface layer (LiDARFrame/PerceptionResult)
        -> Feature Adapter (PerceptionResult -> RegionFeatures)
        -> Importance Engine v1 -> final importance
        -> Resolution Engine v1 -> selected resolution
        -> existing Adaptive 2.5D Mapper interface

Run: py -m pytest tests/test_stage1_integration.py -q
"""
import numpy as np

from src.data_types import RegionFeatures
from src.feature_adapter import (
    adapt_perception_to_regions,
    build_perception_from_labeled_points,
)
from src.importance_engine import ImportanceEngine
from src.interface_validator import validate_region_features
from src.resolution_engine import ResolutionEngine
from src.stage1_engines import AdaptiveMapper2_5D
from src.stage1_engines import ImportanceEngine as LegacyImportanceEngine
from src.stage1_engines import ResolutionEngine as LegacyResolutionEngine


def _synthetic_70m_scene(seed=42):
    """Pedestrian-like cluster + road-like plane, both ~70 m from the sensor."""
    rng = np.random.default_rng(seed)
    ped = np.column_stack([
        70 + rng.normal(0, 0.2, 200), 2 + rng.normal(0, 0.2, 200),
        rng.normal(0.9, 0.1, 200), rng.uniform(0.4, 0.7, 200),
    ])
    road = np.column_stack([
        70 + rng.uniform(-4, 4, 300), -6 + rng.uniform(-2, 2, 300),
        rng.normal(0, 0.02, 300), rng.uniform(0.2, 0.5, 300),
    ])
    pts = np.vstack([ped, road])
    labels = np.array(["pedestrian"] * len(ped) + ["road"] * len(road), dtype=str)
    conf = np.array([0.85] * len(ped) + [0.95] * len(road), dtype=float)
    return pts, labels, conf


def test_stage1_integration_synthetic_to_mapper():
    pts, labels, conf = _synthetic_70m_scene()

    # 1. Existing interface layer.
    pr = build_perception_from_labeled_points("stage1-e2e", pts, labels, conf)

    # 2. Existing feature adapter -> RegionFeatures (9-Sept contract preserved).
    regions = adapt_perception_to_regions(pr, bin_m=4.0, min_points=15)
    assert len(regions) >= 2
    for r in regions:
        assert isinstance(r, RegionFeatures)
        assert validate_region_features(r) is True

    # 3. New Importance Engine v1 consumes RegionFeatures directly (no redesign).
    imp = ImportanceEngine()
    scored = [(r, imp.calculate(r)) for r in regions]
    for r, s in scored:
        assert 0.0 <= s.base_importance <= 1.0
        assert 0.0 <= s.final_importance <= 1.0

    # 4. New Resolution Engine v1 consumes final importance only.
    res = ResolutionEngine()
    resolved = []
    for r, s in scored:
        res_m = res.select_resolution(s.final_importance)
        assert res_m in (0.05, 0.10, 0.20, 0.50)
        resolved.append((r, s, res_m))

    # 5. Semantic-vs-distance check on the integrated path (both clusters ~70 m).
    by_label = {}
    for r, s, res_m in resolved:
        by_label.setdefault(r.semantic_label, []).append((s.final_importance, res_m))
    assert "pedestrian" in by_label and "road" in by_label
    ped_best = max(f for f, _ in by_label["pedestrian"])
    road_best = max(f for f, _ in by_label["road"])
    assert ped_best > road_best
    ped_res = min(m for f, m in by_label["pedestrian"] if f == ped_best)
    road_res = min(m for f, m in by_label["road"] if f == road_best)
    assert ped_res <= road_res

    # 6. Existing Adaptive 2.5D Mapper interface accepts the same regions.
    ix = np.floor(pts[:, 0] / 4.0).astype(int)
    iy = np.floor(pts[:, 1] / 4.0).astype(int)
    _, inv = np.unique(np.column_stack([ix, iy]), axis=0, return_inverse=True)
    full_legacy = []
    for r in regions:
        mask = inv == r.region_id
        if mask.sum() == 0:
            continue
        full_legacy.append(r.to_legacy_region(pts[mask]))
    assert len(full_legacy) > 0
    mapper = AdaptiveMapper2_5D(LegacyImportanceEngine(), LegacyResolutionEngine())
    mres = mapper.map_regions_legacy(full_legacy)
    assert len(mres.cells) > 0
    assert len(mres.importance) > 0
    assert mres.n_points == sum(int((inv == r.region_id).sum()) for r in regions
                                if (inv == r.region_id).sum() > 0)
    for cell in mres.cells:
        assert cell.resolution in (0.05, 0.10, 0.20, 0.50)
        assert 0.0 <= cell.importance <= 1.0
