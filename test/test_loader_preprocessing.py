"""Tests for the 10-September loader + preprocessing pipeline.

All inputs are SYNTHETIC (hand-built .bin buffers and RNG point clouds).
No dataset, no download. Run: python -m pytest tests/ -q
"""
import os
import numpy as np
import pandas as pd
import pytest

from src.lidar_loader import load_lidar_bin, load_sample_lidar
from src.preprocessing import preprocess_points, save_processed_frame


def _write_bin(path, cloud_nx5):
    np.asarray(cloud_nx5, dtype=np.float32).tofile(path)


def test_load_lidar_bin_keeps_xyzi_only(tmp_path):
    p = str(tmp_path / "cloud.bin")
    cloud = np.array([[1, 2, 3, 0.5, 7], [4, 5, 6, 0.9, 3]], dtype=np.float32)
    _write_bin(p, cloud)
    pts = load_lidar_bin(p)
    assert pts.shape == (2, 4)
    np.testing.assert_allclose(pts, [[1, 2, 3, 0.5], [4, 5, 6, 0.9]])


def test_load_lidar_bin_rejects_bad_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_lidar_bin(str(tmp_path / "nope.bin"))
    bad = str(tmp_path / "bad.bin")
    np.zeros(7, dtype=np.float32).tofile(bad)  # not multiple of 5
    with pytest.raises(ValueError):
        load_lidar_bin(bad)


class _FakeNuScenes:
    def __init__(self, dataroot):
        self.dataroot = dataroot

    def get(self, table, token):
        if table == "sample":
            return {"token": "S1", "data": {"LIDAR_TOP": "SD1"}}
        assert table == "sample_data"
        return {"token": "SD1", "timestamp": 1234,
                "filename": "samples/LIDAR_TOP/fake.bin"}


def test_load_sample_lidar_info_contract(tmp_path):
    raw = str(tmp_path / "samples" / "LIDAR_TOP")
    os.makedirs(raw)
    cloud = np.zeros((10, 5), dtype=np.float32)
    cloud[:, 0] = np.arange(10)
    _write_bin(os.path.join(raw, "fake.bin"), cloud)
    pts, info = load_sample_lidar(_FakeNuScenes(str(tmp_path)), "S1")
    assert pts.shape == (10, 4)
    assert info == {"sample_token": "S1", "lidar_token": "SD1",
                    "timestamp": 1234.0,
                    "source_file": "samples/LIDAR_TOP/fake.bin"}


def _synthetic_raw(seed=0, n=1000):
    rng = np.random.default_rng(seed)
    pts = np.column_stack([rng.uniform(-30, 30, n), rng.uniform(-30, 30, n),
                           rng.normal(0, 1, n), rng.uniform(0, 1, n)])
    pts[0] = [np.nan, 0, 0, 0.5]   # non-finite
    pts[1] = [0, 0, 0, 0.3]        # zero range
    pts[2] = [500, 0, 0, 0.5]      # outside ROI
    return pts


def test_preprocess_counts_and_contract():
    clean, counts = preprocess_points(_synthetic_raw())
    assert counts["n_raw"] == 1000
    assert counts["n_nonfinite_removed"] == 1
    assert counts["n_zero_range_removed"] == 1
    assert counts["n_outside_roi_removed"] >= 1
    assert counts["n_processed"] == clean.shape[0] > 0
    assert clean.shape[1] == 4 and np.isfinite(clean).all()


def test_preprocess_rejects_empty_and_bad_shape():
    with pytest.raises(ValueError):
        preprocess_points(np.zeros((0, 4)))
    with pytest.raises(ValueError):
        preprocess_points(np.zeros((10, 3)))
    with pytest.raises(ValueError):
        preprocess_points(np.full((5, 4), np.nan))  # all removed


def test_save_processed_frame_files(tmp_path):
    clean, counts = preprocess_points(_synthetic_raw(n=200))
    meta = {"frame_id": "S1", "sample_token": "S1", "lidar_token": "SD1",
            "timestamp": 999.0, "source_file": "samples/LIDAR_TOP/f.bin"}
    npy_path, csv_path = save_processed_frame(clean, meta, counts, str(tmp_path))
    assert os.path.basename(npy_path) == "S1_LIDAR_TOP_xyzi.npy"
    back = np.load(npy_path)
    assert back.shape[1] == 4 and np.isfinite(back).all()
    df = pd.read_csv(csv_path)
    for col in ("frame_id", "sample_token", "lidar_token", "timestamp",
                "source_file", "n_raw", "n_processed"):
        assert col in df.columns
    assert df.iloc[0]["sample_token"] == "S1"
