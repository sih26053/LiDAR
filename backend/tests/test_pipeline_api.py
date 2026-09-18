"""Pipeline API tests: one real-data integration run + contract checks."""

import pytest
from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)
REAL_FRAME = "5991fad3280c4f84b331536c32001a04"


def _check_contract(result: dict):
    for key in ("frame_id", "timestamp", "map_cells", "importance",
                "resolution", "semantic", "timing"):
        assert key in result, f"missing key {key}"
    assert result["frame_id"] == REAL_FRAME
    assert result["timestamp"] == 1535385092150099.0
    assert result["map_cell_count"] == len(result["map_cells"]) > 0
    cell = result["map_cells"][0]
    for key in ("x", "y", "elevation", "resolution", "importance",
                "semantic_class", "semantic_source"):
        assert key in cell, f"cell missing {key}"
    assert cell["resolution"] in (0.05, 0.10, 0.20, 0.50)
    assert 0.0 <= cell["importance"] <= 1.0
    # Semantic provenance must use the annotation vocabulary, never fake ML.
    assert cell["semantic_source"] in ("annotation", "fallback", "lidarseg",
                                       "lidarseg_annotation", "object_annotation",
                                       "unknown", "model")
    assert "model" not in result["semantic"]["mode"].lower() or "no trained model" in result["semantic"]["mode"]
    assert result["timing"]["total_latency_ms"] is not None
    assert result["timing"]["total_latency_ms"] > 0


@pytest.mark.integration
def test_replay_run_real_frame():
    r = client.post("/replay/run", json={"frame_id": REAL_FRAME})
    assert r.status_code == 200, r.text[:500]
    _check_contract(r.json()["result"])


@pytest.mark.integration
def test_results_and_metrics_after_run():
    client.post("/replay/run", json={"frame_id": REAL_FRAME})
    r = client.get(f"/results/{REAL_FRAME}")
    assert r.status_code == 200
    _check_contract(r.json()["result"])
    m = client.get(f"/metrics/{REAL_FRAME}")
    assert m.status_code == 200
    body = m.json()
    assert body["map_cell_count"] > 0
    assert body["latency_ms"] is not None and body["latency_ms"] > 0


def test_results_missing_frame_404():
    r = client.get("/results/definitely_not_a_frame")
    assert r.status_code == 404
    assert r.json()["error_code"] == "RESULT_NOT_FOUND"


def test_demo_status_shape():
    r = client.get("/demo/status")
    assert r.status_code == 200
    body = r.json()
    for key in ("backend", "configuration_loaded", "replay_available",
                "available_frames", "last_frame_id", "last_status"):
        assert key in body
