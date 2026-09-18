"""Health + config endpoint tests (no ML execution)."""

from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "paradox-protocol-backend"


def test_config_loads_frozen_values():
    r = client.get("/config")
    assert r.status_code == 200
    body = r.json()
    assert body["final_version"] == "prototype-final-v1"
    assert body["max_mapping_distance_m"] == 100.0
    assert body["integration_cell_size_m"] == 2.0
    assert "annotation" in body["semantic_source_mode"]
    levels = [tuple(x) for x in body["resolution_levels"]]
    assert (0.7, 0.05) in levels
