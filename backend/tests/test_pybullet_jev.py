"""PyBullet + Jev track: live-dep refusals honest, pure logic verified."""

import pytest
from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


def test_simulator_reports_live_truthfully(monkeypatch):
    try:
        __import__("pybullet")
        live = True
    except ImportError:
        live = False
    r = client.get("/simulation/pybullet")
    assert r.status_code == 200
    body = r.json()
    assert body["pybullet_importable"] is live
    if live:
        assert "active" in body["simulator"]
    else:
        assert "BLOCKED" in body["simulator"]
    assert body["decision_engine"] == "laya"
    assert "laya" in body  # local engine block (availability varies)


def test_simulation_run_live_or_blocked():
    try:
        __import__("pybullet")
        live = True
    except ImportError:
        live = False
    r = client.post("/simulation/run", json={"steps": 2})
    assert r.status_code == 200
    body = r.json()
    if live:
        assert body["status"] == "COMPLETED"
        assert body["n_steps_recorded"] == 2
    else:
        assert body["status"] == "BLOCKED"
        assert body["n_steps_recorded"] == 0
        assert "reason" in body


def test_decision_endpoints_empty_honestly():
    from backend.services import decision_store

    decision_store.clear()
    assert client.get("/decision/current").status_code == 404
    h = client.get("/decision/history").json()
    assert h["count"] == 0 and h["records"] == []
    m = client.get("/metrics/current").json()
    assert m["decisions_recorded"] == 0


def test_decision_record_roundtrip():
    from backend.services import decision_store

    decision_store.clear()
    decision_store.record({"frame_id": "f1", "decision_model": "jev",
                           "proposed_action": "stop", "confidence": 0.9,
                           "safety_status": "SAFE_TO_EXECUTE",
                           "executed_action": "stop",
                           "decision_latency_ms": 12.5})
    cur = client.get("/decision/current").json()
    assert cur["frame_id"] == "f1" and cur["confidence"] == 0.9
    decision_store.clear()


def test_jev_adapter_and_policy_without_service(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("JEV_ENDPOINT", raising=False)
    from src.decision import jev_client as C
    # Hermetic regardless of any real key in config/local_secrets.py.
    monkeypatch.setattr(C, "resolve_api_key", lambda: ("", ""))
    from src.decision import jev_decision as J
    from src.decision import policy_interface as P
    from src.decision.jev_state_adapter import FIELD_NAMES, to_named_state

    assert len(FIELD_NAMES) == 13
    named = to_named_state([0.5] * 13, [15.0] * 8)
    rec = P.decide_with_safety(P.JevPolicy(), named, 40.0, True, False)
    assert rec["proposed_action"] is None
    assert rec["executed_action"] == "stop"
    assert rec["safety_override"] is False  # fallback stop passes cleanly
    assert J.service_status()["available"] is False
    with pytest.raises(ValueError):
        to_named_state([0.5] * 12)
