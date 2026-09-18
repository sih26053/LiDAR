"""Replay tests: real manifest + real bundled frame, no fabricated IDs."""

from fastapi.testclient import TestClient

from backend.app import app
from backend.services import replay_service

client = TestClient(app)
REAL_FRAME = "5991fad3280c4f84b331536c32001a04"


def test_frames_lists_real_manifest():
    frames = replay_service.list_frames()
    assert len(frames) == 7
    assert REAL_FRAME in {f["frame_id"] for f in frames}


def test_frames_endpoint():
    r = client.get("/frames")
    assert r.status_code == 200
    assert r.json()["count"] == 7


def test_replay_load_real_frame():
    r = client.post("/replay/load", json={"frame_id": REAL_FRAME})
    assert r.status_code == 200
    body = r.json()
    assert body["frame_id"] == REAL_FRAME
    assert body["timestamp"] == 1535385092150099.0
    assert body["point_count"] > 0
    assert body["load_status"] == "loaded"


def test_replay_load_unknown_frame_404():
    r = client.post("/replay/load", json={"frame_id": "does_not_exist"})
    assert r.status_code == 404
    assert r.json()["error_code"] == "FRAME_NOT_FOUND"
