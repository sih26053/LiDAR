"""Recorded frames + SQLite + replay-Jev tests (Phase 13).

Unit/store/route tests use a temp DB (store.configure) and mocked Jev
transport — no live API calls. The live integration test lives in
test_record_replay_live.py (small, keyed).
"""

import json

import pytest


@pytest.fixture()
def tmpdb(tmp_path, monkeypatch):
    from backend.services import store

    path = tmp_path / "test.db"
    store.configure(path)
    yield path
    store.close()
    from backend.services.store import DEFAULT_DB_PATH

    store.configure(DEFAULT_DB_PATH)


def _pts(n=50):
    import numpy as np

    rng = np.random.default_rng(7)
    pts = np.hstack([rng.uniform(-10, 10, (n, 3)),
                     np.full((n, 1), 0.5)])
    return np.ascontiguousarray(pts)


# 1-5. Recorder files -----------------------------------------------------

def test_npy_write_and_reload(tmp_path):
    import numpy as np

    from backend.services import frame_recorder as R

    monkeypatch_dir = tmp_path / "rec"
    import backend.services.frame_recorder as Rm
    old = Rm.RECORDINGS_ROOT
    Rm.RECORDINGS_ROOT = monkeypatch_dir
    try:
        rec = R.record_frame("RUN_X", 3, "live-00003", "straight_road", 1.5,
                             _pts(), {"map_cells": []}, [0.5] * 13,
                             {"source": "jev"}, {"verdict": "SAFE_TO_EXECUTE"},
                             {"executed_action": "stop"}, {"x": 1.0})
        p = monkeypatch_dir / "RUN_X" / rec["points_path"]
        assert p.is_file()
        back = np.load(p)
        assert back.shape == (50, 4) and back.dtype == np.float64
        assert np.array_equal(back, _pts())
        meta = json.loads((monkeypatch_dir / "RUN_X" / rec["metadata_path"]).read_text())
        assert meta["frame_id"] == "RUN_X_f00003"
        assert meta["run_id"] == "RUN_X" and meta["scenario"] == "straight_road"
        assert meta["lidar"]["point_count"] == 50
    finally:
        Rm.RECORDINGS_ROOT = old


def test_recorder_rejects_bad_points():
    import numpy as np

    from backend.services.frame_recorder import RecorderError, validate_points

    with pytest.raises(RecorderError):
        validate_points(np.zeros((10, 3)))
    with pytest.raises(RecorderError):
        validate_points(np.full((5, 4), np.nan))
    with pytest.raises(RecorderError):
        validate_points(np.zeros((0, 4)))


def test_frame_uniqueness(tmp_path):
    from backend.services import frame_recorder as R
    import backend.services.frame_recorder as Rm

    old = Rm.RECORDINGS_ROOT
    Rm.RECORDINGS_ROOT = tmp_path
    try:
        a = R.record_frame("RUN_X", 0, "live-00000", None, 0.0, _pts(10),
                           {"map_cells": []}, [0.5] * 13, {}, {}, {}, {})
        b = R.record_frame("RUN_X", 1, "live-00001", None, 0.0, _pts(10),
                           {"map_cells": []}, [0.5] * 13, {}, {}, {}, {})
        assert a["frame_id"] != b["frame_id"]
    finally:
        Rm.RECORDINGS_ROOT = old


# 6-16. Store --------------------------------------------------------------

def test_schema_and_crud(tmpdb):
    from backend.services import store

    store.create_run("RUN_A", "autonomous", scenario="straight_road")
    store.create_frame("RUN_A_f00000", "RUN_A", 1.0, 0, "straight_road",
                       "p.npy", 10, [0.1] * 13, {"x": 0.0}, {"t": 1},
                       "m.json")
    store.create_map_cells("RUN_A_f00000", [
        {"x": 1.0, "y": 2.0, "occupancy": 1.0, "weird": "kept"}])
    store.record_decision("RUN_A_f00000", "jev", model="m",
                          proposed_action="forward", confidence=0.9,
                          probabilities={"forward": 0.9}, latency_ms=5.0)
    store.record_safety_event("RUN_A_f00000", "SAFE_TO_EXECUTE")
    store.record_execution("RUN_A_f00000", "jev", "forward", dx=0.1)
    assert len(store.list_frames(run_id="RUN_A")) == 1
    assert store.get_map_cells("RUN_A_f00000")[0]["weird"] == "kept"
    assert store.get_decision("RUN_A_f00000")["proposed_action"] == "forward"
    assert store.get_frame_result("RUN_A_f00000")["map_cell_count"] == 1
    assert store.list_runs()[0]["frame_count"] == 1


def test_db_reopen(tmpdb):
    from backend.services import store

    store.create_run("RUN_B", "manual")
    store.create_frame("RUN_B_f00000", "RUN_B", 2.0, 0, None, "p.npy",
                       5, [0.2] * 13, {}, {}, "m.json")
    store.close()
    # fresh connection, same file: data survives restart
    assert store.get_frame("RUN_B_f00000")["run_id"] == "RUN_B"
    assert store.get_run("RUN_B")["mode"] == "manual"


def test_replay_rows_and_history(tmpdb):
    from backend.services import store

    store.create_run("RUN_C", "autonomous")
    store.create_frame("RUN_C_f00000", "RUN_C", 3.0, 0, None, "p.npy",
                       5, [0.3] * 13, {}, {}, "m.json")
    store.record_decision("RUN_C_f00000", "jev", proposed_action="stop",
                          confidence=0.6)
    store.record_decision("RUN_C_f00000", "replay-jev", proposed_action="forward",
                          confidence=0.8, replay_run_id="replay-1")
    store.record_decision("RUN_C_f00000", "replay-jev", proposed_action="stop",
                          confidence=0.5, replay_run_id="replay-2")
    live = store.get_decision("RUN_C_f00000", source="jev")
    assert live["proposed_action"] == "stop"  # original immutable
    hist = store.get_replay_history("RUN_C_f00000")
    assert [h["replay_run_id"] for h in hist] == ["replay-1", "replay-2"]
    assert store.get_decision("RUN_C_f00000",
                              replay_run_id="replay-2")["confidence"] == 0.5


# 17-19. Replay path purity -------------------------------------------------

def _stage_npy(run_tag: str, pts):
    """Stage points into the REAL default recordings dir (path resolution
    under test); caller must rmtree the run dir afterwards."""
    import shutil

    import numpy as np

    from backend.services.store import PROJECT_ROOT

    d = PROJECT_ROOT / "results" / "recordings" / run_tag / "points"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "p.npy", pts)
    return f"results/recordings/{run_tag}/points/p.npy", d.parent


def test_replay_never_touches_live_vehicle(tmpdb, tmp_path, monkeypatch):
    import numpy as np
    import shutil

    from backend.services import pybullet_live as live
    from backend.services import replay_jev as rj
    from backend.services import store

    rel, rundir = _stage_npy("_test_RUN_D", _pts(120))
    try:
        store.create_run("_test_RUN_D", "autonomous")
        store.create_frame("_test_RUN_D_f00000", "_test_RUN_D", 4.0, 0, None,
                           rel, 120, [0.5] * 13, {}, {}, "m.json")
        sentinel = live._live.get("env")
        live._live["env"] = object()  # replay must never touch this
        try:
            import urllib.request

            class FakeResp:
                def __enter__(self): return self
                def __exit__(self, *a): return False
                def read(self):
                    return json.dumps({
                        "model": "typesafe/jev-1.13",
                        "answers": {"action": {"choice": "forward", "confidence": 0.8,
                                                       "probabilities": {"forward": 0.8}}}}).encode()

            monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: FakeResp())
            rec = rj.run_replay_decision("_test_RUN_D_f00000", "replay-t1",
                                         engine="jev")
            assert rec["source"] == "replay-jev"
            assert rec["action"] == "forward" and rec["stored"] is True
            assert live._live.get("env") is not None  # untouched sentinel
        finally:
            live._live["env"] = sentinel
        row = store.get_decision("_test_RUN_D_f00000", source="replay-jev")
        assert row["replay_run_id"] == "replay-t1"
    finally:
        shutil.rmtree(rundir, ignore_errors=True)


# 20-22. Failures ------------------------------------------------------------

def test_invalid_frame_errors():
    from fastapi.testclient import TestClient

    from backend.app import app
    from backend.services.replay_service import FrameNotFoundError

    client = TestClient(app)
    r = client.post("/replay/jev-decide", json={"frame_id": "NOPE-NOT-A-FRAME"})
    assert r.status_code == 404
    assert r.json()["error_code"] == "FRAME_NOT_FOUND"
    try:
        from backend.services import replay_jev as rj

        rj.load_recorded_points("NOPE-NOT-A-FRAME")
        raise AssertionError("should have raised")
    except FrameNotFoundError:
        pass


def test_jev_failure_recorded_not_fabricated(tmpdb, tmp_path, monkeypatch):
    import shutil
    import urllib.request

    from backend.services import replay_jev as rj
    from backend.services import store

    rel, rundir = _stage_npy("_test_RUN_E", _pts(120))
    try:
        store.create_run("_test_RUN_E", "autonomous")
        store.create_frame("_test_RUN_E_f00000", "_test_RUN_E", 5.0, 0, None,
                           rel, 120, [0.5] * 13, {}, {}, "m.json")

        def boom(*a, **k):
            raise TimeoutError("slow")

        monkeypatch.setattr(urllib.request, "urlopen", boom)
        rec = rj.run_replay_decision("_test_RUN_E_f00000", "replay-t2",
                                     engine="jev")
        assert rec["action"] is None and rec["error"] is not None
        assert rec["engine_status"] in ("FAILED", "UNAVAILABLE", "INVALID")
        row = store.get_decision("_test_RUN_E_f00000", source="replay-jev")
        assert row["proposed_action"] is None and row["error"] is not None
    finally:
        shutil.rmtree(rundir, ignore_errors=True)


def test_db_failure_degrades_gracefully(tmp_path):
    from backend.services import frame_recorder as R
    import backend.services.frame_recorder as Rm
    from backend.services import store

    bad = tmp_path / "afile"
    bad.write_text("x")
    with pytest.raises(Exception):
        store.configure(bad / "t.db")
        store.create_run("RUN_F", "autonomous")
    store.close()
    from backend.services.store import DEFAULT_DB_PATH

    store.configure(DEFAULT_DB_PATH)
    # flat-file recorder independent of DB: still works
    old = Rm.RECORDINGS_ROOT
    Rm.RECORDINGS_ROOT = tmp_path / "rec2"
    try:
        rec = R.record_frame("RUN_F", 0, "live-0", None, 0.0, _pts(10),
                             {"map_cells": []}, [0.5] * 13, {}, {}, {}, {})
        assert (tmp_path / "rec2" / "RUN_F" / rec["points_path"]).is_file()
    finally:
        Rm.RECORDINGS_ROOT = old


# 23-26. Backward compat ------------------------------------------------------

def test_decision_store_api_unchanged():
    from backend.services import decision_store

    decision_store.clear()
    decision_store.record({"frame_id": "z", "source": "jev"})
    assert decision_store.current()["frame_id"] == "z"
    assert decision_store.count() == 1
    assert decision_store.history(5)[0]["source"] == "jev"
    decision_store.clear()


def test_frames_lists_both_origins(tmpdb):
    from fastapi.testclient import TestClient

    from backend.app import app
    from backend.services import store

    store.create_run("RUN_G", "manual")
    store.create_frame("RUN_G_f00000", "RUN_G", 6.0, 0, "straight_road",
                       "p.npy", 5, [0.5] * 13, {}, {}, "m.json")
    body = TestClient(app).get("/frames").json()
    origins = {f["origin"] for f in body["frames"]}
    assert "nuscenes" in origins and "live-recorded" in origins
    rec = [f for f in body["frames"] if f["origin"] == "live-recorded"][0]
    assert rec["run_id"] == "RUN_G" and rec["scenario"] == "straight_road"
    assert body["live_recorded"] >= 1 and body["nuscenes"] >= 1


def test_results_for_recorded_frame(tmpdb):
    from fastapi.testclient import TestClient

    from backend.app import app
    from backend.services import result_service, store

    store.create_run("RUN_H", "manual")
    store.create_frame("RUN_H_f00000", "RUN_H", 7.0, 0, None, "p.npy",
                       5, [0.5] * 13, {}, {}, "m.json")
    result_service.store_result("RUN_H_f00000", {"frame_id": "RUN_H_f00000"}, status="success")
    r = TestClient(app).get("/results/RUN_H_f00000")
    assert r.status_code == 200 and r.json()["result"]["frame_id"] == "RUN_H_f00000"
    result_service.clear()


def test_replay_load_recorded_fallback(tmpdb):
    """POST /replay/load must serve recorded frames (not just /replay/run)."""
    import shutil

    from fastapi.testclient import TestClient

    from backend.app import app
    from backend.services import store

    rel, rundir = _stage_npy("_test_LOAD", _pts(120))
    try:
        store.create_run("_test_LOAD", "autonomous")
        store.create_frame("_test_LOAD_f00000", "_test_LOAD", 2.5, 0, None,
                           rel, 120, [0.5] * 13, {}, {}, "m.json")
        body = TestClient(app).post(
            "/replay/load", json={"frame_id": "_test_LOAD_f00000"}).json()
        assert body["load_status"] == "loaded"
        assert body["origin"] == "live-recorded"
        assert body["point_count"] == 120
    finally:
        shutil.rmtree(rundir, ignore_errors=True)


def test_nuscenes_replay_untouched():
    from backend.services import replay_service

    frames = replay_service.list_frames()
    assert len(frames) >= 1 and all("frame_id" in f for f in frames)
