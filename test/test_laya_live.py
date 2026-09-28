"""Live PyBullet + Laya integration (Phases 24-26). UNMOCKED.

30 autonomous steps on straight_road (6 real Laya calls), then a
real replay-laya call on a recorded frame, then dashboard-bundle
assertions. Proves Laya -> Safety -> Executor -> PyBullet end to end.
"""

import json

import pytest


@pytest.fixture(scope="module")
def laya_server():
    pytest.importorskip("laya")
    from backend.services import laya_manager

    info = laya_manager.ensure_started(wait_s=600.0)
    assert info["health"].get("healthy")
    return info


def test_24_live_loop_laya_end_to_end(laya_server):
    pytest.importorskip("pybullet")
    import time

    from fastapi.testclient import TestClient

    from backend.app import app
    from backend.services import pybullet_live as live
    from backend.services import store

    client = TestClient(app)
    r = client.post("/simulation/live/start",
                    json={"mode": "autonomous", "decision_interval_steps": 5,
                          "scenario_id": "straight_road"})
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]
    deadline = time.time() + 600
    while time.time() < deadline:
        if live.status()["steps"] >= 30:
            break
        time.sleep(2)
    st = live.status()
    assert st["steps"] >= 30
    assert st["laya_calls"] >= 6, st
    assert st["laya_successful"] >= 1, st  # real typed decisions returned
    snap = client.get("/simulation/state").json()
    assert snap["simulator"] == "pybullet"
    assert "laya" in str(snap["decision"]["model"]).lower()
    assert snap["decision"]["backend"] == "LOCAL"
    assert snap["decision"]["device"] in ("CPU", "CUDA", "UNKNOWN")
    assert snap["source"] in ("laya", "fallback")
    assert snap["vehicle"]["x"] is not None
    assert snap["lidar_points"] > 0 and snap["map"]["cell_count"] > 0
    assert len(snap["map"]["cells"]) > 0  # real live map in snapshot
    cur = client.get("/decision/current").json()
    assert cur["decision_model"] in ("laya", "fallback", "manual")
    assert st["recorder_writes"] >= 30 and st["recorder_error"] is None
    assert store.get_run(run_id)["decision_model"] is not None
    assert "laya" in (store.get_run(run_id)["decision_model"] or "").lower() or \
        "typed-decisions" in (store.get_run(run_id)["decision_model"] or "").lower()
    n_laya = 0
    for f in store.list_frames(run_id=run_id, limit=40):
        d = store.get_decision(f["frame_id"], source="laya")
        if d is not None:
            n_laya += 1
    assert n_laya >= 1
    print(f"\nrun={run_id} steps={st['steps']} laya_ok={st['laya_successful']} "
          f"dist={st['distance_m']} model={snap['decision']['model']}")
    client.post("/simulation/live/stop")
    # leave run_id for the replay test via DB latest
    return run_id


def test_25_replay_laya(laya_server):
    pytest.importorskip("pybullet")
    from backend.services import pybullet_live as live
    from backend.services import replay_laya as rl
    from backend.services import store

    assert live._live.get("env") is None
    runs = []
    for r in store.list_runs(limit=20):
        for f in store.list_frames(run_id=r["run_id"], limit=3):
            if store.get_decision(f["frame_id"], source="laya") is not None:
                runs.append(r)
                break
    assert runs, "no laya run recorded"
    run_id = runs[0]["run_id"]
    frames = store.list_frames(run_id=run_id, limit=5)
    assert frames
    fid = frames[0]["frame_id"]
    before = store.get_decision(fid, source="laya")
    rec = rl.run_replay_decision(fid, "replay-laya-int-1")
    assert rec["source"] == "replay-laya" and rec["stored"] is True
    assert rec["action"] in ("forward", "turn_left", "turn_right", "stop", None)
    assert live._live.get("env") is None  # replay moved no vehicle
    after = store.get_decision(fid, source="laya")
    assert (before or {}).get("id") == (after or {}).get("id")  # immutable
    hist = rl.get_replay_history(fid)
    assert len(hist) >= 1 and hist[-1]["replay_run_id"] == "replay-laya-int-1"
    print(f"\nreplay frame={fid} action={rec['action']} conf={rec['confidence']} "
          f"latency={rec['latency_ms']}")


def test_26_dashboard_shows_laya():
    from pathlib import Path

    bundle = None
    for js in (Path("frontend/dist/assets")).glob("*.js"):
        text = js.read_text(encoding="utf-8", errors="replace")
        if "Live Control" in text:
            bundle = text
            break
    assert bundle is not None, "built dashboard bundle missing"
    assert "Live Control — PyBullet + Laya" in bundle
    assert "layaStatusOf" in bundle or "MANUAL (not Laya)" in bundle
    assert "Jev 1.13 (typesafe/jev-1.13)" not in bundle
    assert "TYPESAFE_API_KEY not set" not in bundle
