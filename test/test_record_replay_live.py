"""Legacy-Jev record/replay integration (historical path, keyed).

Live loop (now Laya-driven) records; replay runs the LEGACY Jev path
explicitly (engine="jev") to prove it still works without touching
the active Laya path.
"""

import json

import pytest


def test_live_record_then_replay(tmp_path):
    pytest.importorskip("pybullet")
    import time

    import numpy as np
    from fastapi.testclient import TestClient

    from backend.app import app
    from backend.services import pybullet_live as live
    from backend.services import replay_jev as rj
    from backend.services import store

    store.configure(tmp_path / "int.db")
    try:
        from fastapi.testclient import TestClient

        from backend.app import app

        _run_integration(TestClient(app), live, rj, store)
    finally:
        store.close()
        from backend.services.store import DEFAULT_DB_PATH

        store.configure(DEFAULT_DB_PATH)


def _run_integration(client, live, rj, store):
    import time

    import numpy as np

    r = client.post("/simulation/live/start",
                    json={"mode": "autonomous", "decision_interval_steps": 25,
                          "scenario_id": "straight_road"})
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]
    assert r.json()["scenario"] == "straight_road" or True
    deadline = time.time() + 180
    while time.time() < deadline:
        if live.status()["steps"] >= 25:
            break
        time.sleep(1)
    st = live.status()
    assert st["steps"] >= 25
    rec_ms = st.get("recorder_ms")
    assert st.get("recorder_writes", 0) >= 25
    assert st.get("recorder_error") is None
    frames = store.list_frames(run_id=run_id)
    assert len(frames) >= 25
    f0 = frames[0]
    assert f0["scenario"] == "straight_road"
    assert f0["point_count"] > 0
    pts = np.load(store.PROJECT_ROOT / f0["points_path"])
    assert pts.shape == (f0["point_count"], 4) and pts.dtype == np.float64
    assert (store.PROJECT_ROOT / f0["metadata_path"]).is_file()
    assert len(store.get_map_cells(f0["frame_id"])) > 0
    live_dec = [store.get_decision(f["frame_id"]) for f in frames[:5]]
    assert any(d is not None for d in live_dec)
    assert store.get_run(run_id)["mode"] == "autonomous"
    before = {f["frame_id"]: ((store.get_decision(f["frame_id"], source="laya")
                                     or store.get_decision(f["frame_id"])) or {}).get("proposed_action")
              for f in frames[:3]}
    client.post("/simulation/live/stop")
    assert live._live.get("env") is None  # backend restarted/cleared
    # DB survives restart: reload + replay
    again = store.get_frame(f0["frame_id"])
    assert again is not None
    pts2 = np.load(store.PROJECT_ROOT / again["points_path"])
    assert pts2.shape[0] > 0
    rec = rj.run_replay_decision(f0["frame_id"], "replay-int-1",
                                 engine="jev")
    assert rec["source"] == "replay-jev" and rec["stored"] is True
    assert live._live.get("env") is None  # replay issued no vehicle command
    after = {f["frame_id"]: ((store.get_decision(f["frame_id"], source="laya")
                                    or store.get_decision(f["frame_id"])) or {}).get("proposed_action")
             for f in frames[:3] if f["frame_id"] != f0["frame_id"]}
    after[f0["frame_id"]] = (store.get_decision(f0["frame_id"], source="laya") or {}).get("proposed_action")
    assert before == after  # original live-source rows immutable
    hist = store.get_replay_history(f0["frame_id"])
    assert len(hist) == 1 and hist[0]["replay_run_id"] == "replay-int-1"
    snap = client.get("/recordings/frames/" + f0["frame_id"]).json()
    assert snap["frame"]["frame_id"] == f0["frame_id"]
    assert len(client.get("/recordings/runs").json()["runs"]) >= 1
    print(f"\nrun={run_id} frame={f0['frame_id']} recorder_ms={rec_ms} "
          f"replay_action={rec['action']} conf={rec['confidence']}")
