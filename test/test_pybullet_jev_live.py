"""PyBullet + Jev integration tests (GAP 12-18, 22).

Live tests run on .venv-pb (Python 3.11 + pybullet); pure/transport tests
run anywhere. Nothing is fabricated: UNAVAILABLE paths are asserted as
UNAVAILABLE, never as success.
"""

import json

import pytest


# 1. Local secret configuration -------------------------------------------

def test_secret_example_exists_and_placeholder_is_missing():
    from pathlib import Path

    from src.decision import jev_client as C

    assert Path("config/local_secrets.example.py").is_file()
    assert C.PLACEHOLDER_KEY == "PASTE_YOUR_OPENROUTER_API_KEY_HERE"
    assert C.resolve_api_key.__doc__ is not None


def test_placeholder_key_counts_as_missing(monkeypatch):
    import config.local_secrets as S

    from src.decision import jev_client as C

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(S, "OPENROUTER_API_KEY", "PASTE_YOUR_OPENROUTER_API_KEY_HERE")
    key, source = C.resolve_api_key()
    assert key == "" and source == ""


def test_env_key_resolves_without_exposure(monkeypatch):
    from src.decision import jev_client as C

    monkeypatch.setenv("OPENROUTER_API_KEY", "sentinel-key-abc-123")
    key, source = C.resolve_api_key()
    assert key == "sentinel-key-abc-123" and source == "OPENROUTER_API_KEY"
    st = C.service_status()
    assert "sentinel-key-abc-123" not in json.dumps(st)


def test_endpoint_defaults_to_decisions_api(monkeypatch):
    from src.decision import jev_client as C

    monkeypatch.delenv("JEV_ENDPOINT", raising=False)
    ep, src = C.resolve_endpoint()
    assert ep == "https://openrouter.ai/api/alpha/decisions" and src == "default"
    assert "chat/completions" not in ep
    monkeypatch.setenv("JEV_ENDPOINT", "http://127.0.0.1:9/x")
    assert C.resolve_endpoint()[0] == "http://127.0.0.1:9/x"


# 7/8. Jev request construction + response parsing -------------------------

def _ok_body():
    return json.dumps({
        "model": "typesafe/jev-1.13",
        "answers": {"action": {"choice": "left", "confidence": 0.81,
                               "probabilities": {"forward": 0.05, "left": 0.81,
                                                 "right": 0.09, "stop": 0.05}}},
    }).encode()


def test_jev_request_shape_and_ok_parse(monkeypatch):
    import urllib.request

    from src.decision import jev_client as C

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    monkeypatch.delenv("JEV_ENDPOINT", raising=False)
    seen = {}

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return _ok_body()

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode())
        seen["headers"] = dict(req.header_items())
        return FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    rec = C.JevClient().decide({"sector_0_range": 0.5}, frame_id="f-1")
    assert seen["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert seen["body"]["model"] == "typesafe/jev-1.13"
    assert set(seen["body"]) == {"model", "state", "questions"}
    assert set(seen["body"]["questions"]["action"]) == {"type", "instructions", "criteria"}
    assert "test-key-not-real" not in json.dumps(seen["body"])
    assert rec.status == "OK" and rec.proposed_action == "turn_left"
    assert rec.confidence == 0.81 and rec.latency_ms >= 0
    assert rec.frame_id == "f-1" and rec.error is None


def test_jev_invalid_and_failed_paths(monkeypatch):
    import urllib.request

    from src.decision import jev_client as C

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    monkeypatch.setenv("JEV_ENDPOINT", "http://127.0.0.1:9/nope")

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"answers": {"action": {"choice": "fly"}}}'

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: FakeResp())
    rec = C.JevClient().decide({"sector_0_range": 0.5})
    assert rec.status == "INVALID" and rec.proposed_action is None

    def boom(*a, **k):
        raise ConnectionRefusedError("nope")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    rec = C.JevClient().decide({"sector_0_range": 0.5})
    assert rec.status == "FAILED" and "test-key-not-real" not in (rec.error or "")


# 9/10. Confidence gate + safety -------------------------------------------

def _named_state():
    n = {f"sector_{i}_range": 1.0 for i in range(8)}
    n.update({"obstacle_density": 0.0, "moving_share": 0.0,
              "static_share": 0.0, "mean_importance": 0.1,
              "mean_uncertainty": 0.1})
    return n


def test_confidence_gate_falls_back(monkeypatch):
    import urllib.request

    from src.decision.policy_interface import decide_with_safety, get_policy

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    monkeypatch.delenv("JEV_ENDPOINT", raising=False)

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return _ok_body()

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: FakeResp())
    policy = get_policy("jev")
    policy.min_confidence = 0.99  # above the 0.81 response: gate must fail
    rec = decide_with_safety(policy, _named_state(), 40.0, True, False,
                             frame_id="g-1", run_id="run-test")
    assert rec["proposed_action"] == "turn_left"
    assert rec["executed_action"] == "stop"
    assert rec["source"] == "fallback" and rec["run_id"] == "run-test"


def test_safety_overrides_blocked_forward():
    from src.safety_controller import evaluate

    v = evaluate("forward", 2.0, map_valid=True, emergency=False)
    assert v["final_action"] == "stop" and v["verdict"] == "OVERRIDE_TO_STOP"
    v = evaluate("forward", 40.0, map_valid=True, emergency=False)
    assert v["final_action"] == "forward" and v["verdict"] == "SAFE_TO_EXECUTE"
    v = evaluate("stop", 40.0, map_valid=True, emergency=False)
    assert v["final_action"] == "stop"


# 2/3/4/11. Live simulator: lidar, frame, executor --------------------------

def test_live_lidar_frame_and_executor():
    pytest.importorskip("pybullet")
    from src.simulation.pybullet_action_executor import PyBulletActionExecutor
    from src.simulation.pybullet_env import PyBulletEnv
    from src.simulation.pybullet_lidar import PyBulletLidar, to_lidar_frame

    env = PyBulletEnv()
    env.connect()
    env.reset(seed=42)
    try:
        lidar = PyBulletLidar()
        scan = lidar.scan(env)
        pts = lidar.to_points(scan)
        assert pts.shape[1] == 4 and pts.shape[0] > 0
        frame = to_lidar_frame(pts, "live-test-00001", 0.0)
        assert frame.source == "pybullet_sim"
        before = env.get_vehicle_state()
        out = PyBulletActionExecutor(env).execute("forward")
        after = env.get_vehicle_state()
        assert out["action"] == "forward" and out["executed"] is True
        assert "pose_before" in out and "pose_after" in out
        assert (after["x"], after["y"]) != (before["x"], before["y"])
        assert before["x"] == pytest.approx(out["pose_before"]["x"])
    finally:
        env.close()


# 5/6. Stages 1-6 + 13-D contract on a live frame --------------------------

def test_live_stages_1_to_6_and_state():
    pytest.importorskip("pybullet")
    from src.decision.jev_state_adapter import FIELD_NAMES, to_named_state
    from src.simulation.pybullet_env import PyBulletEnv
    from src.simulation.pybullet_lidar import PyBulletLidar, to_lidar_frame
    from src.simulation.stages_runner import run_stages_1_to_6

    env = PyBulletEnv()
    env.connect()
    env.reset(seed=42)
    try:
        lidar = PyBulletLidar()
        pts = lidar.to_points(lidar.scan(env))
        frame = to_lidar_frame(pts, "live-test-00002", 0.0)
        out = run_stages_1_to_6(frame.points, frame.frame_id, frame.timestamp)
        vec = out["state"]["state_vector"]
        assert len(vec) == 13 and all(0.0 <= v <= 1.0 for v in vec)
        assert len(FIELD_NAMES) == 13
        named = to_named_state(vec, out["state"]["sector_ranges_m"])
        assert named["state_dim"] == 13
    finally:
        env.close()


# 12. Backend live APIs + WS ------------------------------------------------

def test_backend_live_endpoints_and_ws():
    pytest.importorskip("pybullet")
    import time

    from fastapi.testclient import TestClient

    from backend.app import app
    from backend.services import pybullet_live

    client = TestClient(app)
    try:
        assert client.get("/system/status").status_code == 200
        r = client.post("/simulation/live/start",
                        json={"mode": "manual",
                              "decision_interval_steps": 5})
        assert r.status_code == 200 and r.json()["mode"] == "manual"
        deadline = time.time() + 60
        snap = None
        while time.time() < deadline:
            s = client.get("/simulation/state")
            if s.status_code == 200:
                snap = s.json()
                break
            time.sleep(1)
        assert snap is not None and snap["simulator"] == "pybullet"
        assert snap["lidar_points"] > 0 and len(snap["rl_state"]) == 13
        blob = json.dumps(snap)
        assert "Bearer" not in blob and "OPENROUTER_API_KEY" not in blob
        with client.websocket_connect("/ws/live") as ws:
            msg = ws.receive_json()
            assert "frame_id" in msg and "system_status" in msg
            assert "OPENROUTER_API_KEY" not in msg
        mm = client.post("/simulation/action/stop").json()
        assert mm["source"] == "manual" and mm["executed_action"] == "stop"
        assert client.get("/decision/history").json()["count"] >= 1
        assert client.get("/metrics/current").status_code == 200
    finally:
        client.post("/simulation/live/stop")
        pybullet_live.stop()


# 15. Closed loop chain (1 step, real stages) --------------------------------

def test_closed_loop_single_step_chain():
    pytest.importorskip("pybullet")
    from scripts.run_pybullet_closed_loop import main

    trace = main(steps=1, decision_interval_steps=1, run_id="run-test-1")
    assert trace["status"] == "COMPLETED"
    s = trace["steps"][0]
    assert s["points"] > 0 and s["map_cells"] > 0
    assert len(s["rl_state"]) == 13
    assert s["decision"]["source"] in ("jev", "fallback")
    assert s["loop_latency_ms"] >= 0 and s["safety_latency_ms"] >= 0


# Phase 2/3. Simulator-boundary representation ------------------------------

def test_lidar_ring_attribution_ground_wall_obstacle():
    """Ring-by-ring return census (documents what each surface yields).

    Regression guard: -6-deg ring paints walls (~8 m) + ground (~17 m);
    0-deg ring detects in-lane obstacles at true range and flies over
    the 1 m walls; +6-deg ring sees sky (dropped, never invented).
    """
    import math

    pytest.importorskip("pybullet")
    from src.simulation.pybullet_env import PyBulletEnv
    from src.simulation.pybullet_lidar import PyBulletLidar

    env = PyBulletEnv()
    env.connect()
    env.reset(seed=42)
    try:
        ob_ids = set(env._obstacles)
        lidar = PyBulletLidar()
        scan = lidar.scan(env)
        ground, wall, ob, sky = [], [], [], 0
        for idx, h in enumerate(scan["hits"]):
            oid = int(h[0])
            if oid < 0:
                sky += 1
                continue
            rng = math.dist(scan["origins"][idx], h[3])
            if oid in ob_ids:
                ob.append(rng)
            elif abs(h[3][2]) < 0.15:
                ground.append(rng)
            else:
                wall.append(rng)
        assert len(ground) > 50 and 16.0 < min(ground) < 18.0
        assert len(wall) > 20 and 7.0 < min(wall) < 16.0
        assert len(ob) >= 1 and min(ob) > 20.0  # true obstacle range
        assert sky > 300  # upward/over-wall rays: no return, dropped
        pts = lidar.to_points(scan)
        assert pts.shape[0] == len(ground) + len(wall) + len(ob)
    finally:
        env.close()


def test_explicit_empty_scenario_gives_open_road():
    """scenario={'obstacles': []} must mean open road (no silent defaults)."""
    pytest.importorskip("pybullet")
    from src.simulation.pybullet_env import PyBulletEnv

    env = PyBulletEnv()
    env.connect()
    try:
        env.reset(seed=42, scenario={"obstacles": []})
        assert env.get_obstacles() == []
        env.reset(seed=42)
        assert len(env.get_obstacles()) > 0
    finally:
        env.close()


def test_state_geometry_correspondence():
    """State values must track geometry: open road -> forward sectors
    clearer than wall-bound sides; a 10 m in-lane obstacle must shrink
    forward clearance and nearest-forward distance."""
    pytest.importorskip("pybullet")
    from src.simulation.pybullet_env import PyBulletEnv
    from src.simulation.pybullet_lidar import PyBulletLidar, to_lidar_frame
    from src.simulation.stages_runner import run_stages_1_to_6

    def state_for(scenario):
        env = PyBulletEnv()
        env.connect()
        env.reset(seed=42, scenario=scenario)
        try:
            lidar = PyBulletLidar()
            pts = lidar.to_points(lidar.scan(env))
            frame = to_lidar_frame(pts, "corr-00001", 0.0)
            return run_stages_1_to_6(frame.points, frame.frame_id,
                                     frame.timestamp)["state"]
        finally:
            env.close()

    open_st = state_for({"obstacles": []})
    blocked_st = state_for({"obstacles": [{"xyz": [-40.0, 0.0, 1.0],
                                           "half": [1.5, 1.0, 1.0]}]})
    o_sec, b_sec = open_st["sector_ranges_m"], blocked_st["sector_ranges_m"]
    # Sides face the real walls in both cases (corridor constraint).
    assert min(o_sec[1], o_sec[2], o_sec[5], o_sec[6]) < 12.0
    # In-lane obstacle at 10 m must read closer than open-road forward.
    assert min(b_sec[7], b_sec[0]) < min(o_sec[7], o_sec[0])
    assert (blocked_st["safety"]["nearest_forward_obstacle_m"] is not None
            and blocked_st["safety"]["nearest_forward_obstacle_m"]
            < open_st["safety"]["nearest_forward_obstacle_m"])
    for st in (open_st, blocked_st):
        assert len(st["state_vector"]) == 13
        assert all(0.0 <= v <= 1.0 for v in st["state_vector"])

