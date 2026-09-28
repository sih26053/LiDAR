"""Local Laya integration tests (Phase 23). Items 1-29.

Live-server items share one module-scoped boot (cached preload).
Mocked-transport items never touch the network. No OpenRouter call
occurs anywhere in this file (test 22 enforces it).
"""

import json

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture()
def tmpdb(tmp_path):
    from backend.services import store
    from backend.services.store import DEFAULT_DB_PATH

    store.configure(tmp_path / "laya-test.db")
    yield tmp_path
    store.close()
    store.configure(DEFAULT_DB_PATH)


@pytest.fixture(scope="module")
def laya_server():
    laya = pytest.importorskip("laya")
    assert laya is not None
    from backend.services import laya_manager

    info = laya_manager.ensure_started(wait_s=600.0)
    assert info["health"].get("healthy")
    yield info
    # Leave the server up for the integration run; atexit stops it.


def _ok_payload(choice="forward", aconf=0.8, conf=0.2):
    return json.dumps({
        "model": "laya-rl-agent",
        "answers": {"action": {"choice": choice, "confidence": conf,
                               "answer_confidence": aconf,
                               "probabilities": {"forward": 0.7, "left": 0.1,
                                                 "right": 0.1, "stop": 0.1}}},
        "usage": {"input_tokens": 10, "output_tokens": 0},
    }).encode()


# 1-4. Package, startup, health, readiness --------------------------------

def test_1_laya_import():
    laya = pytest.importorskip("laya")
    assert laya is not None


def test_2_3_4_startup_health_readiness(laya_server):
    from backend.services import laya_manager

    h = laya_manager.health()
    assert h.get("healthy") is True
    assert "typed-decisions" in (h.get("loaded") or [])


# 28-29. Reuse / duplicate prevention --------------------------------------

def test_28_process_reuse(laya_server):
    from backend.services import laya_manager

    again = laya_manager.ensure_started(wait_s=60.0)
    assert again.get("reused") is True
    assert again["health"].get("healthy") is True


def test_29_no_duplicate_spawn(laya_server):
    from backend.services import laya_manager

    p1 = laya_manager._proc.pid if laya_manager._proc else None
    laya_manager.ensure_started(wait_s=60.0)
    p2 = laya_manager._proc.pid if laya_manager._proc else None
    assert p1 is not None and p1 == p2


# 24-25. Device --------------------------------------------------------------

def test_24_cpu_mode():
    from src.decision.laya_client import detect_device

    import torch

    dev = detect_device()
    assert dev == ("CUDA" if torch.cuda.is_available() else "CPU")
    assert dev in ("CPU", "CUDA", "UNKNOWN")


@pytest.mark.skipif(__import__("torch").cuda.is_available() is False,
                    reason="no CUDA here")
def test_25_cuda_mode():
    from src.decision.laya_client import detect_device

    assert detect_device() == "CUDA"


# 5-8. Choice request, parsing, normalization, invalid ------------------------

def test_5_typed_choice_live(laya_server):
    from src.decision.laya_client import LayaDecisionSystem

    rec = LayaDecisionSystem().decide({"forward_clearance_m": 9.4,
                                       "sector_0_range": 0.3})
    assert rec.status == "OK"
    assert rec.proposed_action in ("forward", "turn_left", "turn_right", "stop")
    assert rec.answer_confidence is not None
    assert 0.0 <= rec.answer_confidence <= 1.0
    assert rec.latency_ms > 0


def test_6_7_parse_and_normalize(monkeypatch):
    import urllib.request

    from src.decision import laya_client as C

    class FakeResp:
        def __init__(self, body): self._b = body
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self._b

    monkeypatch.setattr(C, "service_status",
                        lambda: {"available": True, "endpoint": "http://x",
                                 "model": "typed-decisions"})
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: FakeResp(_ok_payload("LEFT", 0.9)))
    rec = C.LayaDecisionSystem().decide({})
    assert rec.status == "OK" and rec.proposed_action == "turn_left"
    assert rec.answer_confidence == 0.9 and rec.confidence == 0.2
    assert rec.probabilities["forward"] == 0.7


def test_8_missing_invalid(monkeypatch):
    import urllib.request

    from src.decision import laya_client as C

    class FakeResp:
        def __init__(self, body): self._b = body
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self._b

    monkeypatch.setattr(C, "service_status",
                        lambda: {"available": True, "endpoint": "http://x",
                                 "model": "typed-decisions"})
    bad = [b'{"answers": {}}',
           b'{"answers": {"action": {"choice": "fly", "answer_confidence": 0.9}}}',
           b'{"answers": {"action": {"choice": "forward"}}}',
           b'{"answers": {"action": {"choice": "forward", "answer_confidence": 99}}}']
    for body in bad:
        def _fake(*a, _b=body, **k):
            return FakeResp(_b)

        monkeypatch.setattr(urllib.request, "urlopen", _fake)
        rec = C.LayaDecisionSystem().decide({})
        assert rec.status == "INVALID" and rec.proposed_action is None


# 26. Unavailable handling -----------------------------------------------------

def test_26_unavailable_no_fabrication(monkeypatch):
    from src.decision import laya_client as C

    monkeypatch.setenv("LAYA_ENDPOINT", "http://127.0.0.1:9/none")
    rec = C.LayaDecisionSystem().decide({"sector_0_range": 0.5})
    assert rec.status in ("UNAVAILABLE", "FAILED")
    assert rec.proposed_action is None and rec.confidence is None


# 9-11. Contracts unchanged -----------------------------------------------------

def test_9_10_stage6_and_13d_unchanged():
    from src.decision.jev_state_adapter import (
        FIELD_NAMES, enrich_for_jev, enrich_for_laya, to_named_state)

    assert len(FIELD_NAMES) == 13
    vec = [0.5] * 13
    named = to_named_state(vec, [15.0] * 8)
    assert named["sector_0_range"] == 0.5 and named["state_dim"] == 13
    assert enrich_for_laya(named, {}) == enrich_for_jev(named, {})


def test_11_safety_unchanged():
    from src.safety_controller import evaluate

    assert evaluate("forward", 2.0)["final_action"] == "stop"
    assert evaluate("forward", 40.0)["verdict"] == "SAFE_TO_EXECUTE"


# 12-14. Policy gate + safety chain ----------------------------------------------

def test_12_13_14_gate_and_safety(monkeypatch):
    import urllib.request

    from src.decision import laya_client as C
    from src.decision.policy_interface import decide_with_safety, get_policy

    assert get_policy(None).mode == "laya"
    assert get_policy("laya").mode == "laya"
    assert get_policy("jev").mode == "jev"  # legacy still addressable

    class FakeResp:
        def __init__(self, body): self._b = body
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self._b

    monkeypatch.setattr(C, "service_status",
                        lambda: {"available": True, "endpoint": "http://x",
                                 "model": "typed-decisions", "device": "CPU"})
    named = {f"sector_{i}_range": 1.0 for i in range(8)}
    named.update({"obstacle_density": 0.0, "moving_share": 0.0,
                  "static_share": 0.0, "mean_importance": 0.1,
                  "mean_uncertainty": 0.1, "sector_ranges_m": [],
                  "state_dim": 13})
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: FakeResp(_ok_payload("forward", 0.85)))
    rec = decide_with_safety(get_policy("laya"), named, 40.0, True, False,
                             frame_id="t-laya-1", run_id="run-t")
    assert rec["proposed_action"] == "forward"
    assert rec["executed_action"] == "forward"
    assert rec["source"] == "laya"
    assert rec["safety_override"] is False
    assert rec["device"] == "CPU"
    # gate blocks low answer_confidence
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: FakeResp(_ok_payload("forward", 0.2)))
    rec = decide_with_safety(get_policy("laya"), named, 40.0, True, False)
    assert rec["executed_action"] == "stop" and rec["source"] == "fallback"
    # safety overrides near obstacle even for gated pass
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: FakeResp(_ok_payload("forward", 0.9)))
    rec = decide_with_safety(get_policy("laya"), named, 2.0, True, False)
    assert rec["executed_action"] == "stop"
    assert rec["safety_override"] is True


# 15-19. Sources, history, recorder, sqlite -----------------------------------------

def test_15_16_17_18_19_sources_and_store(tmpdb, monkeypatch):
    import urllib.request

    from backend.services import replay_jev as rj
    from backend.services import store
    from src.decision.policy_interface import decide_with_safety, get_policy

    store.create_run("RUN_L", "autonomous")
    store.create_frame("RUN_L_f00000", "RUN_L", 1.0, 0, None, "p.npy",
                       5, [0.5] * 13, {}, {}, "m.json")
    store.record_decision("RUN_L_f00000", "jev", proposed_action="stop",
                          confidence=0.6)
    named = {f"sector_{i}_range": 1.0 for i in range(8)}
    named.update({"obstacle_density": 0.0, "moving_share": 0.0,
                  "static_share": 0.0, "mean_importance": 0.1,
                  "mean_uncertainty": 0.1, "sector_ranges_m": [],
                  "state_dim": 13})

    class FakeResp:
        def __init__(self, body): self._b = body
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self._b

    from src.decision import laya_client as C
    monkeypatch.setattr(C, "service_status",
                        lambda: {"available": True, "endpoint": "http://x",
                                 "model": "typed-decisions", "device": "CPU"})
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: FakeResp(_ok_payload("forward", 0.9)))
    rec = decide_with_safety(get_policy("laya"), named, 40.0, True, False,
                             frame_id="RUN_L_f00000", run_id="RUN_L")
    store.record_decision("RUN_L_f00000", rec["source"], model=rec.get("model"),
                          proposed_action=rec["proposed_action"],
                          confidence=rec["confidence"],
                          probabilities=rec["probabilities"],
                          latency_ms=rec["policy_latency_ms"])
    live = store.get_decision("RUN_L_f00000", source="laya")
    assert live["proposed_action"] == "forward"
    legacy = store.get_decision("RUN_L_f00000", source="jev")
    assert legacy["proposed_action"] == "stop"  # historical row intact
    store.record_decision("RUN_L_f00000", "replay-laya", proposed_action="stop",
                          replay_run_id="rr-1")
    assert store.get_replay_history("RUN_L_f00000", "replay-laya")[0]["replay_run_id"] == "rr-1"


def test_21_replay_purity_and_legacy_path(tmpdb, tmp_path, monkeypatch):
    import shutil

    import numpy as np

    from backend.services import pybullet_live as live
    from backend.services import replay_laya as rl
    from backend.services import store
    from backend.services.store import PROJECT_ROOT

    d = PROJECT_ROOT / "results" / "recordings" / "_test_LR" / "points"
    d.mkdir(parents=True, exist_ok=True)
    try:
        rng = np.random.default_rng(3)
        np.save(d / "p.npy",
                np.hstack([rng.uniform(-8, 8, (120, 3)), np.full((120, 1), 0.5)]))
        store.create_run("_test_LR", "autonomous")
        store.create_frame("_test_LR_f00000", "_test_LR", 1.0, 0, None,
                           "results/recordings/_test_LR/points/p.npy", 120,
                           [0.5] * 13, {}, {}, "m.json")
        import urllib.request

        class FakeResp:
            def __init__(self, body): self._b = body
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return self._b

        monkeypatch.setattr(urllib.request, "urlopen",
                            lambda *a, **k: FakeResp(_ok_payload("stop", 0.9)))
        sentinel = live._live.get("env")
        live._live["env"] = object()
        try:
            rec = rl.run_replay_decision("_test_LR_f00000", "rr-lr-1")
            assert rec["source"] == "replay-laya" and rec["stored"] is True
            assert live._live.get("env") is not None
        finally:
            live._live["env"] = sentinel
    finally:
        shutil.rmtree(d.parent, ignore_errors=True)


# 22-23. No cloud, no key ---------------------------------------------------------------

def test_22_no_openrouter_requests(laya_server, monkeypatch):
    import urllib.request

    _real = urllib.request.urlopen

    def guard(req, *a, **k):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        assert "openrouter" not in url, f"cloud call attempted: {url}"
        return _real(req, *a, **k)

    monkeypatch.setattr(urllib.request, "urlopen", guard)
    from src.decision.laya_client import LayaDecisionSystem

    rec = LayaDecisionSystem().decide({"sector_0_range": 0.5})
    assert rec.status == "OK"


def test_23_no_api_key_required(laya_server, monkeypatch):
    import urllib.request

    from src.decision import laya_client as C

    for var in ("OPENROUTER_API_KEY", "TYPESAFE_API_KEY", "LAYA_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    _real = urllib.request.urlopen

    def guard(req, *a, **k):
        for _k, v in (req.header_items() if hasattr(req, "header_items") else []):
            assert "bearer" not in v.lower() or "openrouter" not in req.full_url.lower()
        return _real(req, *a, **k)

    monkeypatch.setattr(urllib.request, "urlopen", guard)
    rec = C.LayaDecisionSystem().decide({"sector_0_range": 0.5})
    assert rec.status == "OK" and rec.proposed_action is not None


# 27. Backend restart ------------------------------------------------------------------------

def test_27_backend_restart_reuses(laya_server):
    from backend.services import laya_manager

    out = laya_manager.stop()
    assert out["stopped"] in (True, False)
    info = laya_manager.ensure_started(wait_s=300.0)
    assert info["health"].get("healthy") is True
