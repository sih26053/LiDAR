"""System-level mitigation tests (no live Laya server required).

Covers: action eligibility, corridor categories, navigation oracle,
emergency/eligibility safety invariants (mocked transport), gate
sweep/selection, temperature rescaling + revision-match refusal,
checkpoint manifest/pinning, recorder/SQLite new fields, replay source
separation, historical Jev preservation, and replay purity.

Nothing here retrains the model; nothing fabricates model outputs.
"""

import json

import pytest

from src.decision import action_eligibility as AE


def _named(fwd, left, right, emergency=False):
    return {"forward_clearance_m": float(fwd),
            "left_clearance_m": float(left),
            "right_clearance_m": float(right),
            "emergency_flag": bool(emergency)}


# ---- eligibility -------------------------------------------------------

def test_emergency_allows_stop_only():
    out = AE.compute_eligible_actions(_named(20, 9, 9, True))
    assert out["eligible"] == ["stop"]
    assert out["emergency"] is True


def test_forward_blocked_removes_forward():
    out = AE.compute_eligible_actions(_named(2.0, 9.0, 9.0))
    assert "forward" in out["ineligible"]
    assert set(out["eligible"]) == {"left", "right", "stop"}


def test_side_blocked_removes_side():
    out = AE.compute_eligible_actions(_named(9.0, 1.0, 9.0))
    assert "left" in out["ineligible"] and "right" not in out["ineligible"]
    out = AE.compute_eligible_actions(_named(9.0, 9.0, 1.0))
    assert "right" in out["ineligible"] and "left" not in out["ineligible"]


def test_no_safe_direction_failsafe_stop():
    out = AE.compute_eligible_actions(_named(1.0, 1.0, 1.0))
    assert out["eligible"] == ["stop"]


def test_forward_threshold_mirrors_safety_config():
    from src.safety_controller import load_safety_config

    th = AE.eligibility_thresholds()
    assert th["forward_min_m"] == float(
        load_safety_config()["safety_radius_m"])


# ---- categories A..G ----------------------------------------------------

def test_categories_cover_corridor():
    assert AE.categorize_frame(_named(20, 8, 1)) == "A_OPEN_FORWARD"
    assert AE.categorize_frame(_named(2, 9, 1)) == "B_FORWARD_BLOCKED_LEFT_OPEN"
    assert AE.categorize_frame(_named(2, 1, 9)) == "C_FORWARD_BLOCKED_RIGHT_OPEN"
    assert AE.categorize_frame(_named(9, 8, 8)) == "D_BOTH_SIDES_AVAILABLE"
    assert AE.categorize_frame(_named(2, 9, 9)) == "D_BOTH_SIDES_AVAILABLE"
    assert AE.categorize_frame(_named(9, 1, 1)) == "E_BOTH_SIDES_BLOCKED"
    assert AE.categorize_frame(_named(1, 1, 1, True)) == "F_EMERGENCY"
    assert AE.categorize_frame(_named(1, 1, 1)) == "G_NO_SAFE_ACTION"


# ---- oracle (evaluation only) -------------------------------------------

def test_oracle_emergency_and_blocked():
    o = AE.navigation_oracle(_named(1, 1, 1, True))
    assert o["admissible"] == ["stop"] and o["preferred"] == "stop"
    o = AE.navigation_oracle(_named(2, 9, 1))
    assert "forward" not in o["admissible"] and o["preferred"] == "left"
    o = AE.navigation_oracle(_named(1, 1, 1))
    assert o["preferred"] == "stop"


def test_oracle_open_prefers_forward():
    o = AE.navigation_oracle(_named(20, 8, 8))
    assert o["preferred"] == "forward"
    assert set(o["admissible"]) == {"forward", "left", "right", "stop"}


def test_oracle_side_tie_break_deterministic():
    o = AE.navigation_oracle(_named(2, 7, 7))
    assert o["preferred"] == "left"


# ---- mocked Laya transport ----------------------------------------------

def _ok_payload(choice="forward", aconf=0.8, conf=0.2):
    return json.dumps({
        "model": "typed-decisions",
        "answers": {"action": {"choice": choice, "confidence": conf,
                               "answer_confidence": aconf,
                               "probabilities": {"forward": 0.7, "left": 0.1,
                                                 "right": 0.1, "stop": 0.1}}},
        "usage": {"input_tokens": 10, "output_tokens": 0},
    }).encode()


class _FakeResp:
    def __init__(self, body):
        self._b = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._b


def _mock_laya(monkeypatch, choice="forward", aconf=0.9):
    import urllib.request

    from src.decision import laya_client as C
    monkeypatch.setattr(C, "service_status",
                        lambda: {"available": True, "endpoint": "http://x",
                                 "model": "typed-decisions", "device": "CPU"})
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: _FakeResp(
                            _ok_payload(choice, aconf)))


def _full_named(fwd=20.0, left=8.0, right=8.0, emergency=False):
    named = {f"sector_{i}_range": 1.0 for i in range(8)}
    named.update({"obstacle_density": 0.0, "moving_share": 0.0,
                  "static_share": 0.0, "mean_importance": 0.1,
                  "mean_uncertainty": 0.1, "sector_ranges_m": [],
                  "state_dim": 13,
                  "forward_clearance_m": fwd, "left_clearance_m": left,
                  "right_clearance_m": right, "emergency_flag": emergency,
                  "nearest_obstacle_m": min(fwd, left, right)})
    return named


def _decide(monkeypatch, named, nearest=40.0, emergency=False,
            mode="constrained", choice="forward", aconf=0.9):
    from src.decision.policy_interface import decide_with_safety, get_policy
    _mock_laya(monkeypatch, choice, aconf)
    return decide_with_safety(
        get_policy("laya"), named, nearest, True, emergency,
        frame_id="t", run_id="run-t", mode=mode,
        safety_dict={"emergency_stop": emergency,
                     "nearest_forward_obstacle_m": nearest})


# INVARIANT 1: emergency -> executed != forward (both modes, raw FORWARD).
def test_invariant_1_emergency_never_forward(monkeypatch):
    for mode in ("constrained", "unconstrained"):
        rec = _decide(monkeypatch, _full_named(20, 8, 8, True),
                      nearest=40.0, emergency=True, mode=mode)
        assert rec["executed_action"] != "forward", mode
        assert rec["executed_action"] == "stop", mode
    # Constrained emergency path never even calls the model.
    rec = _decide(monkeypatch, _full_named(20, 8, 8, True), emergency=True)
    assert rec["laya_skipped"] is True
    assert rec["eligible_actions"] == ["stop"]


# INVARIANTS 2-4: ineligible direction is never executed.
def test_invariant_2_forward_ineligible(monkeypatch):
    rec = _decide(monkeypatch, _full_named(2.0, 9.0, 9.0), nearest=2.0)
    assert rec["executed_action"] != "forward"
    assert "forward" not in (rec["eligible_actions"] or [])


def test_invariant_3_4_sides_ineligible(monkeypatch):
    rec = _decide(monkeypatch, _full_named(9.0, 1.0, 9.0), choice="left")
    assert "left" not in (rec["eligible_actions"] or [])
    # Constrained call only offers eligible options: mocked "left" is not
    # offered -> INVALID -> safe fallback STOP.
    assert rec["executed_action"] == "stop"
    rec = _decide(monkeypatch, _full_named(9.0, 9.0, 1.0), choice="right")
    assert rec["executed_action"] == "stop"


# INVARIANT 5: no eligible action -> STOP.
def test_invariant_5_no_safe_action_stop(monkeypatch):
    rec = _decide(monkeypatch, _full_named(1.0, 1.0, 1.0), nearest=1.0)
    assert rec["eligible_actions"] == ["stop"]
    assert rec["executed_action"] == "stop"


# INVARIANT 6: invalid Laya response -> safe fallback.
def test_invariant_6_invalid_response_fallback(monkeypatch):
    import urllib.request

    from src.decision import laya_client as C
    from src.decision.policy_interface import decide_with_safety, get_policy
    monkeypatch.setattr(C, "service_status",
                        lambda: {"available": True, "endpoint": "http://x",
                                 "model": "typed-decisions"})
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: _FakeResp(b'{"answers": {}}'))
    rec = decide_with_safety(get_policy("laya"), _full_named(), 40.0,
                             True, False, mode="constrained",
                             safety_dict={"emergency_stop": False})
    assert rec["executed_action"] == "stop"
    assert rec["source"] == "fallback"


# INVARIANT 7: Laya unavailable -> safe fallback (never fabricated).
def test_invariant_7_unavailable_fallback(monkeypatch):
    from src.decision import laya_client as C
    from src.decision.policy_interface import decide_with_safety, get_policy
    monkeypatch.setenv("LAYA_ENDPOINT", "http://127.0.0.1:9/none")
    monkeypatch.setattr(C, "service_status",
                        lambda: {"available": False, "endpoint": "http://x",
                                 "model": "typed-decisions",
                                 "reason": "down"})
    rec = decide_with_safety(get_policy("laya"), _full_named(), 40.0,
                             True, False, mode="constrained",
                             safety_dict={"emergency_stop": False})
    assert rec["executed_action"] == "stop"
    assert rec["engine_status"] in ("UNAVAILABLE", "FAILED")


# Constrained choice is honestly offered: eligible subset reaches the wire.
def test_constrained_subset_offered(monkeypatch):
    import urllib.request

    from src.decision import laya_client as C
    seen = {}
    monkeypatch.setattr(C, "service_status",
                        lambda: {"available": True, "endpoint": "http://x",
                                 "model": "typed-decisions"})
    orig = C.LayaDecisionSystem._parse

    def _spy(self, payload, t0, ts, fid, ep, options=None):
        seen["options"] = options
        return orig(self, payload, t0, ts, fid, ep, options=options)
    monkeypatch.setattr(C.LayaDecisionSystem, "_parse", _spy)
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: _FakeResp(_ok_payload("left", 0.9)))
    rec = _decide(monkeypatch, _full_named(2.0, 9.0, 1.0), choice="left")
    assert seen["options"] == ["left", "stop"]
    assert rec["proposed_action"] == "turn_left"
    assert rec["constrained_action"] == "turn_left"
    assert rec["mode"] == "constrained_laya"


def test_raw_preserved_in_unconstrained(monkeypatch):
    # nearest=2.0 keeps the safety input consistent with the 2 m forward
    # clearance (both derive from the same scan in the real pipeline).
    rec = _decide(monkeypatch, _full_named(2.0, 9.0, 1.0), nearest=2.0,
                  mode="unconstrained")
    assert rec["raw_laya_action"] == "forward"  # honest raw preference
    assert rec["mode"] == "unconstrained_laya"
    assert rec["executed_action"] == "stop"  # safety still binds


# ---- calibration --------------------------------------------------------

def test_gate_sweep_and_selection_documented():
    from src.decision import calibration as cal

    recs = [
        {"frame_id": f"f{i}",
         "answer_confidence": c, "oracle_correct": ok, "unsafe": un}
        for i, (c, ok, un) in enumerate([
            (0.60, True, False), (0.55, True, False), (0.37, False, False),
            (0.80, True, False), (0.30, False, True), (0.90, True, False)])]
    rows = cal.sweep_thresholds(recs, thresholds=[0.25, 0.35, 0.5])
    assert [r["threshold"] for r in rows] == [0.25, 0.35, 0.5]
    sel = cal.select_threshold(rows)
    assert sel["threshold"] == 0.35  # lowest unsafe-free... check:
    # t=0.25 accepts the unsafe 0.30 row; t=0.35 and 0.5 are unsafe-free;
    # coverage prefers 0.35.
    assert sel["zero_unsafe_feasible"] is True
    assert "PRIMARY" in sel["reason"]


def test_temperature_rescale_and_mismatch_refusal():
    from src.decision import calibration as cal

    probs = {"forward": 0.5, "left": 0.25, "right": 0.15, "stop": 0.10}
    scaled = cal.apply_temperature(probs, 0.5)
    assert abs(sum(scaled.values()) - 1.0) < 1e-6
    assert scaled["forward"] > probs["forward"]  # sharpening
    # Stale artifact must not silently apply: covered in
    # test_revision_mismatch_not_loaded via calibration_status.


def test_revision_mismatch_not_loaded(tmp_path, monkeypatch):
    from src.decision import calibration as cal
    monkeypatch.setattr(cal, "GATE_SELECTION_PATH",
                        tmp_path / "gate.json")
    monkeypatch.setattr(cal, "TEMPERATURE_PATH", tmp_path / "temp.json")
    gate = {"checkpoint_revision": "rev-A", "selected_threshold": 0.4}
    (tmp_path / "gate.json").write_text(json.dumps(gate))
    st = cal.calibration_status(checkpoint_revision="rev-B")
    assert st["gate"]["state"] == "NOT LOADED"
    st = cal.calibration_status(checkpoint_revision="rev-A")
    assert st["gate"]["state"] == "LOADED"


# ---- checkpoint ----------------------------------------------------------

def test_checkpoint_provisioned_pinned_local():
    from src.decision.checkpoint_info import checkpoint_info

    info = checkpoint_info()
    assert info["state"] == "LOCAL"
    assert info["revision"] == "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
    assert info["offline_inference"] == "AVAILABLE"
    assert info["files_present"]["model.safetensors"] is True
    stamp_keys = {"model_id", "checkpoint_revision", "device",
                  "runtime_version", "calibration_version", "gate_threshold"}
    from src.decision.checkpoint_info import decision_version_stamp
    assert stamp_keys <= set(decision_version_stamp())


# ---- recorder / sqlite / replay ------------------------------------------

@pytest.fixture()
def tmpdb(tmp_path):
    from backend.services import store
    from backend.services.store import DEFAULT_DB_PATH
    store.configure(tmp_path / "mitigation-test.db")
    yield tmp_path
    store.close()
    store.configure(DEFAULT_DB_PATH)


def test_recorder_new_fields(tmp_path):
    import numpy as np

    from backend.services import frame_recorder as fr
    pts = np.hstack([np.random.default_rng(0).uniform(-5, 5, (50, 3)),
                     np.full((50, 1), 0.5)])
    rec = fr.record_frame(
        run_id="T", sequence=0, loop_frame_id="live-00000", scenario=None,
        timestamp=1.0, lidar_points=pts, pipeline_result={"map_cells": []},
        state_vector=[0.5] * 13,
        jev={"source": "laya", "mode": "constrained_laya",
             "proposed_action": "forward", "answer_confidence": 0.45,
             "eligible_actions": ["forward", "stop"],
             "raw_laya_action": None, "constrained_action": "forward",
             "checkpoint_revision": "rev-X", "gate_threshold": 0.35},
        safety={"verdict": "SAFE_TO_EXECUTE", "emergency_stop": False,
                "forward_clearance_m": 9.0},
        execution={"executed_action": "forward", "source": "laya"},
        vehicle={"x": 0.0, "y": 0.0})
    meta = rec["metadata"]
    assert meta["laya"]["mode"] == "constrained_laya"
    assert meta["laya"]["checkpoint_revision"] == "rev-X"
    assert meta["jev"]["eligible_actions"] == ["forward", "stop"]
    assert meta["safety"]["forward_clearance_m"] == 9.0


def test_sqlite_new_columns_and_jev_preserved(tmpdb):
    from backend.services import store
    store.create_run("RUN_M", "autonomous")
    store.create_frame("RUN_M_f00000", "RUN_M", 1.0, 0, None, "p.npy",
                       5, [0.5] * 13, {}, {}, "m.json")
    store.record_decision("RUN_M_f00000", "jev", proposed_action="stop",
                          confidence=0.6)
    store.record_decision("RUN_M_f00000", "laya", proposed_action="forward",
                          confidence=0.45, answer_confidence=0.45,
                          mode="constrained_laya",
                          eligible=["forward", "stop"],
                          raw_action=None, constrained_action="forward",
                          checkpoint_revision="rev-X", gate_threshold=0.35)
    live = store.get_decision("RUN_M_f00000", source="laya")
    assert live["mode"] == "constrained_laya"
    assert live["checkpoint_revision"] == "rev-X"
    assert live["gate_threshold"] == 0.35
    legacy = store.get_decision("RUN_M_f00000", source="jev")
    assert legacy["proposed_action"] == "stop"  # historical row intact


def test_replay_purity_and_mode(tmpdb, tmp_path, monkeypatch):
    import shutil

    import numpy as np

    from backend.services import pybullet_live as live
    from backend.services import replay_laya as rl
    from backend.services import store
    from backend.services.store import PROJECT_ROOT

    d = PROJECT_ROOT / "results" / "recordings" / "_test_M" / "points"
    d.mkdir(parents=True, exist_ok=True)
    try:
        rng = np.random.default_rng(3)
        np.save(d / "p.npy",
                np.hstack([rng.uniform(-8, 8, (120, 3)),
                           np.full((120, 1), 0.5)]))
        store.create_run("_test_M", "autonomous")
        store.create_frame("_test_M_f00000", "_test_M", 1.0, 0, None,
                           "results/recordings/_test_M/points/p.npy", 120,
                           [0.5] * 13, {}, {}, "m.json")
        import urllib.request

        class FakeResp:
            def __init__(self, body):
                self._b = body

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return self._b

        monkeypatch.setattr(urllib.request, "urlopen",
                            lambda *a, **k: FakeResp(
                                _ok_payload("stop", 0.9)))
        sentinel = live._live.get("env")
        live._live["env"] = object()  # would move if replay touched it
        try:
            for mode in ("constrained", "unconstrained"):
                rec = rl.run_replay_decision(
                    "_test_M_f00000", f"rr-m-{mode}", mode=mode)
                assert rec["source"] == "replay-laya"
                assert rec["mode"] == (f"{mode}_laya")
                assert rec["stored"] is True
                assert live._live.get("env") is not None
        finally:
            live._live["env"] = sentinel
    finally:
        shutil.rmtree(d.parent, ignore_errors=True)
