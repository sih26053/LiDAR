"""Jev enrichment + criteria tests (Phases A/B). NO live calls.

Phase A: 13-D contract preserved; enrichment fields exist, typed, and
derived from existing data; density semantics documented; emergency
flag correct.
Phase B: mocked transport parses forward/left/right/stop/conf/probs
and handles malformed/missing/failed paths.
"""

import json


def _vec():
    return [0.3123, 0.2592, 0.2592, 0.3123, 0.3123, 0.2592, 0.2592, 0.3123,
            1.0, 0.1692, 0.1846, 0.4671, 0.1666]


def _ranges():
    return [9.37, 7.97, 7.97, 9.37, 9.37, 7.97, 7.97, 9.37]


# Phase A ---------------------------------------------------------------

def test_13d_contract_unchanged():
    from src.decision.jev_state_adapter import FIELD_NAMES, to_named_state

    assert len(FIELD_NAMES) == 13
    assert FIELD_NAMES[8] == "obstacle_density"
    named = to_named_state(_vec(), _ranges())
    for f in FIELD_NAMES:
        assert f in named
    assert named["state_dim"] == 13


def test_enrichment_fields_and_derivation():
    from src.decision.jev_state_adapter import enrich_for_jev, to_named_state

    safety = {"nearest_forward_obstacle_m": 12.53, "emergency_stop": False}
    en = enrich_for_jev(to_named_state(_vec(), _ranges()), safety)
    assert en["forward_clearance_m"] == 9.37
    assert en["left_clearance_m"] == 7.97
    assert en["right_clearance_m"] == 7.97
    assert en["nearest_obstacle_m"] == 7.97
    assert en["nearest_forward_obstacle_m"] == 12.53
    assert en["emergency_flag"] is False
    assert en["obstacle_density"] == 1.0
    assert isinstance(en["forward_clearance_m"], float)
    # 13-D fields preserved exactly
    for i in range(8):
        assert en[f"sector_{i}_range"] == _vec()[i]


def test_enrichment_meter_fallback_and_emergency():
    from src.decision.jev_state_adapter import enrich_for_jev, to_named_state

    named = to_named_state([0.5] * 13, [])
    en = enrich_for_jev(named, {"nearest_forward_obstacle_m": None,
                                "emergency_stop": True})
    assert en["forward_clearance_m"] == 15.0  # 0.5 x 30.0 fallback
    assert en["emergency_flag"] is True
    assert en["nearest_forward_obstacle_m"] is None
    en2 = enrich_for_jev(named)
    assert en2["emergency_flag"] is False
    assert en2["nearest_forward_obstacle_m"] is None


def test_density_semantics_documented():
    from src.decision.jev_state_adapter import DENSITY_SEMANTICS, enrich_for_jev

    assert "evidence presence" in DENSITY_SEMANTICS
    assert "NOT" in DENSITY_SEMANTICS and "blocked" in DENSITY_SEMANTICS
    en = enrich_for_jev({"sector_%d_range" % i: 0.5 for i in range(8)} | {
        "obstacle_density": 1.0, "moving_share": 0.0, "static_share": 0.0,
        "mean_importance": 0.1, "mean_uncertainty": 0.1,
        "sector_ranges_m": [], "state_dim": 13})
    assert en["obstacle_density_semantics"] == DENSITY_SEMANTICS
    doc = (enrich_for_jev.__doc__ or "").lower()
    assert "forward_clearance_m" in doc and "derived" in doc


def test_criteria_distinguish_unsafe_vs_directional():
    from src.decision.jev_client import build_question

    q = build_question()
    assert q["type"] == "choice"
    assert set(q["criteria"]) == {"forward", "left", "right", "stop"}
    assert "emergency" in q["criteria"]["stop"].lower()
    assert "no direction" in q["criteria"]["stop"].lower()
    for d in ("forward", "left", "right"):
        assert "clearance" in q["criteria"][d].lower()
    assert "clearance" in q["instructions"].lower()


# Phase B (mocked; no network) -------------------------------------------

def _client_with(monkeypatch, body: bytes = None, boom: bool = False):
    import urllib.request

    from src.decision import jev_client as C

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    monkeypatch.delenv("JEV_ENDPOINT", raising=False)

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return body

    if boom:
        def _boom(*a, **k):
            raise ConnectionRefusedError("nope")
        monkeypatch.setattr(urllib.request, "urlopen", _boom)
    else:
        monkeypatch.setattr(urllib.request, "urlopen",
                            lambda *a, **k: FakeResp())
    return C.JevClient()


def _ok(choice, conf):
    return json.dumps({
        "model": "typesafe/jev-1.13",
        "answers": {"action": {"choice": choice, "confidence": conf,
                               "probabilities": {"forward": 0.7, "left": 0.1,
                                                 "right": 0.1, "stop": 0.1}}},
    }).encode()


def test_parse_forward_left_right_stop(monkeypatch):
    cases = [("forward", "forward"), ("left", "turn_left"),
             ("right", "turn_right"), ("stop", "stop")]
    for choice, expect in cases:
        client = _client_with(monkeypatch, _ok(choice, 0.75))
        rec = client.decide({"sector_0_range": 0.5})
        assert rec.status == "OK" and rec.proposed_action == expect
        assert rec.confidence == 0.75
        assert rec.probabilities["forward"] == 0.7


def test_malformed_missing_and_failed(monkeypatch):
    import urllib.request

    from src.decision import jev_client as C

    c = _client_with(monkeypatch, b'{"answers": {"action": {"choice": "fly", "confidence": 0.9}}}')
    assert c.decide({"a": 1}).status == "INVALID"
    c = _client_with(monkeypatch, b'{"answers": {"action": {"choice": "forward"}}}')
    assert c.decide({"a": 1}).status == "INVALID"  # missing confidence
    c = _client_with(monkeypatch, b'{"answers": {"action": {"choice": "forward", "confidence": 99}}}')
    assert c.decide({"a": 1}).status == "INVALID"  # out of range
    c = _client_with(monkeypatch, boom=True)
    rec = c.decide({"a": 1})
    assert rec.status == "FAILED" and "test-key-not-real" not in (rec.error or "")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(C, "resolve_api_key", lambda: ("", ""))
    rec = C.JevClient().decide({"a": 1})
    assert rec.status == "UNAVAILABLE" and rec.proposed_action is None
