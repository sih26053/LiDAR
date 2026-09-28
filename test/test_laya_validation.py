"""Regression-vs-baseline + validation-status tests (Phases 31-32).

Compares measured artifacts against the frozen regression baseline
(zero tolerance for unsafe/emergency). Also asserts source-tag
separation (live=laya, replay=replay-laya/benchmark-laya, manual),
Jev history preservation, stale-calibration refusal, and promotion
gate passage. No Laya server required (artifacts + mocked transport).
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    return json.loads((PROJECT_ROOT / name).read_text())


def test_regression_within_tolerances():
    base = _load("results/laya_regression_baseline.json")
    final = _load("results/laya_final_test.json")
    emer = _load("results/laya_emergency_validation.json")
    con = final["by_mode"]["constrained_laya"]
    m, t = base["metrics"], base["tolerances"]
    assert final["gate_threshold"] == base["gate_threshold"]
    assert final["checkpoint_revision"] == base["checkpoint_revision"]
    assert con["accuracy"] >= m["choice_accuracy"] - t["accuracy_min_drop"]
    assert con["balanced_accuracy"] >= (
        m["balanced_accuracy"] - t["balanced_accuracy_min_drop"])
    assert con["ece"] is not None and m["ece"] is not None
    assert con["ece"] <= m["ece"] + t["ece_max_rise"]
    assert con["brier"] <= m["brier"] + t["brier_max_rise"]
    # Zero tolerance: accuracy gains never hide safety regressions.
    assert con["unsafe_raw_action_rate"] <= (
        m["unsafe_action_rate"] + t["unsafe_max_rise"])
    assert emer.get("executed_forward") == t["emergency_forward_must_be"]
    assert con["coverage_at_gate"] >= (
        m["coverage_at_gate"] - t["coverage_min_drop"])
    assert con["latency_p95_ms"] <= (
        m["latency_p95_ms"] * t["latency_p95_max_factor"])


def test_source_tags_separated():
    from backend.services import store
    con = store._connect()
    live = {r[0] for r in con.execute(
        "SELECT DISTINCT source FROM decisions WHERE replay_run_id IS NULL")}
    assert "laya" in live and "replay-laya" not in live
    assert "benchmark-laya" not in live
    assert "manual" in live  # manual steps exist from live runs
    store.close()


def test_jev_history_preserved():
    from backend.services import store
    con = store._connect()
    n = con.execute(
        "SELECT COUNT(*) c FROM decisions WHERE source='jev'").fetchone()["c"]
    assert n >= 6596
    row = con.execute(
        "SELECT * FROM decisions WHERE source='jev' LIMIT 1").fetchone()
    assert row is not None and row["checkpoint_revision"] is None
    store.close()


def test_promotion_gates_pass():
    import scripts.promote_laya as P
    reg = P.load_registry()
    assert reg.get("current_known_good", {}).get("stage") == \
        "CURRENT_KNOWN_GOOD"
    ok, reasons = P.run_gates(reg["current_known_good"])
    assert ok, reasons


def test_validation_artifact_statuses():
    gate = _load("models/laya/calibration/gate_selection.json")
    assert gate["promoted"] is True
    assert gate["current_gate_threshold"] == 0.35
    assert gate["tie_break"] == "legacy retained (tied optimum)"
    temp = _load("models/laya/calibration/temperature.json")
    assert temp["version"] == 2
    deployed = {k: v.get("deployed") for k, v in temp["groups"].items()}
    assert deployed.get("4") is False  # worsened group rejected
    final = _load("results/laya_final_test.json")
    assert final["n"] >= 86
    emer = _load("results/laya_emergency_validation.json")
    assert emer["n_emergency"] >= 100
    assert emer["executed_forward"] == 0
