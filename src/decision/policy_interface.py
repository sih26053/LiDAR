"""Phase 22 — Decision backend abstraction (DQN baseline + Jev default).

DecisionPolicy
    ├── DQNPolicy  (legacy baseline: existing rl_agent, argmax Q-values)
    └── JevPolicy  (default: JevDecisionSystem + confidence gate)

`decide_with_safety()` chains policy -> confidence gate (Jev only; the
DQN exposes no calibrated confidence, documented) -> deterministic
safety layer -> execution record. Jev can NEVER reach hardware directly.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from src.decision.jev_decision import JevDecisionSystem, load_jev_config
from src.safety_controller import evaluate as safety_evaluate

DEFAULT_MODE = "jev"


class DecisionPolicy:
    mode = "base"

    def decide(self, named_state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class DQNPolicy(DecisionPolicy):
    """Legacy baseline: untrained NumPy DQN (no confidence available)."""

    mode = "dqn"

    def decide(self, named_state: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np

        from src import rl_agent

        t0 = time.perf_counter()
        vec = [named_state[k] for k in
               __import__("src.decision.jev_state_adapter",
                          fromlist=["FIELD_NAMES"]).FIELD_NAMES]
        out = rl_agent.get_agent().act(np.asarray(vec, dtype=np.float64))
        return {"proposed_action": out["action"], "confidence": None,
                "confidence_available": False,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "status": "OK-UNCALIBRATED",
                "note": "DQN exposes no calibrated confidence; safety layer still applies"}


class JevPolicy(DecisionPolicy):
    """Default backend: Jev + configurable confidence gate."""

    mode = "jev"

    def __init__(self):
        self.system = JevDecisionSystem()
        self.min_confidence = float(load_jev_config().get(
            "confidence_policy", {}).get("min_confidence", 0.6))
        self.fallback = str(load_jev_config().get(
            "confidence_policy", {}).get("fallback_action", "stop"))

    def decide(self, named_state: Dict[str, Any]) -> Dict[str, Any]:
        raw = self.system.decide(named_state)
        if raw["status"] != "OK":
            return {**raw, "confidence_available": True,
                    "gated_action": None,
                    "gate": f"service {raw['status']}; safe fallback applies"}
        if raw["confidence"] < self.min_confidence:
            return {**raw, "confidence_available": True,
                    "gated_action": None,
                    "gate": (f"confidence {raw['confidence']:.3f} < "
                             f"{self.min_confidence}; safe fallback applies")}
        return {**raw, "confidence_available": True,
                "gated_action": raw["proposed_action"], "gate": "passed"}


def get_policy(mode: str | None = None) -> DecisionPolicy:
    mode = (mode or DEFAULT_MODE).lower()
    if mode == "dqn":
        return DQNPolicy()
    return JevPolicy()


def _default_jev_log_path() -> str:
    from pathlib import Path as _P
    return str(_P(__file__).resolve().parent.parent.parent
               / "results" / "decisions" / "jev_requests.jsonl")


def append_jev_log(entry: Dict[str, Any], log_path: str | None = None) -> None:
    """Best-effort per-decision Jev log (never raises into the control loop).

    Entry carries no credentials by construction (callers never supply any).
    """
    import json as _json
    from pathlib import Path as _P
    path = _P(log_path or _default_jev_log_path())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(_json.dumps(entry, default=str) + "\n")
    except (OSError, ValueError):
        pass


def decide_with_safety(policy: DecisionPolicy, named_state: Dict[str, Any],
                       nearest_forward_obstacle_m: float | None,
                       map_valid: bool = True,
                       emergency: bool = False,
                       frame_id: str | None = None,
                       log_path: str | None = None,
                       run_id: str | None = None,
                       source: str | None = None) -> Dict[str, Any]:
    """Policy -> confidence gate -> safety layer -> full audit record.

    `source` tags action origin: "jev" | "dqn" | "fallback" | "manual".
    When omitted it defaults to the policy mode, or "fallback" when no
    decision was produced and the safe fallback applies. Manual actions
    must pass source="manual" explicitly so they never contaminate
    Jev performance metrics.
    """
    import datetime as _dt
    t0 = time.perf_counter()
    prop = policy.decide(named_state)
    candidate = prop.get("gated_action", prop.get("proposed_action"))
    if candidate is None:
        candidate = "stop"
        gate_note = prop.get("gate", prop.get("status", "no decision"))
        resolved_source = source or "fallback"
    else:
        gate_note = prop.get("gate", "passed-or-uncalibrated")
        resolved_source = source or policy.mode
    t_safety = time.perf_counter()
    verdict = safety_evaluate(candidate, nearest_forward_obstacle_m,
                              map_valid=map_valid, emergency=emergency)
    safety_ms = round((time.perf_counter() - t_safety) * 1000.0, 3)
    record = {"decision_model": policy.mode,
              "run_id": run_id,
              "source": resolved_source,
              "frame_id": frame_id,
              "proposed_action": prop.get("proposed_action"),
              "confidence": prop.get("confidence"),
              "probabilities": prop.get("probabilities"),
              "jev_status": prop.get("status"),
              "jev_error": prop.get("error"),
              "endpoint": prop.get("endpoint"),
              "model": prop.get("model"),
              "usage": prop.get("usage"),
              "gate": gate_note,
              "safety_status": verdict["verdict"],
              "safety_reason": verdict["reason"],
              "executed_action": verdict["final_action"],
              "safety_override": verdict["verdict"] != "SAFE_TO_EXECUTE",
              "policy_latency_ms": prop.get("latency_ms"),
              "safety_latency_ms": safety_ms,
              "total_latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)}
    if policy.mode == "jev":
        append_jev_log({
            "run_id": run_id,
            "timestamp": _dt.datetime.fromtimestamp(
                prop.get("request_ts") or time.time(),
                tz=_dt.timezone.utc).isoformat(),
            "frame_id": frame_id,
            "endpoint": prop.get("endpoint"),
            "model": prop.get("model"),
            "request_status": prop.get("status"),
            "selected_action": prop.get("proposed_action"),
            "probabilities": prop.get("probabilities"),
            "confidence": prop.get("confidence"),
            "latency_ms": prop.get("latency_ms"),
            "safety_latency_ms": safety_ms,
            "fallback": gate_note,
            "error": prop.get("error"),
            "safety_status": verdict["verdict"],
            "executed_action": verdict["final_action"],
        }, log_path)
    return record
