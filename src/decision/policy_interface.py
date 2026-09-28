"""Decision backend abstraction (DQN baseline + Laya active).

DecisionPolicy (DecisionEngine interface: decide(state) -> record)
    ├── DQNPolicy  (legacy baseline: existing rl_agent, argmax Q-values)
    ├── JevPolicy  (LEGACY/DISABLED: cloud Jev; kept for historical
    │               records/tests only, NOT in the active path)
    └── LayaPolicy (ACTIVE default: local Laya + Laya confidence gate)

`decide_with_safety()` chains policy -> confidence gate -> deterministic
safety layer -> execution record. No engine can reach hardware directly.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from src.decision.jev_decision import JevDecisionSystem, load_jev_config
from src.decision.laya_client import LayaDecisionSystem, load_laya_config
from src.safety_controller import evaluate as safety_evaluate

DEFAULT_MODE = "laya"


class DecisionPolicy:
    """DecisionEngine interface: decide(named_state) -> decision record."""

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
    """LEGACY/DISABLED cloud backend (kept for historical records/tests).

    NOT in the active path. The live runtime uses LayaPolicy.
    """

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
    if mode == "jev":
        return JevPolicy()
    return LayaPolicy()


def resolve_gate_config() -> Dict[str, Any]:
    """Structured confidence-gate config (never a hardcoded 0.4 here).

    Source of truth: config/laya_config.json `confidence_gate`
    {enabled, metric, threshold}. Legacy `laya_policy.min_confidence`
    is honoured as fallback; code fallback is 0.6 (documented).
    """
    cfg = load_laya_config()
    gate = dict(cfg.get("confidence_gate", {}) or {})
    legacy = dict(cfg.get("laya_policy", {}) or {})
    threshold = gate.get("threshold", legacy.get("min_confidence", 0.6))
    return {
        "enabled": bool(gate.get("enabled", True)),
        "metric": str(gate.get("metric", legacy.get(
            "confidence_field", "answer_confidence"))),
        "threshold": float(threshold),
        "fallback_action": str(gate.get("fallback_action", legacy.get(
            "fallback_action", "stop"))),
        "source": ("confidence_gate" if "threshold" in gate
                   else ("laya_policy.min_confidence" if "min_confidence" in legacy
                         else "code-fallback-0.6")),
    }


class LayaPolicy(DecisionPolicy):
    """ACTIVE backend: local Laya + Laya confidence gate.

    Gate applies to `answer_confidence` (selected-choice probability,
    Laya-native scale) with the configured threshold (see
    resolve_gate_config(); the value in config is a SELECTED threshold
    under a documented validation objective, NOT a validated optimum).
    Raw `confidence` is passed through for logging only. Neither is
    task accuracy.

    decide(named_state, eligible=None): eligible=None offers all four
    choices (unconstrained/diagnostic); an eligible subset offers ONLY
    those choices (constrained/production).
    """

    mode = "laya"

    def __init__(self):
        self.system = LayaDecisionSystem()
        gate = resolve_gate_config()
        self.min_confidence = gate["threshold"]
        self.gate_metric = gate["metric"]
        self.gate_enabled = gate["enabled"]
        self.gate_source = gate["source"]
        self.fallback = gate["fallback_action"]

    def decide(self, named_state: Dict[str, Any],
               eligible: list | tuple | None = None) -> Dict[str, Any]:
        raw = self.system.decide(named_state, eligible=eligible)
        if isinstance(raw, dict):
            pass
        else:
            raw = raw.to_dict()
        raw = self._apply_temperature(raw)
        gate_value = raw.get("answer_confidence")
        base = {**raw, "confidence": gate_value,
                "confidence_available": True,
                "answer_confidence": gate_value,
                "laya_confidence_raw": raw.get("confidence"),
                "gate_metric": self.gate_metric,
                "gate_threshold": self.min_confidence}
        if raw["status"] != "OK":
            return {**base,
                    "gated_action": None,
                    "gate": f"service {raw['status']}; safe fallback applies"}
        if not self.gate_enabled:
            return {**base,
                    "gated_action": raw["proposed_action"],
                    "gate": "gate disabled by configuration; no fallback"}
        if gate_value is None or gate_value < self.min_confidence:
            shown = "missing" if gate_value is None else f"{gate_value:.3f}"
            return {**base,
                    "gated_action": None,
                    "gate": (f"laya answer_confidence {shown} < "
                             f"{self.min_confidence}; safe fallback applies")}
        return {**base,
                "gated_action": raw["proposed_action"], "gate": "passed"}

    def _apply_temperature(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the temperature artifact when (and only when) it matches.

        Match condition: artifact exists, fitted, and its checkpoint
        revision equals the current pinned revision. Otherwise probabilities
        pass through unscaled and temperature_applied=False is recorded.
        """
        raw = dict(raw)
        raw["temperature_applied"] = False
        probs = raw.get("probabilities")
        if not isinstance(probs, dict) or not probs:
            return raw
        try:
            from src.decision import calibration as _cal
            from src.decision.checkpoint_info import checkpoint_info

            temp = _cal.load_temperature()
            if not (temp and temp.get("fitted")):
                return raw
            rev = checkpoint_info().get("revision")
            art_rev = temp.get("checkpoint_revision")
            if not art_rev or not rev or art_rev != rev:
                # Calibration mismatch: never silently use stale scaling.
                raw["temperature_mismatch"] = True
                return raw
            n_opts = len(raw.get("options") or probs)
            if int(temp.get("n_options", n_opts)) != n_opts:
                raw["temperature_mismatch"] = True
                return raw
            scaled = _cal.apply_temperature(
                {str(k): float(v) for k, v in probs.items()},
                float(temp.get("temperature", 1.0)))
            raw["probabilities_raw"] = probs
            raw["probabilities"] = scaled
            choice = str(raw.get("proposed_action") or "")
            api_choice = {"forward": "forward", "turn_left": "left",
                          "turn_right": "right", "stop": "stop"}.get(choice)
            if api_choice in scaled:
                raw["answer_confidence"] = scaled[api_choice]
            raw["temperature_applied"] = True
            raw["temperature"] = temp.get("temperature")
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        return raw


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
                       source: str | None = None,
                       mode: str = "constrained",
                       safety_dict: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Eligibility -> policy -> confidence gate -> safety -> audit record.

    Full chain: Stage-6 named state -> Action Eligibility Layer ->
    Laya (constrained subset or unconstrained all-four) -> confidence
    gate -> deterministic safety layer -> execution record.

    `mode`: "constrained" (production: Laya is offered ONLY eligible
    actions; single-eligible STOP skips inference deterministically) or
    "unconstrained" (diagnostics: Laya sees all four; the raw preference
    is recorded honestly and the constrained projection is computed
    deterministically from the raw distribution). The mode is recorded
    on every result ("unconstrained_laya" / "constrained_laya").

    `source` tags action origin: "laya" | "replay-laya" | "jev" (legacy)
    | "replay-jev" (legacy) | "dqn" | "fallback" | "manual".
    When omitted it defaults to the policy mode, or "fallback" when no
    decision was produced and the safe fallback applies. Manual actions
    must pass source="manual" explicitly so they never contaminate
    autonomy metrics.

    Raw model behavior is NEVER rewritten: raw_* fields carry the
    unconstrained model output; constrained_* fields carry the
    eligibility-bound result. System-level mitigation stays
    distinguishable from model-level learning.
    """
    import datetime as _dt
    t0 = time.perf_counter()
    mode = str(mode or "constrained").lower()
    if mode not in ("constrained", "unconstrained"):
        raise ValueError("mode must be 'constrained' or 'unconstrained'.")
    mode_tag = ("constrained_laya" if mode == "constrained"
                else "unconstrained_laya")

    # --- Action Eligibility Layer (deterministic, pre-decision) ---
    elig_info: Dict[str, Any] = {}
    try:
        from src.decision.action_eligibility import (
            categorize_frame, compute_eligible_actions)

        elig = compute_eligible_actions(named_state, safety_dict)
        elig_info = {
            "eligible_actions": list(elig["eligible"]),
            "ineligible_actions": dict(elig["ineligible"]),
            "eligibility_emergency": bool(elig["emergency"]),
            "eligibility_thresholds": dict(elig["thresholds"]),
            "category": categorize_frame(named_state, safety_dict),
        }
    except (ValueError, TypeError, KeyError, AttributeError):
        elig_info = {"eligible_actions": None, "eligibility_error":
                     "clearances unavailable; eligibility skipped"}

    eligible = elig_info.get("eligible_actions")
    raw_rec: Dict[str, Any] = {}
    con_rec: Dict[str, Any] = {}
    laya_skipped = False
    skip_reason = None

    def _gate_view(d: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": d.get("proposed_action"),
                "confidence": d.get("confidence"),
                "answer_confidence": d.get("answer_confidence"),
                "probabilities": d.get("probabilities"),
                "status": d.get("status")}

    if (policy.mode == "laya" and mode == "constrained"
            and isinstance(eligible, list)):
        if eligible == ["stop"]:
            # Deterministic fast path: nothing to choose; never call the
            # model on the live path with a single STOP option, and never
            # burn ~9 s of CPU inference before an emergency STOP.
            laya_skipped = True
            skip_reason = ("single eligible action (STOP): deterministic; "
                           "no Laya inference on the live path")
            con_rec = {"action": "stop",
                       "proposed_action": "stop", "confidence": 1.0,
                       "answer_confidence": 1.0,
                       "probabilities": {"stop": 1.0},
                       "status": "SKIPPED-DETERMINISTIC",
                       "gated_action": "stop",
                       "gate": ("eligible set is STOP-only "
                                f"({'emergency' if elig_info.get('eligibility_emergency') else 'no safe direction'}); "
                                "deterministic STOP, no inference"),
                       "gate_metric": "n/a (deterministic)",
                       "gate_threshold": None,
                       "latency_ms": 0.0}
            prop = dict(con_rec)
        else:
            con = policy.decide(named_state, eligible=eligible)
            con_rec = _gate_view(con)
            con_rec.update({"status": con.get("status"),
                            "gate": con.get("gate"),
                            "gate_metric": con.get("gate_metric"),
                            "gate_threshold": con.get("gate_threshold"),
                            "latency_ms": con.get("latency_ms"),
                            "temperature_applied": con.get(
                                "temperature_applied", False)})
            prop = con
    elif policy.mode == "laya" and mode == "unconstrained":
        raw = policy.decide(named_state, eligible=None)
        raw_rec = _gate_view(raw)
        raw_rec.update({"status": raw.get("status"), "gate": raw.get("gate"),
                        "latency_ms": raw.get("latency_ms"),
                        "temperature_applied": raw.get(
                            "temperature_applied", False)})
        # Deterministic projection of the raw distribution onto the
        # eligible set (NOT a second model call).
        if isinstance(eligible, list) and isinstance(
                raw.get("probabilities"), dict):
            probs = raw["probabilities"]
            proj = max(eligible, key=lambda c: (
                float(probs.get(c, -1.0)),
                1 if c == "stop" else 0))
            con_rec = {"action": {"forward": "forward", "left": "turn_left",
                                  "right": "turn_right",
                                  "stop": "stop"}[proj],
                       "projected_from": raw_rec.get("action"),
                       "projection": True}
        prop = raw
    else:
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
    try:
        from src.decision.checkpoint_info import decision_version_stamp
        stamp = decision_version_stamp()
    except (OSError, ValueError, AttributeError):
        stamp = {"model_id": prop.get("model"), "checkpoint_revision": None,
                 "device": prop.get("device"), "runtime_version": None,
                 "calibration_version": None,
                 "gate_threshold": prop.get("gate_threshold")}
    record = {"decision_model": policy.mode,
              "run_id": run_id,
              "source": resolved_source,
              "mode": mode_tag,
              "frame_id": frame_id,
              "proposed_action": prop.get("proposed_action"),
              "confidence": prop.get("confidence"),
              "answer_confidence": prop.get("answer_confidence"),
              "engine_confidence_raw": prop.get("laya_confidence_raw",
                                                prop.get("confidence")),
              "device": prop.get("device"),
              "probabilities": prop.get("probabilities"),
              "engine_status": prop.get("status"),
              "engine_error": prop.get("error"),
              "jev_status": prop.get("status"),  # legacy alias; prefer engine_status
              "jev_error": prop.get("error"),  # legacy alias; prefer engine_error
              "endpoint": prop.get("endpoint"),
              "model": prop.get("model"),
              "usage": prop.get("usage"),
              "gate": gate_note,
              "gate_metric": prop.get("gate_metric"),
              "gate_threshold": prop.get("gate_threshold"),
              "temperature_applied": prop.get("temperature_applied", False),
              "eligible_actions": elig_info.get("eligible_actions"),
              "ineligible_actions": elig_info.get("ineligible_actions"),
              "category": elig_info.get("category"),
              "raw_laya_action": (raw_rec.get("action")
                                  if raw_rec else None),
              "raw_laya_probabilities": (raw_rec.get("probabilities")
                                         if raw_rec else None),
              "raw_laya_confidence": (raw_rec.get("answer_confidence")
                                      if raw_rec else None),
              "raw_laya_status": (raw_rec.get("status") if raw_rec else None),
              "constrained_action": (con_rec.get("action")
                                     if con_rec else None),
              "laya_skipped": laya_skipped,
              "skip_reason": skip_reason,
              "model_id": stamp.get("model_id"),
              "checkpoint_revision": stamp.get("checkpoint_revision"),
              "runtime_version": stamp.get("runtime_version"),
              "calibration_version": stamp.get("calibration_version"),
              "safety_status": verdict["verdict"],
              "safety_reason": verdict["reason"],
              "executed_action": verdict["final_action"],
              "safety_override": verdict["verdict"] != "SAFE_TO_EXECUTE",
              "policy_latency_ms": prop.get("latency_ms"),
              "safety_latency_ms": safety_ms,
              "total_latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)}
    if policy.mode in ("jev", "laya"):
        default_log = (_default_laya_log_path() if policy.mode == "laya"
                       else _default_jev_log_path())
        append_jev_log({
            "run_id": run_id,
            "timestamp": _dt.datetime.fromtimestamp(
                prop.get("request_ts") or prop.get("timestamp") or time.time(),
                tz=_dt.timezone.utc).isoformat(),
            "frame_id": frame_id,
            "mode": mode_tag,
            "endpoint": prop.get("endpoint"),
            "model": prop.get("model"),
            "device": prop.get("device"),
            "request_status": prop.get("status"),
            "selected_action": prop.get("proposed_action"),
            "probabilities": prop.get("probabilities"),
            "confidence": prop.get("confidence"),
            "answer_confidence": prop.get("answer_confidence"),
            "engine_confidence_raw": prop.get("laya_confidence_raw"),
            "latency_ms": prop.get("latency_ms"),
            "safety_latency_ms": safety_ms,
            "fallback": gate_note,
            "error": prop.get("error"),
            "eligible_actions": elig_info.get("eligible_actions"),
            "category": elig_info.get("category"),
            "raw_laya_action": record.get("raw_laya_action"),
            "constrained_action": record.get("constrained_action"),
            "checkpoint_revision": stamp.get("checkpoint_revision"),
            "calibration_version": stamp.get("calibration_version"),
            "safety_status": verdict["verdict"],
            "executed_action": verdict["final_action"],
        }, log_path or default_log)
    return record


def _default_laya_log_path() -> str:
    from pathlib import Path as _P
    return str(_P(__file__).resolve().parent.parent.parent
               / "results" / "decisions" / "laya_requests.jsonl")
