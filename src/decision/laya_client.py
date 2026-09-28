"""Canonical local Laya transport (loopback Decisions API).

Single implementation of the Laya HTTP transport used by:
    policy_interface.LayaPolicy        (active decision backend)
    scripts/check_laya.py              (local gate probe)
    backend/services/laya_manager.py   (server lifecycle)

Contract (local ONLY):
    POST {endpoint}  {"model", "state", "questions"}
    endpoint = http://127.0.0.1:18001/v1/systemone (configurable)
    model    = "typed-decisions" (local checkpoint)

No cloud calls, no API keys. Laya answers are Jev-shaped:
answers.action = {choice, probabilities, confidence,
answer_confidence}. The gate uses `answer_confidence`
(selected-choice probability, Laya-native scale); raw `confidence`
is recorded but never gated. Neither is task accuracy.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_ENDPOINT = "http://127.0.0.1:18001/v1/systemone"
DEFAULT_HEALTH = "http://127.0.0.1:18001/health"
DEFAULT_MODEL = "typed-decisions"

VALID_ACTIONS = ("forward", "turn_left", "turn_right", "stop")
API_CHOICES = ("forward", "left", "right", "stop")
CHOICE_TO_ACTION = {"forward": "forward", "left": "turn_left",
                    "right": "turn_right", "stop": "stop"}


def load_laya_config() -> Dict[str, Any]:
    try:
        return json.loads((PROJECT_ROOT / "config" / "laya_config.json").read_text())
    except (OSError, ValueError):
        return {}


def resolve_endpoint() -> Tuple[str, str]:
    """(endpoint, source): explicit LAYA_ENDPOINT env, else config, else default."""
    env_ep = os.environ.get("LAYA_ENDPOINT", "").strip()
    if env_ep:
        return env_ep, "LAYA_ENDPOINT"
    cfg_ep = str(load_laya_config().get("endpoint", "") or "").strip()
    if cfg_ep:
        return cfg_ep, "config"
    return DEFAULT_ENDPOINT, "default"


def resolve_health_endpoint() -> str:
    env_ep = os.environ.get("LAYA_HEALTH_ENDPOINT", "").strip()
    if env_ep:
        return env_ep
    cfg = str(load_laya_config().get("health_endpoint", "") or "").strip()
    return cfg or DEFAULT_HEALTH


def resolve_model() -> str:
    env_model = os.environ.get("LAYA_MODEL", "").strip()
    if env_model:
        return env_model
    cfg_model = str(load_laya_config().get("model", "") or "").strip()
    return cfg_model or DEFAULT_MODEL


def build_question(eligible: list | tuple | None = None) -> Dict[str, Any]:
    """Choice question over all four actions, or ONLY the eligible subset.

    Constrained mode (production): `eligible` is the Action Eligibility
    Layer output, so in an emergency the model is offered {"stop"} only
    and can never propose FORWARD on the live path. Unconstrained mode
    (diagnostics): `eligible=None` offers all four and records the raw
    model preference honestly.
    """
    cfg_q = load_laya_config().get("question", {}) or {}
    instructions = str(cfg_q.get("instructions", "") or
                       "Select one action from the structured state.")
    options = list(eligible) if eligible else list(API_CHOICES)
    for o in options:
        if o not in API_CHOICES:
            raise ValueError(f"unknown eligible option {o!r}")
    if not options:
        raise ValueError("eligible options must not be empty (use ['stop'])")
    criteria = {}
    try:
        for k in options:
            v = dict(cfg_q.get("criteria", {}) or {}).get(k, "")
            if v:
                criteria[k] = str(v)
    except (TypeError, ValueError):
        criteria = {}
    if set(criteria) != set(options):
        raise ValueError(
            "laya question criteria must define exactly "
            f"{sorted(options)} for this call")
    out = {"type": "choice", "instructions": instructions, "criteria": criteria}
    if eligible:
        out["eligible_options"] = list(options)
    return out


def detect_device() -> str:
    """Actual torch device (never claimed without probing)."""
    try:
        import torch

        return "CUDA" if torch.cuda.is_available() else "CPU"
    except Exception:
        return "UNKNOWN"


def service_status() -> Dict[str, Any]:
    """Local availability probe (health endpoint, no inference)."""
    from backend.services import laya_manager

    endpoint, ep_source = resolve_endpoint()
    health = laya_manager.health()
    available = bool(health.get("healthy"))
    out = {"available": available, "backend": "LOCAL",
           "endpoint": endpoint, "endpoint_source": ep_source,
           "model": resolve_model(), "device": detect_device(),
           "timeout_s": float(load_laya_config().get("timeout_s", 30.0))}
    if not available:
        out["reason"] = health.get("reason", "Laya server not healthy")
    else:
        out["server"] = {k: health.get(k) for k in ("status", "loaded", "device")}
    return out


@dataclass
class LayaDecisionResult:
    """Stable internal decision record (never carries credentials)."""
    proposed_action: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None
    answer_confidence: Optional[float] = None
    latency_ms: float = 0.0
    status: str = "UNAVAILABLE"  # OK | UNAVAILABLE | FAILED | INVALID
    model: str = DEFAULT_MODEL
    timestamp: float = 0.0
    frame_id: Optional[str] = None
    error: Optional[str] = None
    endpoint: Optional[str] = None
    device: Optional[str] = None
    usage: Optional[Any] = None
    options: Optional[list] = None  # eligible subset offered (None = all four)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LayaDecisionSystem:
    """Thin client over the local Laya Decisions API (stdlib urllib)."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.cfg = dict(config or load_laya_config())
        self.timeout_s = float(self.cfg.get("timeout_s", 30.0))
        self.max_retries = int(self.cfg.get("max_retries", 1))
        self.model = str(self.cfg.get("model", "") or "").strip() or resolve_model()

    def decide(self, named_state: Dict[str, Any],
               frame_id: str | None = None,
               eligible: list | tuple | None = None) -> LayaDecisionResult:
        t0 = time.perf_counter()
        request_ts = time.time()
        status = service_status()
        endpoint = status["endpoint"]
        if not status["available"]:
            return LayaDecisionResult(
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                timestamp=request_ts, frame_id=frame_id, status="UNAVAILABLE",
                error=status.get("reason", "Laya unavailable"),
                model=self.model, endpoint=endpoint, device=status.get("device"))
        body = json.dumps({"model": self.model,
                           "state": dict(named_state),
                           "questions": {"action": build_question(
                               eligible)}}).encode()
        last_error = ""
        offered = list(eligible) if eligible else None
        for _ in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    endpoint, data=body,
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    payload = json.loads(resp.read().decode())
                return self._parse(payload, t0, request_ts, frame_id, endpoint,
                                   options=offered)
            except Exception as exc:  # noqa: BLE001 - recorded, not raised
                last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
        return LayaDecisionResult(
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            timestamp=request_ts, frame_id=frame_id, status="FAILED",
            error=last_error, model=self.model, endpoint=endpoint,
            device=detect_device(), options=offered)

    def _parse(self, payload: Dict[str, Any], t0: float,
               request_ts: float, frame_id: str | None,
               endpoint: str,
               options: list | None = None) -> LayaDecisionResult:
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        try:
            answer = payload.get("answers", {}).get("action", {})
            choice = str(answer.get("choice", "") or "").lower()
            conf = answer.get("confidence", None)
            conf_f = float(conf) if conf is not None else None
            aconf = answer.get("answer_confidence", None)
            aconf_f = float(aconf) if aconf is not None else None
        except (TypeError, ValueError, AttributeError):
            choice, conf_f, aconf_f = "", None, None
            answer = {}
        allowed = set(options) if options else set(API_CHOICES)
        if (choice not in allowed
                or choice not in API_CHOICES or aconf_f is None
                or not math.isfinite(aconf_f) or not (0.0 <= aconf_f <= 1.0)
                or (conf_f is not None and not math.isfinite(conf_f))):
            return LayaDecisionResult(
                latency_ms=ms, timestamp=request_ts, frame_id=frame_id,
                status="INVALID", model=self.model, endpoint=endpoint,
                device=detect_device(), options=options,
                error=(f"unusable response (choice={str(answer.get('choice', None))[:40]!r}, "
                       f"answer_confidence={str(answer.get('answer_confidence', None))[:40]!r}, "
                       f"offered={sorted(allowed)!r})"))
        probs = None
        raw_probs = answer.get("probabilities", None)
        if isinstance(raw_probs, dict):
            try:
                probs = {str(k): float(v) for k, v in raw_probs.items()
                         if math.isfinite(float(v))}
            except (TypeError, ValueError):
                probs = None
        if conf_f is not None and not (0.0 <= conf_f <= 1.0):
            conf_f = None
        return LayaDecisionResult(
            proposed_action=CHOICE_TO_ACTION[choice], probabilities=probs,
            confidence=conf_f, answer_confidence=aconf_f,
            latency_ms=ms, timestamp=request_ts, frame_id=frame_id,
            status="OK", model=str(payload.get("model", self.model)),
            endpoint=endpoint, device=detect_device(),
            usage=payload.get("usage", None), options=options)
