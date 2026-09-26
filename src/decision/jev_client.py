"""Canonical Jev transport layer (OpenRouter Decisions API).

Single implementation of the Jev HTTP transport used by:
    jev_decision.JevDecisionSystem  (stable project interface)
    policy_interface.JevPolicy      (confidence gate + safety chain)
    scripts/check_jev.py            (live gate probe)

Contract (structured Decisions API ONLY):
    POST {endpoint}  {"model", "state", "questions"}
    model    = "typesafe/jev-1.13"   (never jev-latest here)
    endpoint = JEV_ENDPOINT env, else https://openrouter.ai/api/alpha/decisions
    key      = OPENROUTER_API_KEY env, else TYPESAFE_API_KEY env,
               else config.local_secrets.OPENROUTER_API_KEY (server-side only)

The key is sent as a Bearer header and is NEVER logged, persisted,
returned, or exposed to the frontend. Without a key the client reports
UNAVAILABLE ("Jev credentials not configured") and produces no decision.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "typesafe/jev-1.13"

# Repo-canonical actions (shared with safety/executor/DQN layers).
VALID_ACTIONS = ("forward", "turn_left", "turn_right", "stop")

# Decisions-API choice keys for the action question mapped to repo actions.
API_CHOICES = ("forward", "left", "right", "stop")
CHOICE_TO_ACTION = {"forward": "forward", "left": "turn_left",
                    "right": "turn_right", "stop": "stop"}

DEFAULT_QUESTION = {
    "instructions": "Which vehicle action should be selected based only on the supplied state?",
    "criteria": {
        "forward": "Forward movement is safe and useful.",
        "left": "Turning left is the safest useful action.",
        "right": "Turning right is the safest useful action.",
        "stop": "Stopping is the safest appropriate action.",
    },
}

PLACEHOLDER_KEY = "PASTE_YOUR_OPENROUTER_API_KEY_HERE"


def load_jev_config() -> Dict[str, Any]:
    try:
        return json.loads((PROJECT_ROOT / "config" / "jev_config.json").read_text())
    except (OSError, ValueError):
        return {}


def resolve_api_key() -> Tuple[str, str]:
    """Resolve (key, source-label) without ever exposing the value.

    Precedence: OPENROUTER_API_KEY env > TYPESAFE_API_KEY env >
    config.local_secrets.OPENROUTER_API_KEY. The placeholder counts as missing.
    """
    for var in ("OPENROUTER_API_KEY", "TYPESAFE_API_KEY"):
        val = os.environ.get(var, "").strip()
        if val and val != PLACEHOLDER_KEY:
            return val, var
    try:
        from config.local_secrets import OPENROUTER_API_KEY as file_key
        val = str(file_key or "").strip()
        if val and val != PLACEHOLDER_KEY:
            return val, "config.local_secrets"
    except (ImportError, AttributeError):
        pass
    return "", ""


def resolve_endpoint() -> Tuple[str, str]:
    """(endpoint, source): explicit JEV_ENDPOINT env, else the default."""
    env_ep = os.environ.get("JEV_ENDPOINT", "").strip()
    if env_ep:
        return env_ep, "JEV_ENDPOINT"
    return DEFAULT_ENDPOINT, "default"


def resolve_model() -> str:
    env_model = os.environ.get("JEV_MODEL", "").strip()
    if env_model:
        return env_model
    cfg_model = str(load_jev_config().get("model", "") or "").strip()
    return cfg_model or DEFAULT_MODEL


def build_question() -> Dict[str, Any]:
    cfg_q = load_jev_config().get("question", {}) or {}
    instructions = str(cfg_q.get("instructions", "") or
                       DEFAULT_QUESTION["instructions"])
    criteria = dict(DEFAULT_QUESTION["criteria"])
    try:
        criteria.update({k: str(v) for k, v in
                         dict(cfg_q.get("criteria", {}) or {}).items()
                         if k in API_CHOICES})
    except (TypeError, ValueError):
        pass
    return {"type": "choice", "instructions": instructions,
            "criteria": criteria}


def service_status() -> Dict[str, Any]:
    """Honest availability probe (no network call, no key exposure)."""
    cfg = load_jev_config()
    _, key_source = resolve_api_key()
    endpoint, ep_source = resolve_endpoint()
    if not key_source:
        return {"available": False,
                "reason": "Jev credentials not configured",
                "endpoint": endpoint, "endpoint_source": ep_source,
                "model": resolve_model()}
    return {"available": True, "key_source": key_source,
            "endpoint": endpoint, "endpoint_source": ep_source,
            "model": resolve_model(),
            "timeout_s": float(cfg.get("timeout_s", 8.0))}


@dataclass
class JevDecisionResult:
    """Stable internal decision record (never carries credentials)."""
    proposed_action: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None
    latency_ms: float = 0.0
    status: str = "UNAVAILABLE"  # OK | UNAVAILABLE | FAILED | INVALID
    model: str = DEFAULT_MODEL
    timestamp: float = 0.0
    frame_id: Optional[str] = None
    error: Optional[str] = None
    endpoint: Optional[str] = None
    usage: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class JevClient:
    """Thin client over the OpenRouter Decisions API (stdlib urllib)."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.cfg = dict(config or load_jev_config())
        self.timeout_s = float(self.cfg.get("timeout_s", 8.0))
        self.max_retries = int(self.cfg.get("max_retries", 1))
        self.model = str(self.cfg.get("model", "") or "").strip() or resolve_model()

    def decide(self, named_state: Dict[str, Any],
               frame_id: str | None = None) -> JevDecisionResult:
        t0 = time.perf_counter()
        request_ts = time.time()
        status = service_status()
        endpoint = status["endpoint"]
        if not status["available"]:
            return JevDecisionResult(
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                timestamp=request_ts, frame_id=frame_id, status="UNAVAILABLE",
                error=status["reason"], model=self.model, endpoint=endpoint)
        key, _ = resolve_api_key()
        body = json.dumps({"model": self.model,
                           "state": dict(named_state),
                           "questions": {"action": build_question()}}).encode()
        last_error = ""
        for _ in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    endpoint, data=body,
                    headers={"Content-Type": "application/json",
                             "Authorization": "Bearer " + key},
                    method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    payload = json.loads(resp.read().decode())
                return self._parse(payload, t0, request_ts, frame_id, endpoint)
            except Exception as exc:  # noqa: BLE001 - recorded, not raised
                last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
        return JevDecisionResult(
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            timestamp=request_ts, frame_id=frame_id, status="FAILED",
            error=last_error, model=self.model, endpoint=endpoint)

    def _parse(self, payload: Dict[str, Any], t0: float,
               request_ts: float, frame_id: str | None,
               endpoint: str) -> JevDecisionResult:
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        try:
            answer = payload.get("answers", {}).get("action", {})
            choice = str(answer.get("choice", "") or "").lower()
            conf = answer.get("confidence", None)
            conf_f = float(conf) if conf is not None else None
        except (TypeError, ValueError, AttributeError):
            choice, conf_f = "", None
            answer = {}
        if (choice not in API_CHOICES or conf_f is None
                or not math.isfinite(conf_f) or not (0.0 <= conf_f <= 1.0)):
            return JevDecisionResult(
                latency_ms=ms, timestamp=request_ts, frame_id=frame_id,
                status="INVALID", model=self.model, endpoint=endpoint,
                error=(f"unusable response (choice={str(answer.get('choice', None))[:40]!r}, "
                       f"confidence={str(answer.get('confidence', None))[:40]!r})"))
        probs = None
        raw_probs = answer.get("probabilities", None)
        if isinstance(raw_probs, dict):
            try:
                probs = {str(k): float(v) for k, v in raw_probs.items()
                         if math.isfinite(float(v))}
            except (TypeError, ValueError):
                probs = None
        return JevDecisionResult(
            proposed_action=CHOICE_TO_ACTION[choice], probabilities=probs,
            confidence=conf_f, latency_ms=ms, timestamp=request_ts,
            frame_id=frame_id, status="OK",
            model=str(payload.get("model", self.model)),
            endpoint=endpoint, usage=payload.get("usage", None))
