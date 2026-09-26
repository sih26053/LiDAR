"""Jev decision node (stable project interface).

Transport lives in src/decision/jev_client.py (canonical); this module
keeps the established dict-based interface used by policy_interface,
check_jev, and backend routes. API key comes ONLY from env or
config.local_secrets (server-side); it is sent as a Bearer header and
never logged, persisted, or returned.
"""

from __future__ import annotations

from typing import Any, Dict

from src.decision.jev_client import (
    API_CHOICES,
    CHOICE_TO_ACTION,
    DEFAULT_MODEL,
    DEFAULT_QUESTION,
    VALID_ACTIONS,
    JevClient,
    build_question,
    load_jev_config,
    resolve_api_key,
    resolve_endpoint,
    resolve_model,
    service_status,
)

__all__ = ["VALID_ACTIONS", "API_CHOICES", "CHOICE_TO_ACTION",
           "DEFAULT_MODEL", "DEFAULT_QUESTION", "load_jev_config",
           "service_status", "JevDecisionSystem"]


class JevDecisionSystem:
    """Dict-interface wrapper over the canonical JevClient."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self._client = JevClient(config=config)
        self.cfg = self._client.cfg
        self.timeout_s = self._client.timeout_s
        self.max_retries = self._client.max_retries
        self.model = self._client.model

    def decide(self, named_state: Dict[str, Any]) -> Dict[str, Any]:
        rec = self._client.decide(named_state)
        return {"proposed_action": rec.proposed_action,
                "confidence": rec.confidence,
                "probabilities": rec.probabilities,
                "latency_ms": rec.latency_ms,
                "request_ts": rec.timestamp,
                "status": rec.status,
                "error": rec.error,
                "endpoint": rec.endpoint,
                "model": rec.model,
                "usage": rec.usage}
