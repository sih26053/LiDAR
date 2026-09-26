"""In-memory decision record store (Phase 24).

Appended by simulation steps and PyBullet closed-loop runs; read by
GET /decision/current, /decision/history, /metrics/current. Process
memory only (prototype scope, like result_service). Empty until a real
decision executes -- never pre-filled.
"""

from __future__ import annotations

from typing import Any, Dict, List

_records: List[Dict[str, Any]] = []


def record(entry: Dict[str, Any]) -> Dict[str, Any]:
    _records.append(dict(entry))
    return _records[-1]


def current() -> Dict[str, Any] | None:
    return _records[-1] if _records else None


def history(limit: int = 50) -> List[Dict[str, Any]]:
    n = max(1, min(int(limit), 500))
    return _records[-n:]


def clear() -> None:
    _records.clear()


def count() -> int:
    return len(_records)
