"""In-memory result store (Step 22). Prototype scope: process memory only."""

from __future__ import annotations

from typing import Any, Dict

_results: Dict[str, Dict[str, Any]] = {}
_last_frame_id: str | None = None
_last_status: str | None = None


def store_result(frame_id: str, payload: Dict[str, Any], status: str = "success") -> None:
    global _last_frame_id, _last_status
    _results[str(frame_id)] = payload
    _last_frame_id = str(frame_id)
    _last_status = str(status)


def get_result(frame_id: str) -> Dict[str, Any] | None:
    return _results.get(str(frame_id))


def last_state() -> Dict[str, Any]:
    return {"last_frame_id": _last_frame_id, "last_status": _last_status}


def clear() -> None:
    global _last_frame_id, _last_status
    _results.clear()
    _last_frame_id = None
    _last_status = None
