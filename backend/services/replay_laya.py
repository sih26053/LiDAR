"""Replay-Laya entry points (active engine).

Thin wrappers over backend.services.replay_jev with engine="laya"
(source=replay-laya). No duplicated transport, pipeline, or safety
logic. Legacy Jev replay remains available via the engine="jev"
parameter for historical frames.
"""

from __future__ import annotations

from typing import Any, Dict, List

from backend.services import replay_jev as _rj


def run_replay_decision(frame_id: str, replay_run_id: str,
                        semantic_mode: str = "model",
                        log_path=None,
                        mode: str = "constrained") -> Dict[str, Any]:
    return _rj.run_replay_decision(frame_id, replay_run_id,
                                   semantic_mode=semantic_mode,
                                   log_path=log_path, engine="laya",
                                   mode=mode)


def run_batch(frame_ids: List[str] | None = None,
              run_id: str | None = None,
              limit: int | None = None,
              replay_run_id: str | None = None,
              semantic_mode: str = "model",
              mode: str = "constrained") -> Dict[str, Any]:
    return _rj.run_batch(frame_ids=frame_ids, run_id=run_id, limit=limit,
                         replay_run_id=replay_run_id,
                         semantic_mode=semantic_mode, engine="laya",
                         mode=mode)


def get_replay_history(frame_id: str) -> List[Dict[str, Any]]:
    return _rj.get_replay_history(frame_id, source="replay-laya")
