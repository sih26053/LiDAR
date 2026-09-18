"""Stable internal INPUT contract (Step 0.1)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class InputFrame(BaseModel):
    """One replay frame reference. Strongly typed; adapted to this repo.

    ``frame_id`` is the nuScenes sample token. ``source_path`` points at the
    bundled processed ``.npy`` when known; ``replay_reference`` mirrors the
    manifest/metadata row. ``semantic_source`` is informational only.
    """

    frame_id: str = Field(..., min_length=1)
    scene_id: str | None = None
    timestamp: float | None = None
    source_path: str | None = None
    replay_reference: str | None = None
    semantic_source: str | None = None
