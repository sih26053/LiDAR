"""Consistent error contract (Step 0.4)."""

from __future__ import annotations

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    stage: str
    error_code: str
    message: str
    frame_id: str | None = None
