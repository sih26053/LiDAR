"""GET /health -- API liveness, never a claim about pipeline health."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "paradox-protocol-backend"}
