"""GET /demo/status -- actual backend/demo state, nothing fabricated."""

from __future__ import annotations

from fastapi import APIRouter

from backend.config import load_final_config
from backend.services import replay_service, result_service

router = APIRouter()


@router.get("/demo/status")
def demo_status():
    try:
        load_final_config()
        config_ok = True
    except Exception:
        config_ok = False
    try:
        n_frames = len(replay_service.list_frames())
        replay_ok = n_frames > 0
    except Exception:
        n_frames = 0
        replay_ok = False
    last = result_service.last_state()
    return {"backend": "running", "configuration_loaded": config_ok,
            "replay_available": replay_ok, "available_frames": n_frames,
            "last_frame_id": last.get("last_frame_id"),
            "last_status": last.get("last_status")}
