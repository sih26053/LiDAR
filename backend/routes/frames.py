"""GET /frames -- available replay frames from the frozen manifest."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.services import replay_service

logger = logging.getLogger("paradox.backend.routes.frames")
router = APIRouter()


@router.get("/frames")
def list_frames():
    try:
        frames = replay_service.list_frames()
    except Exception as exc:
        logger.exception("frame listing failed")
        return JSONResponse(
            status_code=500,
            content={"stage": "loading", "error_code": "DATA_LOAD_FAILED",
                     "message": f"Cannot list replay frames: {exc}"[:300], "frame_id": None},
        )
    return {
        "frames": [
            {"frame_id": f["frame_id"], "scene_id": f.get("scene_id"),
             "timestamp": f.get("timestamp"), "source": f.get("source"),
             "point_count": f.get("point_count")}
            for f in frames
        ],
        "count": len(frames),
    }
