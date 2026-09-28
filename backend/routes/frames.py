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
    out = [
        {"frame_id": f["frame_id"], "scene_id": f.get("scene_id"),
         "timestamp": f.get("timestamp"), "source": f.get("source"),
         "point_count": f.get("point_count"), "origin": "nuscenes",
         "run_id": None, "scenario": None}
        for f in frames
    ]
    # Recorded live frames (SQLite) appended with explicit origin;
    # nuScenes entries first, never mixed silently.
    try:
        from backend.services import store

        for f in store.list_frames(limit=5000):
            out.append({"frame_id": f["frame_id"],
                        "scene_id": f.get("scenario") or f.get("run_id"),
                        "timestamp": f.get("timestamp"),
                        "source": f.get("points_path"),
                        "point_count": f.get("point_count"),
                        "origin": "live-recorded",
                        "run_id": f.get("run_id"),
                        "scenario": f.get("scenario")})
    except Exception:
        logger.exception("recorded frame listing failed (nuScenes list preserved)")
    return {"frames": out, "count": len(out),
            "nuscenes": len(frames), "live_recorded": len(out) - len(frames)}
