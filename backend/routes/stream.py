"""Live-stream ingestion: POST /stream/push + POST /stream/simulate.

/stream/push accepts an externally supplied point cloud (JSON array of
[x, y, z, intensity]) and runs the SAME mapping pipeline with NO dataset
attached: annotation lookup is skipped (all-fallback in annotation mode),
while model mode runs the trained classifier. Nothing is replayed from
disk; the response states input_source explicitly.

/stream/simulate pushes a manifest replay frame through the same live
path server-side (clearly labelled simulated) to demonstrate the
streaming interface without external hardware.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.services import pipeline_service, replay_service
from backend.services.pipeline_service import PipelineStageError
from backend.services.replay_service import DataLoadError, FrameNotFoundError

logger = logging.getLogger("paradox.backend.routes.stream")
router = APIRouter()

MAX_STREAM_POINTS = 150_000


class StreamPushRequest(BaseModel):
    points: list[list[float]] = Field(..., min_length=100, max_length=MAX_STREAM_POINTS)
    timestamp: float | None = None
    scene_id: str | None = None
    semantic_mode: str = Field(default="model", pattern="^(annotation|model)$")
    max_map_cells: int | None = Field(default=None, ge=1, le=5000)


def _run_live(raw: np.ndarray, timestamp: float | None, scene_id: str | None,
              mode: str, max_cells: int | None, source: str):
    t_wall0 = time.perf_counter()
    fid = f"live-{int((timestamp or time.time() * 1e6))}"
    meta = {"frame_id": fid, "scene_id": scene_id or "live",
            "timestamp": float(timestamp or time.time() * 1e6),
            "source_path": source}
    return pipeline_service._execute(raw, None, meta, fid, mode, max_cells,
                                     t_wall0, live_input=True)


@router.post("/stream/push")
def stream_push(req: StreamPushRequest):
    try:
        raw = np.asarray(req.points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        return JSONResponse(status_code=400, content={
            "stage": "input", "error_code": "INVALID_REQUEST",
            "message": f"points must be numeric N x 4: {exc}", "frame_id": None})
    if raw.ndim != 2 or raw.shape[1] != 4:
        return JSONResponse(status_code=400, content={
            "stage": "input", "error_code": "INVALID_REQUEST",
            "message": f"points must be N x 4, got shape {raw.shape}",
            "frame_id": None})
    if not np.all(np.isfinite(raw)):
        return JSONResponse(status_code=400, content={
            "stage": "input", "error_code": "INVALID_REQUEST",
            "message": "points must be all finite (NaN/inf rejected).",
            "frame_id": None})
    try:
        result = _run_live(raw, req.timestamp, req.scene_id, req.semantic_mode,
                           req.max_map_cells, source="external-stream")
    except PipelineStageError as exc:
        status = 404 if exc.code in ("FRAME_NOT_FOUND", "MODEL_UNAVAILABLE") else 500
        return JSONResponse(status_code=status, content={
            "stage": exc.stage, "error_code": exc.code,
            "message": str(exc)[:500], "frame_id": None})
    result["stream"] = {"simulated": False, "note": "externally supplied live point cloud; no dataset attached"}
    return {"result": result}


@router.post("/stream/simulate")
def stream_simulate(body: dict):
    frame_id = str((body or {}).get("frame_id") or "")
    mode = str((body or {}).get("semantic_mode") or "model")
    if mode not in ("annotation", "model"):
        return JSONResponse(status_code=400, content={
            "stage": "input", "error_code": "INVALID_REQUEST",
            "message": "semantic_mode must be 'annotation' or 'model'.",
            "frame_id": frame_id or None})
    try:
        raw, meta = replay_service.load_frame_points(frame_id)
    except FrameNotFoundError as exc:
        return JSONResponse(status_code=404, content={
            "stage": "input", "error_code": "FRAME_NOT_FOUND",
            "message": str(exc)[:500], "frame_id": frame_id})
    except DataLoadError as exc:
        return JSONResponse(status_code=500, content={
            "stage": "loading", "error_code": "DATA_LOAD_FAILED",
            "message": str(exc)[:500], "frame_id": frame_id})
    try:
        result = _run_live(np.asarray(raw), meta.get("timestamp"),
                           meta.get("scene_id"), mode, None,
                           source=f"simulated-stream({frame_id})")
    except PipelineStageError as exc:
        return JSONResponse(status_code=500, content={
            "stage": exc.stage, "error_code": exc.code,
            "message": str(exc)[:500], "frame_id": frame_id})
    result["stream"] = {"simulated": True, "note": "replay frame pushed through the live path (no annotations); demonstration only"}
    return {"result": result}
