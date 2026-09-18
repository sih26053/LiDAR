"""POST /replay/load, POST /replay/run, POST /replay/run-sequence.

Thin: validate request -> call service -> return response.
No importance/resolution/map code lives here. No tuning parameters
are accepted: the frontend cannot change frozen weights/thresholds.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.services import pipeline_service, replay_service, result_service
from backend.services.pipeline_service import PipelineStageError
from backend.services.replay_service import DataLoadError, FrameNotFoundError

logger = logging.getLogger("paradox.backend.routes.replay")
router = APIRouter()


class LoadRequest(BaseModel):
    frame_id: str = Field(..., min_length=1)


class RunRequest(BaseModel):
    frame_id: str = Field(..., min_length=1)
    save_output: bool = False
    include_visualization: bool = False
    max_map_cells: int | None = Field(default=None, ge=1, le=5000)


class SequenceRequest(BaseModel):
    frame_ids: list[str] | None = None
    start_frame: str | None = None
    end_frame: str | None = None
    scene_id: str | None = None
    max_map_cells: int | None = Field(default=None, ge=1, le=5000)


def _err(status: int, stage: str, code: str, message: str, frame_id=None):
    logger.warning("frame=%s stage=%s status=error code=%s", frame_id, stage, code)
    return JSONResponse(status_code=status,
                        content={"stage": stage, "error_code": code,
                                 "message": message[:500], "frame_id": frame_id})


@router.post("/replay/load")
def replay_load(req: LoadRequest):
    try:
        entry = replay_service.get_frame_entry(req.frame_id)
        points, _ = replay_service.load_frame_points(req.frame_id)
    except FrameNotFoundError as exc:
        return _err(404, "input", "FRAME_NOT_FOUND", str(exc), req.frame_id)
    except DataLoadError as exc:
        return _err(500, "loading", "DATA_LOAD_FAILED", str(exc), req.frame_id)
    return {"frame_id": entry["frame_id"], "scene_id": entry.get("scene_id"),
            "timestamp": entry.get("timestamp"), "point_count": int(len(points)),
            "load_status": "loaded"}


@router.post("/replay/run")
def replay_run(req: RunRequest):
    t0 = time.perf_counter()
    try:
        result = pipeline_service.run_frame(req.frame_id, max_map_cells=req.max_map_cells)
    except PipelineStageError as exc:
        status = 404 if exc.code == "FRAME_NOT_FOUND" else 500
        result_service.store_result(req.frame_id, {}, status="failed")
        return _err(status, exc.stage, exc.code, str(exc), req.frame_id)
    except Exception as exc:
        logger.exception("frame=%s unexpected pipeline failure", req.frame_id)
        return _err(500, "mapping", "MAPPING_FAILED", f"Unexpected failure: {exc}", req.frame_id)
    api_ms = (time.perf_counter() - t0) * 1000.0
    result_service.store_result(req.frame_id, result, status="success")
    logger.info("frame=%s stage=api status=success api_ms=%.1f", req.frame_id, api_ms)
    if req.save_output:
        _save_output_csv(req.frame_id, result)
    return {"result": result, "cache_hit": False}


@router.post("/replay/run-sequence")
def replay_run_sequence(req: SequenceRequest):
    ids = list(req.frame_ids or [])
    if not ids and req.start_frame:
        try:
            all_frames = [f["frame_id"] for f in replay_service.list_frames()]
        except DataLoadError as exc:
            return _err(500, "loading", "DATA_LOAD_FAILED", str(exc), None)
        if req.scene_id:
            all_frames = [f["frame_id"] for f in replay_service.list_frames()
                          if f.get("scene_id") == req.scene_id]
        try:
            i0 = all_frames.index(req.start_frame)
            i1 = all_frames.index(req.end_frame) if req.end_frame else i0
        except ValueError as exc:
            return _err(404, "input", "FRAME_NOT_FOUND", f"Sequence boundary not found: {exc}", None)
        ids = all_frames[min(i0, i1): max(i0, i1) + 1]
    if not ids:
        return _err(400, "input", "INVALID_REQUEST",
                    "Provide frame_ids or start_frame (and optional end_frame/scene_id).", None)
    outputs, errors = [], []
    for fid in ids:
        try:
            res = pipeline_service.run_frame(fid, max_map_cells=req.max_map_cells)
            result_service.store_result(fid, res, status="success")
            outputs.append(res)
        except PipelineStageError as exc:
            errors.append({"frame_id": fid, "stage": exc.stage, "error_code": exc.code})
    return {"results": outputs, "errors": errors,
            "ran": len(outputs), "requested": len(ids)}


def _save_output_csv(frame_id: str, result: dict) -> None:
    try:
        import pandas as pd

        from backend.services.pipeline_service import PROJECT_ROOT

        out = PROJECT_ROOT / "results" / "final" / "simulation_outputs"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result.get("map_cells", [])).to_csv(out / f"backend_map_{frame_id}.csv", index=False)
    except Exception:
        logger.exception("frame=%s saving backend map csv failed", frame_id)
