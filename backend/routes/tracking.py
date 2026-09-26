"""Tracking endpoints: scene list + POST /tracking/run.

Runs the mapping pipeline over the requested frames (manifest replay frames
and/or consecutive raw scene samples), clusters dynamic-class cells into
per-frame detections, and associates them into persistent tracks with
measured velocity. Association is refused across scene boundaries or gaps
> MAX_DT_S (fresh tracks instead) — see src/tracker.py.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.services import pipeline_service, replay_service
from backend.services.pipeline_service import PipelineStageError

logger = logging.getLogger("paradox.backend.routes.tracking")
router = APIRouter()


class TrackingRequest(BaseModel):
    frame_ids: list[str] | None = None
    sample_tokens: list[str] | None = None
    scene_id: str | None = None
    scene_samples: int = Field(default=6, ge=2, le=20)
    semantic_mode: str = Field(default="annotation", pattern="^(annotation|model)$")


@router.get("/tracking/scenes")
def tracking_scenes():
    """Scenes available for consecutive-sample tracking (live from dataset)."""
    nusc = pipeline_service._nusc_dataset()
    out = []
    for sc in nusc.scene:
        out.append({
            "scene_id": sc.get("name"),
            "scene_token": sc.get("token"),
            "nbr_samples": int(sc.get("nbr_samples", 0)),
            "description": str(sc.get("description", ""))[:120],
        })
    return {"scenes": out, "count": len(out)}


def _scene_sequence(nusc, scene_id: str, count: int) -> list[str]:
    match = [s for s in nusc.scene if s.get("name") == scene_id]
    if not match:
        raise PipelineStageError("input", "FRAME_NOT_FOUND",
                                 f"Scene not found: {scene_id}")
    toks: list[str] = []
    cur = match[0].get("first_sample_token")
    while cur and len(toks) < count:
        toks.append(cur)
        nxt = nusc.get("sample", cur).get("next", "")
        cur = nxt or None
    return toks


@router.post("/tracking/run")
def tracking_run(req: TrackingRequest):
    from src.tracker import (
        CLUSTER_EPS_M,
        GATE_BASE_M,
        GATE_SPEED_M_S,
        MAX_DT_S,
        MAX_MISSES,
        cluster_detections,
        track_sequence,
    )

    jobs: list[tuple[str, str]] = []  # (kind, token)
    manifest_ids = set()
    try:
        manifest_ids = {f["frame_id"] for f in replay_service.list_frames()}
    except Exception:
        manifest_ids = set()
    if req.sample_tokens:
        jobs += [("sample", t) for t in req.sample_tokens]
    elif req.scene_id:
        nusc = pipeline_service._nusc_dataset()
        try:
            toks = _scene_sequence(nusc, req.scene_id, req.scene_samples)
        except PipelineStageError as exc:
            return JSONResponse(status_code=404, content={
                "stage": "input", "error_code": exc.code,
                "message": str(exc), "frame_id": None})
        jobs += [("sample", t) for t in toks]
    else:
        ids = list(req.frame_ids or [])
        if not ids:
            ids = [f["frame_id"] for f in replay_service.list_frames()]
        jobs += [("frame" if i in manifest_ids else "sample", i) for i in ids]

    frames = []
    errors = []
    for kind, tok in jobs:
        try:
            if kind == "frame":
                res = pipeline_service.run_frame(tok, semantic_mode=req.semantic_mode)
            else:
                res = pipeline_service.run_sample_token(tok, semantic_mode=req.semantic_mode)
        except PipelineStageError as exc:
            errors.append({"frame_id": tok, "stage": exc.stage,
                           "error_code": exc.code})
            continue
        frames.append({
            "frame_id": tok,
            "scene_id": res.get("scene_id"),
            "timestamp": res.get("timestamp"),
            "detections": cluster_detections(res.get("map_cells", [])),
            "map_cell_count": res.get("map_cell_count"),
            "total_latency_ms": (res.get("timing") or {}).get("total_latency_ms"),
        })
    if not frames:
        return JSONResponse(
            status_code=500,
            content={"stage": "tracking", "error_code": "TRACKING_FAILED",
                     "message": "No frames processed.", "frame_id": None})
    out = track_sequence(frames)
    out.update({
        "frames": frames,
        "frame_ids": [f["frame_id"] for f in frames],
        "semantic_mode": req.semantic_mode,
        "parameters": {
            "cluster_eps_m": CLUSTER_EPS_M,
            "gate_base_m": GATE_BASE_M,
            "gate_speed_m_s": GATE_SPEED_M_S,
            "max_misses": MAX_MISSES,
            "max_dt_s": MAX_DT_S,
        },
        "errors": errors,
    })
    return out
