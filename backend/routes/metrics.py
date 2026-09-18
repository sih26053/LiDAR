"""GET /metrics/{frame_id} -- measured metrics only, null when unavailable."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.services import metrics_service, result_service

router = APIRouter()


@router.get("/metrics/{frame_id}")
def get_metrics(frame_id: str):
    result = result_service.get_result(frame_id)
    if not result:
        return JSONResponse(
            status_code=404,
            content={"stage": "serialization", "error_code": "RESULT_NOT_FOUND",
                     "message": f"No stored result for frame {frame_id}. Run POST /replay/run first.",
                     "frame_id": frame_id},
        )
    return metrics_service.metrics_from_result(result)
