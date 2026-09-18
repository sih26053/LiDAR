"""GET /results/{frame_id} -- latest successful result or 404."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.services import result_service

router = APIRouter()


@router.get("/results/{frame_id}")
def get_result(frame_id: str):
    result = result_service.get_result(frame_id)
    if not result:
        return JSONResponse(
            status_code=404,
            content={"stage": "serialization", "error_code": "RESULT_NOT_FOUND",
                     "message": f"No stored result for frame {frame_id}. Run POST /replay/run first.",
                     "frame_id": frame_id},
        )
    return {"result": result}
