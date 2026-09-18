"""GET /config -- safe frozen-config metadata (no secrets, no tuning knobs)."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.config import ConfigInvalidError, ConfigNotFoundError, config_info

logger = logging.getLogger("paradox.backend.routes.config")
router = APIRouter()


@router.get("/config")
def get_config():
    try:
        return config_info()
    except (ConfigNotFoundError, ConfigInvalidError) as exc:
        logger.error("config load failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"stage": "input", "error_code": "CONFIG_NOT_FOUND",
                     "message": str(exc)[:300], "frame_id": None},
        )
