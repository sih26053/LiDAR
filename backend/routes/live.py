"""WebSocket live stream (GAP 8): GET/WS /ws/live.

Streams the actual runtime snapshot from backend.services.pybullet_live
at ~2 Hz. Payload carries live values only -- never credentials,
never fabricated numbers. When the live loop is idle the socket sends
{"system_status": "IDLE", ...} heartbeats instead of fake frames.
"""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    from backend.services import pybullet_live

    await websocket.accept()
    try:
        while True:
            snap = pybullet_live.snapshot()
            if snap is None:
                st = pybullet_live.status()
                await websocket.send_json({
                    "frame_id": None,
                    "timestamp": time.time(),
                    "simulator": "pybullet",
                    "system_status": "IDLE",
                    "mode": st.get("mode", "idle"),
                    "message": "Live loop idle; POST /simulation/live/start first.",
                })
            else:
                await websocket.send_json(snap)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
