"""FastAPI entry point (Step 2). App wiring only -- no ML logic here.

Run locally: ``uvicorn backend.app:app --host 127.0.0.1 --port 8000``
"""

from __future__ import annotations

import logging
import os

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import config as config_routes
from backend.routes import demo as demo_routes
from backend.routes import frames as frames_routes
from backend.routes import health as health_routes
from backend.routes import flow as flow_routes
from backend.routes import live as live_routes
from backend.routes import recordings as recordings_routes
from backend.routes import simulation as simulation_routes
from backend.routes import metrics as metrics_routes
from backend.routes import model as model_routes
from backend.routes import model as model_routes
from backend.routes import replay as replay_routes
from backend.routes import results as results_routes
from backend.routes import stream as stream_routes
from backend.routes import tracking as tracking_routes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    from backend.config import load_final_config

    load_final_config()  # fail fast with a clear error if frozen config is bad
    yield


app = FastAPI(
    title="Paradox Protocol Backend",
    description="Local-only SIH prototype backend: replay -> frozen deterministic pipeline (trained point classifier + tracker on board) -> structured results.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS: loopback defaults for local dev; production adds the real Vercel
# origin via CORS_ORIGINS (comma-separated) and/or FRONTEND_URL. Never "*".
_cors_extra = [
    o.strip().rstrip("/")
    for o in (os.environ.get("CORS_ORIGINS", "") + ","
              + os.environ.get("FRONTEND_URL", "")).split(",")
    if o.strip()
]
# Local-only SIH prototype: restrict CORS to loopback frontends.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        *_cors_extra,
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health_routes.router, tags=["health"])
app.include_router(flow_routes.router, tags=["flow"])
app.include_router(live_routes.router, tags=["live"])
app.include_router(recordings_routes.router, tags=["recordings"])
app.include_router(simulation_routes.router, tags=["simulation"])
app.include_router(config_routes.router, tags=["config"])
app.include_router(model_routes.router, tags=["model"])
app.include_router(frames_routes.router, tags=["frames"])
app.include_router(replay_routes.router, tags=["replay"])
app.include_router(stream_routes.router, tags=["stream"])
app.include_router(tracking_routes.router, tags=["tracking"])
app.include_router(results_routes.router, tags=["results"])
app.include_router(metrics_routes.router, tags=["metrics"])
app.include_router(model_routes.router, tags=["model"])
app.include_router(demo_routes.router, tags=["demo"])
