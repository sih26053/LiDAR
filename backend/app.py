"""FastAPI entry point (Step 2). App wiring only -- no ML logic here.

Run locally: ``uvicorn backend.app:app --host 127.0.0.1 --port 8000``
"""

from __future__ import annotations

import logging

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import config as config_routes
from backend.routes import demo as demo_routes
from backend.routes import frames as frames_routes
from backend.routes import health as health_routes
from backend.routes import metrics as metrics_routes
from backend.routes import replay as replay_routes
from backend.routes import results as results_routes

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
    description="Local-only SIH prototype backend: replay -> frozen ML pipeline -> structured results.",
    version="0.1.0",
    lifespan=lifespan,
)

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
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health_routes.router, tags=["health"])
app.include_router(config_routes.router, tags=["config"])
app.include_router(frames_routes.router, tags=["frames"])
app.include_router(replay_routes.router, tags=["replay"])
app.include_router(results_routes.router, tags=["results"])
app.include_router(metrics_routes.router, tags=["metrics"])
app.include_router(demo_routes.router, tags=["demo"])
