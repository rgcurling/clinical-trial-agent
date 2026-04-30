"""
TrialMatch AI — FastAPI production service.

Startup:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4

Docker:
    docker-compose up
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load heavy singletons once at startup so requests don't cold-start."""
    logger.info("TrialMatch AI starting up...")

    # Pre-warm the trialmatch pipeline imports (Claude client initialised lazily)
    try:
        import sys, os as _os
        sys.path.insert(0, "/app/trialmatch")
        sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", "..", "trialmatch"))
        from pipeline.matcher import ClaudeMatcher
        app.state.matcher = ClaudeMatcher()
        logger.info("ClaudeMatcher ready")
    except Exception as exc:
        logger.warning(f"Could not pre-load matcher: {exc}")
        app.state.matcher = None

    logger.info("TrialMatch AI startup complete")
    yield
    logger.info("TrialMatch AI shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="TrialMatch AI",
        description=(
            "Match patient clinical notes to relevant clinical trials "
            "using multi-agent AI (Claude + GPT-4o critic)."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()
