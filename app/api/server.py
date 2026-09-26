"""The FastAPI application (M0.9).

A thin service around the validated core. Its whole job is transport: accept a
video, start the existing pipeline, and serialise what the existing objects
already produce.

Deliberately absent: a database, a queue, a worker pool, a cache server, an
auth layer. M0.9 is single-process and in-memory, and an analysis lives until
the process does.

Run it with::

    pip install -r requirements-api.txt
    uvicorn app.api.server:app --reload

FastAPI and Uvicorn are optional dependencies. Nothing in ``app/`` outside this
package imports them, so the CLI entry points and the whole test suite keep
working on a machine that has never installed them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import analysis, incident
from .schemas import HealthResponse
from .store import AnalysisStore

API_VERSION = "0.9"
SERVICE_NAME = "sentinel-api"

TITLE = "Sentinel API"
DESCRIPTION = (
    "Video → perception → events → memory → reasoning → incident intelligence, "
    "over HTTP. An adapter: every number this service returns was computed by "
    "Sentinel's validated core, not here. Risk scores are ordinal engineering "
    "signals, not probabilities."
)

#: Where a local frontend dev server usually lives. CORS is on by default
#: because the M0.9 frontend is a separate origin; override with
#: SENTINEL_API_CORS_ORIGINS (comma-separated, or "*").
DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)

ENV_CORS_ORIGINS = "SENTINEL_API_CORS_ORIGINS"
ENV_UPLOAD_DIR = "SENTINEL_API_UPLOAD_DIR"


@dataclass
class ApiSettings:
    """Service configuration, from the environment. No secrets here."""

    cors_origins: List[str] = field(
        default_factory=lambda: list(DEFAULT_CORS_ORIGINS)
    )
    upload_dir: Optional[str] = None

    @property
    def cors_enabled(self) -> bool:
        return bool(self.cors_origins)

    @classmethod
    def from_env(cls, environ: Optional[dict] = None) -> "ApiSettings":
        env = os.environ if environ is None else environ
        raw = env.get(ENV_CORS_ORIGINS)
        if raw is None:
            origins = list(DEFAULT_CORS_ORIGINS)
        else:
            origins = [o.strip() for o in raw.split(",") if o.strip()]
        return cls(cors_origins=origins, upload_dir=env.get(ENV_UPLOAD_DIR))

    def describe(self) -> str:
        return (
            f"cors={'on' if self.cors_enabled else 'off'} "
            f"origins={len(self.cors_origins)} "
            f"upload_dir={self.upload_dir or '(temporary)'}"
        )


def create_app(
    settings: Optional[ApiSettings] = None,
    store: Optional[AnalysisStore] = None,
) -> FastAPI:
    """Build an application instance.

    Both arguments are injectable so a test can supply its own store and origins
    without touching the environment.
    """
    settings = settings if settings is not None else ApiSettings.from_env()

    @asynccontextmanager
    async def lifespan(instance: FastAPI):
        yield
        # Uploaded videos are temporary by design: nothing survives the process.
        instance.state.store.shutdown()

    app = FastAPI(
        title=TITLE,
        description=DESCRIPTION,
        version=API_VERSION,
        lifespan=lifespan,
    )
    app.state.settings = settings
    # `store is not None`, not `store or ...`: an empty AnalysisStore is falsy
    # (it defines __len__), and `or` would quietly replace an injected one.
    app.state.store = (
        store if store is not None else AnalysisStore(upload_dir=settings.upload_dir)
    )

    if settings.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(analysis.router)
    app.include_router(incident.router)

    @app.get("/api/health", response_model=HealthResponse, tags=["service"])
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service=SERVICE_NAME,
            api_version=API_VERSION,
            analyses=len(app.state.store),
        )

    return app


#: The ASGI application uvicorn loads.
app = create_app()
