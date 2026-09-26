"""HTTP API layer (M0.9).

An adapter around the validated M0.1-M0.8 core. It runs the existing pipeline,
reads the existing objects, and serialises their existing ``to_dict()``
payloads; it contains no perception, tracking, event, calibration, risk or
grounding logic of its own.

FastAPI is an optional dependency (``requirements-api.txt``), so this package
is imported lazily: ``import app.api`` costs nothing until something asks for
:func:`app.api.server.create_app`. The store and the runner have no web
dependency at all and can be used directly from a script.

    from app.api.server import create_app
    app = create_app()

Layering: nothing in ``app/`` outside this package may import it — asserted by
``tests/test_memory_layering.py``.
"""

from __future__ import annotations

from .store import (
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    STATUSES,
    AnalysisRecord,
    AnalysisStore,
    Progress,
)

__all__ = [
    "STATUSES",
    "STATUS_COMPLETE",
    "STATUS_FAILED",
    "STATUS_QUEUED",
    "STATUS_RUNNING",
    "AnalysisRecord",
    "AnalysisStore",
    "Progress",
    "create_app",
]


def __getattr__(name: str):
    """Import the FastAPI app factory only when it is actually asked for."""
    if name == "create_app":
        from .server import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
