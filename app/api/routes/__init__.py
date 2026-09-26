"""API route modules.

Each module owns one group of endpoints and imports the validated core rather
than reimplementing any part of it.
"""

from __future__ import annotations

from . import analysis, incident

__all__ = ["analysis", "incident"]
