"""Incident reasoner providers.

This is the only package in Sentinel allowed to import a model SDK, and only
the module that needs one does. :mod:`app.intelligence.providers.mock` imports
nothing beyond the standard library, so the whole intelligence layer — prompt,
schema, grounding, CLI — is testable with no network and no credentials.
"""

from __future__ import annotations

__all__ = ["mock"]
