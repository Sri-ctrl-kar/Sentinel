"""Shared pytest fixtures for the Sentinel test suite."""

from __future__ import annotations

import os
import sys

import pytest

# Make the repository root importable without requiring an editable install.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from helpers import make_detection, make_frames  # noqa: E402


@pytest.fixture
def detection_factory():
    return make_detection


@pytest.fixture
def blank_frames():
    """A short stream of frames with no image payload (detector is mocked)."""
    return make_frames


@pytest.fixture(scope="session")
def demo_video(tmp_path_factory):
    """Render the synthetic demo clip once per test session."""
    pytest.importorskip("cv2")
    scripts = os.path.join(ROOT, "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from generate_demo_video import generate

    path = str(tmp_path_factory.mktemp("video") / "demo.mp4")
    return generate(path, width=640, height=384, fps=20, seconds=4.0)
