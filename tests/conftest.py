"""Shared pytest fixtures for the Sentinel test suite."""

from __future__ import annotations

import json
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


@pytest.fixture(scope="session")
def rendered_clip():
    """The benchmark clip, rendered on demand by the project's own script.

    The annotations are checked in; the video is not — `data/clips/*.mp4` is
    gitignored because it is reproducible, and the README renders it with:

        python scripts/generate_demo_video.py \
            -o data/clips/rendered_bay.mp4 --annotations data/clips/rendered_bay.json

    So a fresh clone, a fresh container or a machine that has never run that
    command has the ground truth and no video. Render it here with the same
    script and its own defaults — the ones that command uses — rather than
    letting the evaluator fail deep inside on a missing file. The annotations
    are never rewritten: they are the committed ground truth.
    """
    pytest.importorskip("cv2")
    annotation = os.path.join(ROOT, "data", "clips", "rendered_bay.json")
    if not os.path.exists(annotation):  # pragma: no cover - checked in
        pytest.skip("the benchmark clip annotations are missing")

    with open(annotation, encoding="utf-8") as handle:
        video = json.load(handle)["video"]
    path = os.path.join(os.path.dirname(annotation), video)

    if not os.path.exists(path):
        scripts = os.path.join(ROOT, "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        from generate_demo_video import generate

        generate(path)
    return path
