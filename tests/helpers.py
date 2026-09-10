"""Small builders shared across the test suite."""

from __future__ import annotations

from app.perception.types import Detection, Frame


def make_detection(
    x: float,
    y: float,
    w: float = 40.0,
    h: float = 90.0,
    confidence: float = 0.9,
    class_id: int = 0,
    class_name: str = "person",
) -> Detection:
    """Build a detection from a top-left corner and a size."""
    return Detection(
        bbox=(x, y, x + w, y + h),
        confidence=confidence,
        class_id=class_id,
        class_name=class_name,
    )


def make_frames(count: int = 10, fps: float = 20.0):
    """A stream of frames with no image payload, for use with a mock detector."""
    return [
        Frame(index=i, timestamp=round(i / fps, 4), image=None, width=640, height=384)
        for i in range(count)
    ]
