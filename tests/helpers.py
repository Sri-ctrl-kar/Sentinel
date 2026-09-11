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


# ---------------------------------------------------------------------------
# Deterministic synthetic scenarios
# ---------------------------------------------------------------------------
def run_scenario(
    frames,
    fps: float = 10.0,
    zones=None,
    close_at_end: bool = True,
    tracker_kwargs=None,
    **generator_kwargs,
):
    """Drive the real tracker + generator + temporal memory over scripted data.

    ``frames`` is a list of per-frame detection lists. No video, no model and
    no randomness is involved, so every assertion built on this is exactly
    reproducible.

    Returns the populated :class:`~app.memory.TemporalEventMemory`.
    """
    from app.events.generator import EventGenerator
    from app.memory import TemporalEventMemory
    from app.perception.trackers.byte_iou import ByteIoUTracker

    tracker = ByteIoUTracker(**(tracker_kwargs or {"min_hits": 1, "max_age": 5}))
    generator = EventGenerator(zones=zones, **generator_kwargs)
    memory = TemporalEventMemory()

    timestamp = 0.0
    tracks = []
    for index, detections in enumerate(frames):
        timestamp = round(index / fps, 4)
        tracks = tracker.update(detections, timestamp)
        memory.ingest_many(
            generator.process(
                tracks,
                timestamp=timestamp,
                frame_index=index,
                lost_tracks=tracker.lost_tracks,
            )
        )

    if close_at_end:
        memory.ingest_many(generator.flush(timestamp, tracks, frame_index=len(frames) - 1))
    return memory


def walk(start_x, step, count, y=100.0, **detection_kwargs):
    """A single object translating horizontally: one detection per frame."""
    return [
        [make_detection(start_x + step * i, y, **detection_kwargs)] for i in range(count)
    ]


def hold(x, count, y=100.0, **detection_kwargs):
    """A single object holding position: one detection per frame."""
    return [[make_detection(x, y, **detection_kwargs)] for _ in range(count)]
