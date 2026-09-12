"""The M0.1 perception pipeline: video -> detection -> tracking -> events.

This module owns the wiring only. It knows nothing about YOLO, ByteTrack or
OpenCV specifics — it talks to the ``Detector``, ``Tracker`` and
``EventGenerator`` interfaces, which is what keeps the inference backend
swappable (CPU today, AMD ROCm or ONNX Runtime later).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .config import PipelineConfig
from .events.generator import EventGenerator
from .events.schema import Event
from .memory.temporal import TemporalEventMemory
from .perception.detector import Detector, create_detector
from .perception.tracker import Tracker, create_tracker
from .perception.types import Frame, Track
from .perception.video import VideoSource
from .storage.memory import EventMemory

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Everything a caller needs after a run.

    ``memory`` is the flat, append-only event log (the thing that gets written
    to disk). ``temporal`` is the queryable index built over the same events —
    the M0.2 temporal memory. They hold the same events; they differ in what
    you can ask them.
    """

    memory: EventMemory
    frames_processed: int
    elapsed_seconds: float
    metadata: Dict[str, Any]
    temporal: Optional[TemporalEventMemory] = None
    #: Raw detections returned by the backend, summed over every frame.
    detections_total: int = 0
    #: Confirmed track reports, summed over every frame. Divided by
    #: ``frames_processed`` this is the mean number of entities on screen.
    track_reports_total: int = 0
    #: Distinct track IDs the tracker created during the run.
    track_ids_created: int = 0
    #: Frames each entity was actually reported in. This is track persistence;
    #: it is not the number of events, which sampling thresholds also control.
    track_frames: Dict[str, int] = field(default_factory=dict)

    @property
    def events(self) -> List[Event]:
        return self.memory.all()

    @property
    def fps(self) -> float:
        return (
            self.frames_processed / self.elapsed_seconds
            if self.elapsed_seconds > 0
            else 0.0
        )


class PerceptionPipeline:
    """Runs detection + tracking + event generation over a frame stream.

    The components can be injected, which is what makes the pipeline testable
    without model weights: tests pass a ``MockDetector`` and get a fully real
    tracking and event path.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        detector: Optional[Detector] = None,
        tracker: Optional[Tracker] = None,
        event_generator: Optional[EventGenerator] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self._detector = detector
        self._tracker = tracker
        self._events = event_generator

    # ------------------------------------------------------------------
    # Lazily built components
    # ------------------------------------------------------------------
    @property
    def detector(self) -> Detector:
        if self._detector is None:
            self._detector = create_detector(
                self.config.detector,
                classes=self.config.classes,
                min_confidence=self.config.confidence,
                **self.config.detector_kwargs(),
            )
        return self._detector

    @property
    def tracker(self) -> Tracker:
        if self._tracker is None:
            self._tracker = create_tracker(
                self.config.tracker, **self.config.tracker_kwargs()
            )
        return self._tracker

    @property
    def event_generator(self) -> EventGenerator:
        if self._events is None:
            self._events = EventGenerator(
                sample_interval=self.config.sample_interval,
                movement_threshold=self.config.movement_threshold,
                stationary_threshold=self.config.stationary_threshold,
                stationary_duration=self.config.stationary_duration,
                zones=self.config.zones,
                source=self.config.video_path or None,
            )
        return self._events

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run_frames(
        self,
        frames: Iterable[Frame],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """Run the pipeline over any iterable of frames.

        Kept separate from :meth:`run` so tests (and, later, a live stream) can
        drive the pipeline without a file on disk.
        """
        memory = EventMemory(metadata=dict(metadata or {}))
        temporal = TemporalEventMemory(metadata=dict(metadata or {}))
        self.tracker.reset()
        self.event_generator.reset()

        started = time.perf_counter()
        frames_processed = 0
        detections_total = 0
        track_reports_total = 0
        track_ids: set = set()
        track_frames: Dict[str, int] = {}
        last_timestamp = 0.0
        last_frame_index: Optional[int] = None
        last_tracks: Sequence[Track] = []

        for frame in frames:
            detections = self.detector.detect(frame.image)
            tracks = self.tracker.update(detections, frame.timestamp)
            events = self.event_generator.process(
                tracks,
                timestamp=frame.timestamp,
                frame_index=frame.index,
                lost_tracks=self.tracker.lost_tracks,
            )
            memory.extend(events)
            temporal.ingest_many(events)
            frames_processed += 1
            detections_total += len(detections)
            track_reports_total += len(tracks)
            track_ids.update(t.track_id for t in tracks)
            for track in tracks:
                track_frames[track.entity_id] = (
                    track_frames.get(track.entity_id, 0) + 1
                )
            last_timestamp = frame.timestamp
            last_frame_index = frame.index
            last_tracks = tracks

        # Entities still on screen when the stream ends still deserve a
        # closing event, otherwise their duration is unknowable downstream.
        closing = self.event_generator.flush(
            last_timestamp, last_tracks, frame_index=last_frame_index
        )
        memory.extend(closing)
        temporal.ingest_many(closing)

        elapsed = time.perf_counter() - started
        run_metadata = dict(memory.metadata)
        run_metadata.update(
            {
                "detector": self.detector.info().to_dict(),
                "tracker": self.tracker.name,
                "config": self.config.to_dict(),
                "zones": self.config.zones.to_dict() if self.config.zones else None,
                "frames_processed": frames_processed,
                "detections_total": detections_total,
                "track_ids_created": len(track_ids),
                "processing_seconds": round(elapsed, 3),
                "processing_fps": round(
                    frames_processed / elapsed if elapsed > 0 else 0.0, 2
                ),
            }
        )
        memory.metadata = run_metadata
        temporal.metadata = run_metadata

        return PipelineResult(
            memory=memory,
            frames_processed=frames_processed,
            elapsed_seconds=elapsed,
            metadata=run_metadata,
            temporal=temporal,
            detections_total=detections_total,
            track_reports_total=track_reports_total,
            track_ids_created=len(track_ids),
            track_frames=track_frames,
        )

    def run(self, video_path: Optional[str] = None) -> PipelineResult:
        """Run the pipeline over a local video file."""
        path = video_path or self.config.video_path
        if not path:
            raise ValueError("No video path provided")
        self.config.video_path = path

        with VideoSource(
            path,
            stride=self.config.stride,
            max_frames=self.config.max_frames,
            start_time=self.config.start_time,
        ) as source:
            logger.info(
                "Processing %s (%dx%d @ %.2ffps, %d frames)",
                path,
                source.metadata.width,
                source.metadata.height,
                source.metadata.fps,
                source.metadata.frame_count,
            )
            return self.run_frames(source, metadata={"video": source.metadata.to_dict()})

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Convenience entry point: run a configured pipeline and save the events."""
    pipeline = PerceptionPipeline(config)
    try:
        result = pipeline.run()
    finally:
        pipeline.close()
    if config.output_path:
        result.memory.save_json(config.output_path)
    if config.memory_path and result.temporal is not None:
        result.temporal.save_json(config.memory_path)
    return result
