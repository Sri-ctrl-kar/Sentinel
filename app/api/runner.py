"""The API's frame loop, and the per-frame track record the frontend needs.

Why this exists
---------------
``PerceptionPipeline.run_frames()`` keeps events and throws individual
``Track`` objects away — it counts frames per entity and nothing more. That is
right for the pipeline's job and wrong for a browser, which needs a box per
entity per frame to draw an overlay that does not stutter.

So the runner drives the loop itself, exactly as
``app/benchmark/harness.py:run_benchmark()`` already does, and records the
tracks as they go past. Every component and every threshold comes from
:class:`~app.config.PipelineConfig`; the detector, the tracker and the event
generator are the validated ones, constructed the way the pipeline constructs
them. Nothing here re-implements perception, tracking or event semantics — it
observes them.

``tests/test_api_runner.py`` asserts that: the same clip through this runner and
through ``PerceptionPipeline.run()`` produces identical events, entities and
counts. If this loop ever drifts from the pipeline's, that test fails.

On event density (M0.9 requirement 9)
-------------------------------------
``PipelineConfig.sample_interval`` defaults to 1.0 s, so ``moved`` /
``stationary`` events are about 1 Hz per entity. That default is part of the
validated event semantics and is **not** changed here. Smooth visualisation is
served instead by :class:`FrameRecord`, which is dense at the video's own frame
rate and carries no event meaning at all. Events keep their timestamps, their
actions and their sampling; the overlay gets its own channel.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..accel import DeviceUnavailable, resolve_device
from ..config import PipelineConfig
from ..events.generator import EventGenerator
from ..events.schema import Event
from ..memory.temporal import TemporalEventMemory
from ..perception.detector import create_detector
from ..perception.tracker import create_tracker
from ..perception.types import Track
from ..perception.video import VideoSource
from ..pipeline import PipelineResult
from ..storage.memory import EventMemory
from .store import AnalysisRecord, Progress


@dataclass
class FrameRecord:
    """One frame's tracking state, as the tracker reported it.

    This is a *transcription*, not a computation: ``tracks`` are the objects
    ``Tracker.update()`` returned for this frame, and ``events`` are the events
    the generator emitted on this frame. No box is smoothed, interpolated,
    filtered or invented.
    """

    frame_index: int
    timestamp: float
    tracks: List[Track] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 4),
            "tracks": [t.to_dict() for t in self.tracks],
            "events": [e.to_dict() for e in self.events],
        }


def device_payload(config: PipelineConfig) -> Dict[str, Any]:
    """Sentinel's *semantic* device description for this run.

    ``DeviceSpec.kind`` is the semantic answer (``cpu`` / ``rocm`` / ``cuda`` /
    ``mps``). ``torch_device`` is carried alongside under its own name, clearly
    as the runtime string — on ROCm it is ``"cuda"``, and it must never be read
    as the semantic device.
    """
    spec = resolve_device(config.device)
    payload = spec.to_dict()
    payload["semantic_device"] = spec.kind
    payload["requested"] = config.device
    return payload


def run_analysis(record: AnalysisRecord) -> AnalysisRecord:
    """Run one analysis to completion, updating ``record`` as it goes.

    Never raises: a failure is recorded on the record as ``status: failed``
    with the exception type and message, because this runs detached from the
    request that asked for it and there is nobody left to raise at.
    """
    try:
        _run(record)
    except BaseException as exc:  # noqa: BLE001 - recorded, not swallowed
        record.mark_failed(exc)
    return record


def _run(record: AnalysisRecord) -> None:
    config = record.config
    record.mark_running()
    record.device = device_payload(config)

    # Built exactly as PerceptionPipeline builds them, from the same config.
    detector = create_detector(
        config.detector,
        classes=config.classes,
        min_confidence=config.confidence,
        **config.detector_kwargs(),
    )
    tracker = create_tracker(config.tracker, **config.tracker_kwargs())
    generator = EventGenerator(
        sample_interval=config.sample_interval,
        movement_threshold=config.movement_threshold,
        stationary_threshold=config.stationary_threshold,
        stationary_duration=config.stationary_duration,
        zones=config.zones,
        source=config.video_path or None,
    )

    frames: List[FrameRecord] = []
    memory: Optional[EventMemory] = None
    temporal: Optional[TemporalEventMemory] = None
    started = time.perf_counter()
    frames_processed = 0
    detections_total = 0
    track_reports_total = 0
    track_ids: set = set()
    track_frames: Dict[str, int] = {}
    last_timestamp = 0.0
    last_frame_index: Optional[int] = None
    last_tracks: Sequence[Track] = []

    try:
        with VideoSource(
            config.video_path,
            stride=config.stride,
            max_frames=config.max_frames,
            start_time=config.start_time,
        ) as source:
            record.video_metadata = source.metadata.to_dict()
            total = source.metadata.frame_count or None
            if config.max_frames:
                total = min(total, config.max_frames) if total else config.max_frames
            record.progress = Progress(frames_processed=0, total_frames=total)

            metadata = {"video": source.metadata.to_dict()}
            memory = EventMemory(metadata=dict(metadata))
            temporal = TemporalEventMemory(metadata=dict(metadata))

            for frame in source:
                detections = detector.detect(frame.image)
                tracks = tracker.update(detections, frame.timestamp)
                events = generator.process(
                    tracks,
                    timestamp=frame.timestamp,
                    frame_index=frame.index,
                    lost_tracks=tracker.lost_tracks,
                )
                memory.extend(events)
                temporal.ingest_many(events)

                # The API's own addition: keep the frame's tracking state.
                frames.append(
                    FrameRecord(
                        frame_index=frame.index,
                        timestamp=frame.timestamp,
                        tracks=list(tracks),
                        events=list(events),
                    )
                )

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
                record.progress.frames_processed = frames_processed
    finally:
        detector.close()

    # Closing events, exactly as the pipeline flushes them.
    closing = generator.flush(
        last_timestamp, last_tracks, frame_index=last_frame_index
    )
    if memory is not None:
        memory.extend(closing)
    if temporal is not None:
        temporal.ingest_many(closing)
    if closing and frames:
        frames[-1].events.extend(closing)

    elapsed = time.perf_counter() - started
    run_metadata: Dict[str, Any] = dict(memory.metadata if memory else {})
    run_metadata.update(
        {
            "detector": detector.info().to_dict(),
            "tracker": tracker.name,
            "config": config.to_dict(),
            "zones": config.zones.to_dict() if config.zones else None,
            "frames_processed": frames_processed,
            "detections_total": detections_total,
            "track_ids_created": len(track_ids),
            "processing_seconds": round(elapsed, 3),
            "processing_fps": round(
                frames_processed / elapsed if elapsed > 0 else 0.0, 2
            ),
            "device": record.device,
        }
    )
    if memory is not None:
        memory.metadata = run_metadata
    if temporal is not None:
        temporal.metadata = run_metadata

    record.result = PipelineResult(
        memory=memory if memory is not None else EventMemory(),
        frames_processed=frames_processed,
        elapsed_seconds=elapsed,
        metadata=run_metadata,
        temporal=temporal,
        detections_total=detections_total,
        track_reports_total=track_reports_total,
        track_ids_created=len(track_ids),
        track_frames=track_frames,
    )
    record.frames = frames
    record.mark_complete()


# ---------------------------------------------------------------------------
# The video timeline the frontend draws from
# ---------------------------------------------------------------------------
def risk_at(reports: Sequence[Any], timestamp: float, window: float) -> Optional[Any]:
    """The risk report nearest ``timestamp``, within ``window`` seconds.

    Risk is assessed on its own grid (``risk_step``), which is coarser than the
    frame rate. Attaching the nearest assessment inside a half-step window is
    an association, not an interpolation: no score is invented for a moment the
    engine did not assess.
    """
    best = None
    best_delta = None
    for report in reports:
        delta = abs(report.timestamp - timestamp)
        if delta <= window and (best_delta is None or delta < best_delta):
            best, best_delta = report, delta
    return best


def build_timeline(
    record: AnalysisRecord,
    risk_reports: Optional[Sequence[Any]] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Frame-by-frame rows: tracks, events, and risk where it was assessed."""
    reports = list(risk_reports or [])
    window = max(record.risk_step, 1e-6) / 2.0
    rows: List[Dict[str, Any]] = []

    selected = record.frames[offset:]
    if limit is not None:
        selected = selected[:limit]

    for frame in selected:
        row = frame.to_dict()
        report = risk_at(reports, frame.timestamp, window) if reports else None
        if report is not None and report.assessments:
            top = report.top
            row["risk"] = {
                "timestamp": round(report.timestamp, 4),
                "risk_score": round(top.risk_score, 2),
                "severity": top.severity,
                "incident_type": top.incident_type,
                "involved_entity_ids": list(top.involved_entity_ids),
                "coordinate_space": top.coordinate_space,
                "time_to_risk": top.time_to_risk.to_dict() if top.time_to_risk else None,
            }
        else:
            row["risk"] = None
        rows.append(row)
    return rows
