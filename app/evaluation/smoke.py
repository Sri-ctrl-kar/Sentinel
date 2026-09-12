"""Real-footage smoke test.

Runs the existing pipeline over a user-supplied local video and reports what
happened. It adds no new video handling: :mod:`app.pipeline` remains the only
implementation, and this module formats the result it returns.

What this is
------------
A **diagnostic**. It answers "did the pipeline run, and does its output look
structurally sane on this clip?" — resolution and frame rate were read
correctly, detections were produced, tracks persisted rather than shattering,
events fired, risk was assessed, and the coordinate space is the expected one.

What this is **not**
--------------------
An accuracy measurement. There is no ground truth for a user-supplied clip, so
nothing here can say whether a detection was correct, whether a track follows
the right person, or whether a risk score is justified. Every report carries
that statement in its own output, not just in documentation.

Some numbers that the benchmark reports cannot be reported here at all:

* **ID switches are not measurable without ground truth.** What *can* be
  measured is the symptom a switch produces — tracks that live for only a
  frame or two, and more distinct IDs than there were plausible entities — so
  those are reported instead, clearly labelled as indicators rather than counts.
* **Detection correctness is unknown.** The count of detections says the
  detector fired, not that it fired on the right things.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..spatial import GROUND_PLANE_METERS, IMAGE_PIXELS

#: Printed with every report. Requirement: never present this as accuracy.
SMOKE_TEST_BANNER = (
    "REAL-FOOTAGE SMOKE TEST",
    "NOT A GROUND-TRUTH ACCURACY BENCHMARK",
)

#: A track reported in this many frames or fewer is treated as short-lived.
#: Short-lived tracks are the visible symptom of fragmentation or detector
#: flicker; the threshold is a reporting choice, not a measurement.
SHORT_TRACK_FRAMES = 3


@dataclass
class TrackSummary:
    """Per-entity persistence, as far as it can be known without ground truth."""

    entity_id: str
    class_name: Optional[str]
    #: Frames this entity was actually reported in — the persistence figure.
    frames_present: int
    #: Events recorded for it. Lower than ``frames_present`` by design: event
    #: emission is governed by sampling and movement thresholds, so the two
    #: must never be conflated.
    events_recorded: int
    first_seen: float
    last_seen: float

    @property
    def duration(self) -> float:
        return round(self.last_seen - self.first_seen, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "class_name": self.class_name,
            "frames_present": self.frames_present,
            "events_recorded": self.events_recorded,
            "first_seen": round(self.first_seen, 3),
            "last_seen": round(self.last_seen, 3),
            "duration_seconds": self.duration,
        }


@dataclass
class SmokeTestReport:
    """Everything the smoke test observed about one clip."""

    video_path: str
    width: int
    height: int
    source_fps: float
    frame_count: int
    frames_processed: int
    processing_fps: float
    detector_backend: str
    detector_model: Optional[str]
    accelerator: str
    detections_total: int
    track_ids_created: int
    tracks: List[TrackSummary] = field(default_factory=list)
    event_counts: Dict[str, int] = field(default_factory=dict)
    risk_assessments: int = 0
    max_risk_score: float = 0.0
    max_severity: str = "normal"
    coordinate_space: str = IMAGE_PIXELS
    calibration_active: bool = False
    calibration_name: Optional[str] = None
    zones: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def detections_per_frame(self) -> float:
        return (
            self.detections_total / self.frames_processed
            if self.frames_processed
            else 0.0
        )

    @property
    def short_lived_tracks(self) -> List[TrackSummary]:
        return [t for t in self.tracks if t.frames_present <= SHORT_TRACK_FRAMES]

    @property
    def mean_track_frames(self) -> float:
        if not self.tracks:
            return 0.0
        return sum(t.frames_present for t in self.tracks) / len(self.tracks)

    @property
    def mean_track_duration(self) -> float:
        if not self.tracks:
            return 0.0
        return sum(t.duration for t in self.tracks) / len(self.tracks)

    @property
    def fragmentation_indicator(self) -> float:
        """Share of tracks that barely lived. High means likely fragmentation.

        An *indicator*, not a fragmentation count: without ground truth there
        is no way to know whether two short tracks are one entity split in two
        or two entities that genuinely passed through.
        """
        if not self.tracks:
            return 0.0
        return len(self.short_lived_tracks) / len(self.tracks)

    @property
    def total_events(self) -> int:
        return sum(self.event_counts.values())

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_type": "real_footage_smoke_test",
            "is_accuracy_benchmark": False,
            "disclaimer": (
                "No ground truth exists for this clip. Counts describe what the "
                "pipeline produced, not whether it was correct."
            ),
            "video": {
                "path": self.video_path,
                "resolution": f"{self.width}x{self.height}",
                "source_fps": round(self.source_fps, 3),
                "frame_count": self.frame_count,
            },
            "processing": {
                "frames_processed": self.frames_processed,
                "processing_fps": round(self.processing_fps, 2),
                "realtime_factor": (
                    round(self.processing_fps / self.source_fps, 3)
                    if self.source_fps
                    else None
                ),
            },
            "detection": {
                "backend": self.detector_backend,
                "model": self.detector_model,
                "accelerator": self.accelerator,
                "detections_total": self.detections_total,
                "detections_per_frame": round(self.detections_per_frame, 2),
            },
            "tracking": {
                "track_ids_created": self.track_ids_created,
                "tracks_reported": len(self.tracks),
                "mean_frames_per_track": round(self.mean_track_frames, 2),
                "mean_track_duration_seconds": round(self.mean_track_duration, 3),
                "short_lived_tracks": len(self.short_lived_tracks),
                "fragmentation_indicator": round(self.fragmentation_indicator, 3),
                "id_switches": None,
                "id_switches_note": (
                    "not measurable without ground truth; short_lived_tracks is "
                    "the observable symptom"
                ),
                "tracks": [t.to_dict() for t in self.tracks[:25]],
            },
            "events": {
                "total": self.total_events,
                "by_action": dict(self.event_counts),
            },
            "risk": {
                "assessments": self.risk_assessments,
                "max_risk_score": round(self.max_risk_score, 2),
                "max_severity": self.max_severity,
                "score_interpretation": (
                    "ordinal 0-100 ranking, NOT a calibrated probability"
                ),
            },
            "spatial": {
                "coordinate_space": self.coordinate_space,
                "calibration_active": self.calibration_active,
                "calibration": self.calibration_name,
                "zones": list(self.zones),
            },
            "notes": list(self.notes),
        }

    # ------------------------------------------------------------------
    def to_text(self) -> str:
        lines: List[str] = list(SMOKE_TEST_BANNER)
        lines.append("=" * max(len(line) for line in SMOKE_TEST_BANNER))
        lines.append("")
        lines.append("VIDEO")
        lines.append(f"  path             : {self.video_path}")
        lines.append(f"  resolution       : {self.width}x{self.height}")
        lines.append(
            f"  source fps       : "
            f"{self.source_fps:.2f}" if self.source_fps else "  source fps       : unknown"
        )
        lines.append(
            f"  frame count      : "
            + (str(self.frame_count) if self.frame_count > 0 else "unknown")
        )
        lines.append("")
        lines.append("PROCESSING")
        lines.append(f"  frames processed : {self.frames_processed}")
        lines.append(f"  processing fps   : {self.processing_fps:.2f}")
        if self.source_fps:
            factor = self.processing_fps / self.source_fps
            verdict = "faster than real time" if factor >= 1 else "slower than real time"
            lines.append(f"  realtime factor  : {factor:.2f}x ({verdict})")
        lines.append("")
        lines.append("DETECTION")
        lines.append(f"  backend          : {self.detector_backend}")
        if self.detector_model:
            lines.append(f"  model            : {self.detector_model}")
        lines.append(f"  accelerator      : {self.accelerator}")
        lines.append(f"  detections       : {self.detections_total}")
        lines.append(f"  per frame        : {self.detections_per_frame:.2f}")
        lines.append("")
        lines.append("TRACKING")
        lines.append(f"  track IDs created: {self.track_ids_created}")
        lines.append(f"  tracks reported  : {len(self.tracks)}")
        lines.append(f"  mean frames/track: {self.mean_track_frames:.1f}")
        lines.append(
            f"  mean duration    : {self.mean_track_duration:.2f}s"
            "   (persistence; events are sampled separately)"
        )
        lines.append(
            f"  short-lived (<={SHORT_TRACK_FRAMES} frames): "
            f"{len(self.short_lived_tracks)} "
            f"({self.fragmentation_indicator:.0%} of tracks)"
        )
        lines.append("  ID switches      : not measurable without ground truth")
        if self.tracks:
            lines.append("  longest-lived tracks:")
            for track in sorted(
                self.tracks, key=lambda t: -t.frames_present
            )[:5]:
                lines.append(
                    f"    {track.entity_id:<18} {track.frames_present:>4} frames "
                    f"({track.duration:5.2f}s)  {track.events_recorded:>3} events"
                )
        lines.append("")
        lines.append("EVENTS")
        if self.event_counts:
            for action, count in sorted(self.event_counts.items()):
                lines.append(f"  {action:<17}: {count}")
            lines.append(f"  {'total':<17}: {self.total_events}")
        else:
            lines.append("  none emitted")
        lines.append("")
        lines.append("RISK")
        lines.append(f"  assessments      : {self.risk_assessments}")
        lines.append(f"  max risk score   : {self.max_risk_score:.1f}/100")
        lines.append(f"  max severity     : {self.max_severity}")
        lines.append("  (ordinal ranking, NOT a probability)")
        lines.append("")
        lines.append("SPATIAL")
        lines.append(f"  coordinate space : {self.coordinate_space}")
        lines.append(
            f"  calibration      : "
            + (
                f"ACTIVE ({self.calibration_name})"
                if self.calibration_active
                else "none supplied - image-space fallback"
            )
        )
        lines.append(f"  zones            : {', '.join(self.zones) or 'none'}")
        if self.notes:
            lines.append("")
            lines.append("NOTES")
            for note in self.notes:
                lines.append(f"  - {note}")
        lines.append("")
        lines.append("INTERPRETATION")
        for line in INTERPRETATION:
            lines.append(f"  {line}")
        return "\n".join(lines)


INTERPRETATION = [
    "No ground truth exists for this clip, so nothing here measures accuracy.",
    "Detection and track counts say the pipeline produced output; they do not",
    "say the output was correct. A high short-lived-track share suggests",
    "fragmentation or detector flicker and is worth investigating, but it is an",
    "indicator, not a measured ID-switch count.",
    "",
    "For measured tracking and event quality, annotate a clip and run",
    "`python -m app.evaluation --clip your_annotation.json`.",
]


# ---------------------------------------------------------------------------
def build_report(
    result: Any,
    video_path: str,
    risk_report: Any = None,
    calibration: Any = None,
    zones: Optional[List[str]] = None,
) -> SmokeTestReport:
    """Assemble a report from a completed :class:`~app.pipeline.PipelineResult`.

    Takes an already-computed result rather than running anything itself, so
    there is exactly one video-processing implementation in the repository.
    """
    video = result.metadata.get("video", {}) or {}
    detector = result.metadata.get("detector", {}) or {}
    memory = result.temporal

    tracks: List[TrackSummary] = []
    event_counts: Dict[str, int] = {}
    if memory is not None:
        frames_by_entity: Dict[str, int] = {}
        for event in memory:
            event_counts[event.action] = event_counts.get(event.action, 0) + 1
            frames_by_entity[event.entity_id] = (
                frames_by_entity.get(event.entity_id, 0) + 1
            )
        track_frames = getattr(result, "track_frames", {}) or {}
        for entity_id in memory.entities():
            state = memory.latest_state(entity_id)
            if state is None:
                continue
            tracks.append(
                TrackSummary(
                    entity_id=entity_id,
                    class_name=state.class_name,
                    frames_present=track_frames.get(entity_id, 0),
                    events_recorded=frames_by_entity.get(entity_id, 0),
                    first_seen=state.first_seen or 0.0,
                    last_seen=state.last_seen or 0.0,
                )
            )

    notes: List[str] = []
    if result.frames_processed == 0:
        notes.append("No frames were processed — the video could not be read.")
    if result.detections_total == 0 and result.frames_processed:
        notes.append(
            "No detections at all. Check the detector backend and its class "
            "filter before reading anything else in this report."
        )
    if tracks and len([t for t in tracks if t.frames_present <= SHORT_TRACK_FRAMES]) / len(
        tracks
    ) > 0.5:
        notes.append(
            "More than half of all tracks were short-lived, which suggests "
            "fragmentation or detector flicker on this footage."
        )

    report = SmokeTestReport(
        video_path=video_path,
        width=int(video.get("width", 0)),
        height=int(video.get("height", 0)),
        source_fps=float(video.get("fps", 0.0)),
        frame_count=int(video.get("frame_count", 0)),
        frames_processed=result.frames_processed,
        processing_fps=result.fps,
        detector_backend=detector.get("backend", "unknown"),
        detector_model=detector.get("model"),
        accelerator=detector.get("accelerator", "cpu"),
        detections_total=result.detections_total,
        track_ids_created=result.track_ids_created,
        tracks=tracks,
        event_counts=event_counts,
        calibration_active=calibration is not None,
        calibration_name=getattr(calibration, "name", None),
        coordinate_space=(
            GROUND_PLANE_METERS if calibration is not None else IMAGE_PIXELS
        ),
        zones=list(zones or []),
        notes=notes,
    )

    if risk_report is not None:
        report.risk_assessments = len(risk_report.assessments)
        report.max_risk_score = risk_report.max_score
        report.max_severity = risk_report.severity

    return report
