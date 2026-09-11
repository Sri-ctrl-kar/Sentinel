"""Clip annotation format and the evaluation runner for real footage.

The annotation file
-------------------
One JSON document per clip, designed so a human can write it for a 10-30
second clip without tooling::

    {
      "name": "loading_bay_clip",
      "video": "data/clips/bay.mp4",          // or "frames": "data/clips/bay/"
      "fps": 20.0,
      "notes": "hand annotated, 18s",
      "calibration": {                         // optional
        "image_points": [[100,100],[900,100],[900,500],[100,500]],
        "world_points": [[0,0],[20,0],[20,10],[0,10]]
      },
      "zones": [ {"name": "forklift_bay", "rect": [400,200,800,500]} ],
      "operating_zones": ["forklift_bay"],
      "entities": [
        {"id": "worker_1", "class_name": "person",
         "boxes": {"0": [100,300,140,390], "10": [160,300,200,390]}}
      ],
      "annotated_actions": ["appeared", "disappeared", "entered_zone"],
      "events": [
        {"entity": "worker_1", "action": "entered_zone",
         "time": 3.2, "zone": "forklift_bay"}
      ],
      "unsafe_transitions": [
        {"entities": ["worker_1", "forklift_2"], "time": 5.4}
      ]
    }

Only ``entities`` is required. Boxes are keyed by frame index and **linearly
interpolated between annotated frames**, so a human marks a box every 10 or 20
frames rather than every frame. Everything else is optional: with no
calibration the clip is evaluated in image space, with no events the event
metrics are skipped, and so on.

``annotated_actions`` is the annotator's claim about **completeness**: "for
these actions, I labelled every occurrence". Only those actions are scored.
This matters for honesty — without it, a pipeline emitting `moved` events for a
clip whose annotator never labelled movement would be charged hundreds of false
positives for doing its job. Omit the field and it defaults to the actions that
appear in ``events``, which is the same claim stated implicitly.

No video is committed to this repository. Point ``video`` at a local file, or
generate a rendered clip with ``scripts/generate_demo_video.py --annotations``,
which writes ground truth from its own drawing commands — independent of the
pipeline being evaluated.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..calibration.planar import GroundPlaneCalibration
from ..spatial import Zone, ZoneSet
from .events import AnnotatedEvent


@dataclass
class EntityTrack:
    """One annotated entity's boxes over time."""

    entity_id: str
    class_name: str
    #: frame index -> (x1, y1, x2, y2)
    boxes: Dict[int, Tuple[float, float, float, float]] = field(default_factory=dict)

    @property
    def annotated_frames(self) -> List[int]:
        return sorted(self.boxes)

    def box_at(self, frame: int) -> Optional[Tuple[float, float, float, float]]:
        """Box at ``frame``, linearly interpolated between annotated frames.

        Returns ``None`` outside the annotated span: an entity is not assumed
        to exist before it was first marked or after it was last marked.
        """
        frames = self.annotated_frames
        if not frames or frame < frames[0] or frame > frames[-1]:
            return None
        if frame in self.boxes:
            return self.boxes[frame]

        before = max(f for f in frames if f < frame)
        after = min(f for f in frames if f > frame)
        ratio = (frame - before) / (after - before)
        a, b = self.boxes[before], self.boxes[after]
        return tuple(a[i] + (b[i] - a[i]) * ratio for i in range(4))  # type: ignore[return-value]


@dataclass
class ClipAnnotation:
    """A clip plus everything a human said about it."""

    name: str
    fps: float = 20.0
    video: Optional[str] = None
    frames_dir: Optional[str] = None
    notes: str = ""
    entities: List[EntityTrack] = field(default_factory=list)
    events: List[AnnotatedEvent] = field(default_factory=list)
    #: Actions the annotator claims to have labelled exhaustively. Only these
    #: are scored; anything else is "not annotated", not "wrong".
    annotated_actions: List[str] = field(default_factory=list)
    unsafe_transitions: List[Dict[str, Any]] = field(default_factory=list)
    calibration: Optional[GroundPlaneCalibration] = None
    zones: Optional[ZoneSet] = None
    operating_zones: List[str] = field(default_factory=list)
    source_path: Optional[str] = None

    @property
    def has_calibration(self) -> bool:
        return self.calibration is not None

    @property
    def frame_span(self) -> Tuple[int, int]:
        frames = [f for entity in self.entities for f in entity.annotated_frames]
        return (min(frames), max(frames)) if frames else (0, 0)

    def ground_truth_frames(self) -> List[Dict[str, Tuple[float, float, float, float]]]:
        """Per-frame ``{entity_id: bbox}`` across the annotated span."""
        first, last = self.frame_span
        output = []
        for frame in range(first, last + 1):
            boxes = {}
            for entity in self.entities:
                box = entity.box_at(frame)
                if box is not None:
                    boxes[entity.entity_id] = box
            output.append(boxes)
        return output

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "fps": self.fps,
            "video": self.video,
            "frames": self.frames_dir,
            "notes": self.notes,
            "has_calibration": self.has_calibration,
            "entity_count": len(self.entities),
            "annotated_event_count": len(self.events),
            "annotated_actions": list(self.annotated_actions),
            "unsafe_transition_count": len(self.unsafe_transitions),
            "zones": self.zones.names if self.zones else [],
            "operating_zones": list(self.operating_zones),
        }


# ---------------------------------------------------------------------------
def load_annotation(path: str) -> ClipAnnotation:
    """Read and validate a clip annotation file."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    annotation = parse_annotation(payload)
    annotation.source_path = path
    # Relative media paths resolve against the annotation file, so a clip and
    # its annotation can be moved together.
    base = os.path.dirname(os.path.abspath(path))
    if annotation.video and not os.path.isabs(annotation.video):
        annotation.video = os.path.normpath(os.path.join(base, annotation.video))
    if annotation.frames_dir and not os.path.isabs(annotation.frames_dir):
        annotation.frames_dir = os.path.normpath(
            os.path.join(base, annotation.frames_dir)
        )
    return annotation


def parse_annotation(payload: Dict[str, Any]) -> ClipAnnotation:
    """Build a :class:`ClipAnnotation` from a decoded JSON document."""
    if "entities" not in payload:
        raise ValueError("Annotation must contain an 'entities' list")

    entities = []
    for raw in payload["entities"]:
        for key in ("id", "class_name"):
            if key not in raw:
                raise ValueError(f"Entity is missing required field '{key}'")
        boxes = {}
        for frame, box in (raw.get("boxes") or {}).items():
            if len(box) != 4:
                raise ValueError(
                    f"Entity '{raw['id']}' frame {frame}: box needs 4 values"
                )
            boxes[int(frame)] = tuple(float(v) for v in box)
        if not boxes:
            raise ValueError(f"Entity '{raw['id']}' has no annotated boxes")
        entities.append(
            EntityTrack(entity_id=raw["id"], class_name=raw["class_name"], boxes=boxes)
        )

    events = [
        AnnotatedEvent(
            entity_id=raw["entity"],
            action=raw["action"],
            timestamp=float(raw["time"]),
            zone=raw.get("zone"),
        )
        for raw in payload.get("events", [])
    ]

    calibration = None
    if payload.get("calibration"):
        calibration = GroundPlaneCalibration(
            image_points=payload["calibration"]["image_points"],
            world_points=payload["calibration"]["world_points"],
            name=f"{payload.get('name', 'clip')}_calibration",
            description=payload["calibration"].get("description", ""),
        )

    zones = None
    if payload.get("zones"):
        zones = ZoneSet.from_dicts(payload["zones"])

    declared_actions = payload.get("annotated_actions")
    if declared_actions is None:
        # Implicit claim: the actions that appear in the event list.
        declared_actions = sorted({e.action for e in events})

    return ClipAnnotation(
        name=payload.get("name", "clip"),
        fps=float(payload.get("fps", 20.0)),
        video=payload.get("video"),
        frames_dir=payload.get("frames"),
        notes=payload.get("notes", ""),
        entities=entities,
        events=events,
        annotated_actions=list(declared_actions),
        unsafe_transitions=list(payload.get("unsafe_transitions", [])),
        calibration=calibration,
        zones=zones,
        operating_zones=list(payload.get("operating_zones", [])),
    )


# ---------------------------------------------------------------------------
@dataclass
class ClipRun:
    """What the pipeline produced for one clip, ready to be scored."""

    annotation: ClipAnnotation
    predicted_frames: List[Dict[str, Tuple[float, float, float, float]]]
    memory: Any
    frames_processed: int
    detector_backend: str
    coordinate_space: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip": self.annotation.name,
            "frames_processed": self.frames_processed,
            "detector_backend": self.detector_backend,
            "coordinate_space": self.coordinate_space,
        }


def run_clip(
    annotation: ClipAnnotation,
    detector: str = "blob",
    classes: Optional[Sequence[str]] = None,
    confidence: float = 0.0,
    **pipeline_overrides: Any,
) -> ClipRun:
    """Run the real pipeline over a clip's video and capture its output.

    Imports the pipeline lazily so that the evaluation package as a whole
    stays importable without OpenCV — the metric code is pure Python and the
    layering tests depend on that remaining true.
    """
    if not annotation.video:
        raise ValueError(
            f"Clip '{annotation.name}' has no 'video' path; frame-sequence "
            "evaluation requires one of 'video' or 'frames'"
        )

    from ..config import PipelineConfig
    from ..pipeline import PerceptionPipeline
    from ..spatial import IMAGE_PIXELS, GROUND_PLANE_METERS

    config = PipelineConfig(
        video_path=annotation.video,
        detector=detector,
        classes=list(classes) if classes else None,
        confidence=confidence,
        zones=annotation.zones,
        sample_interval=pipeline_overrides.pop("sample_interval", 0.25),
        movement_threshold=pipeline_overrides.pop("movement_threshold", 20.0),
        stationary_threshold=pipeline_overrides.pop("stationary_threshold", 8.0),
        stationary_duration=pipeline_overrides.pop("stationary_duration", 1.0),
        output_path="",
        **pipeline_overrides,
    )

    pipeline = PerceptionPipeline(config)
    predicted_frames: List[Dict[str, Tuple[float, float, float, float]]] = []

    # Capture per-frame track boxes as the pipeline runs, so tracking can be
    # scored without re-deriving anything from the event log.
    original_update = pipeline.tracker.update

    def capture(detections, timestamp):
        tracks = original_update(detections, timestamp)
        predicted_frames.append(
            {t.entity_id: tuple(t.bbox) for t in tracks}
        )
        return tracks

    pipeline.tracker.update = capture  # type: ignore[method-assign]
    try:
        result = pipeline.run()
    finally:
        pipeline.tracker.update = original_update  # type: ignore[method-assign]
        pipeline.close()

    return ClipRun(
        annotation=annotation,
        predicted_frames=predicted_frames,
        memory=result.temporal,
        frames_processed=result.frames_processed,
        detector_backend=result.metadata.get("detector", {}).get("backend", detector),
        coordinate_space=(
            GROUND_PLANE_METERS if annotation.has_calibration else IMAGE_PIXELS
        ),
    )
