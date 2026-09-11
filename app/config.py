"""Pipeline configuration.

One dataclass holds every knob for a perception run, so the CLI, tests, and
(later) a service layer all configure the pipeline the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .spatial import IMAGE_PIXELS, ZoneSet

# COCO classes relevant to incident intelligence. Restricting the class set
# cuts noise sharply: a detected 'potted plant' is never going to matter to a
# risk assessment, but it will pollute the event log.
DEFAULT_CLASSES: List[str] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "train",
    "truck",
    "traffic light",
    "stop sign",
    "fire hydrant",
    "backpack",
    "handbag",
    "suitcase",
    "knife",
    "cell phone",
]


@dataclass
class PipelineConfig:
    """Everything needed to run video -> detection -> tracking -> events."""

    # --- input -----------------------------------------------------------
    video_path: str = ""
    stride: int = 1
    max_frames: Optional[int] = None
    start_time: float = 0.0

    # --- detection -------------------------------------------------------
    detector: str = "yolo"
    weights: str = "yolov8n.pt"
    device: str = "auto"
    imgsz: int = 640
    half: bool = False
    confidence: float = 0.25
    nms_iou: float = 0.45
    classes: Optional[List[str]] = field(default_factory=lambda: list(DEFAULT_CLASSES))

    # --- tracking --------------------------------------------------------
    tracker: str = "byte_iou"
    track_high_threshold: float = 0.5
    track_low_threshold: float = 0.1
    track_iou_threshold: float = 0.3
    track_max_age: int = 30
    track_min_hits: int = 2

    # --- events ----------------------------------------------------------
    # All thresholds below are in IMAGE PIXELS, not real-world units. See
    # app/spatial.py for why the two must not be conflated.
    sample_interval: float = 1.0
    movement_threshold: float = 40.0
    stationary_threshold: float = 15.0
    stationary_duration: float = 2.0

    # --- zones (image-space regions) --------------------------------------
    zones: Optional[ZoneSet] = None

    # --- output ----------------------------------------------------------
    output_path: str = "data/events.json"
    memory_path: Optional[str] = None

    def detector_kwargs(self) -> Dict[str, Any]:
        """Backend constructor arguments, filtered per backend.

        The mock/blob backends take no model arguments, so passing YOLO's
        would be an error rather than a harmless extra.
        """
        if self.detector in ("yolo", "ultralytics"):
            return {
                "weights": self.weights,
                "device": self.device,
                "confidence": self.confidence,
                "iou": self.nms_iou,
                "imgsz": self.imgsz,
                "half": self.half,
            }
        return {}

    def tracker_kwargs(self) -> Dict[str, Any]:
        return {
            "high_threshold": self.track_high_threshold,
            "low_threshold": self.track_low_threshold,
            "iou_threshold": self.track_iou_threshold,
            "max_age": self.track_max_age,
            "min_hits": self.track_min_hits,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detector": self.detector,
            "weights": self.weights if self.detector in ("yolo", "ultralytics") else None,
            "tracker": self.tracker,
            "stride": self.stride,
            "confidence": self.confidence,
            "classes": self.classes,
            "coordinate_space": IMAGE_PIXELS,
            "sample_interval": self.sample_interval,
            "movement_threshold_px": self.movement_threshold,
            "stationary_threshold_px": self.stationary_threshold,
            "stationary_duration_seconds": self.stationary_duration,
            "zones": self.zones.names if self.zones else [],
        }
