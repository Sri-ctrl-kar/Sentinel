"""Core perception data types.

These types are the contract between the perception layer and everything
downstream (events, storage, and — later — reasoning). They are intentionally
free of any model- or framework-specific imports so that swapping the
inference backend (CPU / CUDA / ROCm / ONNX Runtime) never ripples outward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels


def _as_bbox(values: Sequence[float]) -> BBox:
    if len(values) != 4:
        raise ValueError(f"bbox must have 4 values, got {len(values)}")
    x1, y1, x2, y2 = (float(v) for v in values)
    # Normalise so that x1 <= x2 and y1 <= y2 regardless of backend convention.
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


@dataclass
class Detection:
    """A single object detected in a single frame.

    Backends must return these; nothing downstream should ever see a raw
    model output tensor.
    """

    bbox: BBox
    confidence: float
    class_id: int
    class_name: str

    def __post_init__(self) -> None:
        self.bbox = _as_bbox(self.bbox)
        self.confidence = float(self.confidence)
        self.class_id = int(self.class_id)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": [round(v, 2) for v in self.bbox],
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


@dataclass
class Track:
    """A detection that has been given a persistent identity across frames."""

    track_id: int
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str
    # Frames since this track was created.
    age: int = 0
    # Consecutive frames this track has been matched to a detection.
    hits: int = 0
    # Frames since this track was last matched to a detection.
    time_since_update: int = 0
    velocity: Tuple[float, float] = (0.0, 0.0)
    history: List[Tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.bbox = _as_bbox(self.bbox)

    @property
    def entity_id(self) -> str:
        """Stable, human-readable identity used in events (e.g. ``person_3``)."""
        return f"{self.class_name.replace(' ', '_')}_{self.track_id}"

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def speed(self) -> float:
        vx, vy = self.velocity
        return (vx * vx + vy * vy) ** 0.5

    def to_dict(self) -> Dict[str, Any]:
        cx, cy = self.center
        return {
            "track_id": self.track_id,
            "entity_id": self.entity_id,
            "bbox": [round(v, 2) for v in self.bbox],
            "position": [round(cx, 2), round(cy, 2)],
            "confidence": round(float(self.confidence), 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "age": self.age,
            "hits": self.hits,
        }


@dataclass
class Frame:
    """One decoded video frame plus its position in the stream."""

    index: int
    timestamp: float  # seconds since the start of the video
    image: Any  # numpy.ndarray (H, W, 3) BGR — typed loosely to avoid a hard numpy import
    width: int = 0
    height: int = 0


@dataclass
class DetectorInfo:
    """Describes the active inference backend, for logging and reporting."""

    backend: str
    model: str
    device: str
    accelerator: str = "cpu"  # "cpu" | "cuda" | "rocm"
    classes: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "device": self.device,
            "accelerator": self.accelerator,
        }
