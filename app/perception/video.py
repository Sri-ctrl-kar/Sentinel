"""Video ingestion.

Wraps OpenCV's ``VideoCapture`` so the rest of the pipeline receives
:class:`~app.perception.types.Frame` objects carrying a real timestamp. Every
event Sentinel emits is anchored to that timestamp, so getting it right here
matters more than it looks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from .types import Frame


@dataclass
class VideoMetadata:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3),
            "frame_count": self.frame_count,
            "duration_seconds": round(self.duration, 3),
        }


class VideoSource:
    """Iterate over the frames of a local video file.

    Parameters
    ----------
    path:
        Path to a local video file.
    stride:
        Process every Nth frame. ``stride=2`` halves the compute cost; the
        emitted timestamps stay in real video time either way.
    max_frames:
        Stop after this many *processed* frames. Useful for smoke tests.
    start_time:
        Skip to this offset (seconds) before reading.
    """

    def __init__(
        self,
        path: str,
        stride: int = 1,
        max_frames: Optional[int] = None,
        start_time: float = 0.0,
    ) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")

        import cv2

        self._cv2 = cv2
        self.path = path
        self.stride = int(stride)
        self.max_frames = max_frames
        self.start_time = float(start_time)

        self._capture = cv2.VideoCapture(path)
        if not self._capture.isOpened():
            raise IOError(f"Could not open video file: {path}")

        fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        # Some containers report 0 or a nonsense fps; fall back to a sane value
        # so timestamps remain monotonic and roughly correct.
        self.fps = fps if fps and fps > 0 else 30.0
        self.metadata = VideoMetadata(
            path=path,
            width=int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=self.fps,
            frame_count=int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        if self.start_time > 0:
            self._capture.set(cv2.CAP_PROP_POS_MSEC, self.start_time * 1000.0)

    def __iter__(self) -> Iterator[Frame]:
        raw_index = 0
        emitted = 0
        while True:
            ok, image = self._capture.read()
            if not ok:
                break
            if raw_index % self.stride == 0:
                # Prefer the container's own clock; fall back to frame maths.
                pos_ms = float(self._capture.get(self._cv2.CAP_PROP_POS_MSEC) or 0.0)
                timestamp = pos_ms / 1000.0 if pos_ms > 0 else raw_index / self.fps
                height, width = image.shape[:2]
                yield Frame(
                    index=raw_index,
                    timestamp=round(timestamp, 4),
                    image=image,
                    width=width,
                    height=height,
                )
                emitted += 1
                if self.max_frames is not None and emitted >= self.max_frames:
                    break
            raw_index += 1

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
