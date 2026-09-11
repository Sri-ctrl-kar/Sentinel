"""Dependency-free detector backends used for tests and CI.

``MockDetector`` needs no model weights, no torch and no GPU, which lets the
whole pipeline (video -> detection -> tracking -> events) be exercised
deterministically in unit tests.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence

from ..detector import Detector
from ..types import Detection, DetectorInfo


class MockDetector(Detector):
    """Returns pre-scripted or procedurally generated detections.

    Two modes:

    * **scripted** — pass ``script``: a list of per-frame detection lists,
      replayed in order (the last entry repeats once exhausted).
    * **synthetic** — the default: emits a ``person`` and a ``forklift``
      moving linearly across the frame, so tracking behaviour is testable
      without any video at all.
    """

    name = "mock"

    def __init__(
        self,
        script: Optional[Sequence[Sequence[Detection]]] = None,
        generator: Optional[Callable[[int], List[Detection]]] = None,
        frame_size: Sequence[int] = (640, 384),
    ) -> None:
        self.script = [list(f) for f in script] if script is not None else None
        self.generator = generator
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.calls = 0

    def detect(self, frame: Any) -> List[Detection]:
        index = self.calls
        self.calls += 1

        if self.script is not None:
            if not self.script:
                return []
            return list(self.script[min(index, len(self.script) - 1)])
        if self.generator is not None:
            return list(self.generator(index))
        return self._synthetic(index)

    def _synthetic(self, index: int) -> List[Detection]:
        w, h = self.frame_size
        # A person walking left-to-right.
        px = 20.0 + index * 6.0
        person = Detection(
            bbox=(px, h * 0.4, px + 40.0, h * 0.4 + 90.0),
            confidence=0.90,
            class_id=0,
            class_name="person",
        )
        # A forklift crossing right-to-left; disappears past the frame edge.
        fx = w - 120.0 - index * 8.0
        out = [person]
        if fx > -60.0:
            out.append(
                Detection(
                    bbox=(fx, h * 0.5, fx + 110.0, h * 0.5 + 70.0),
                    confidence=0.82,
                    class_id=7,
                    class_name="truck",
                )
            )
        return out

    def info(self) -> DetectorInfo:
        return DetectorInfo(
            backend="mock",
            model="scripted" if self.script is not None else "synthetic",
            device="cpu",
            accelerator="cpu",
        )
