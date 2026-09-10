"""A contour-based detector that needs only OpenCV.

This is *not* a replacement for a real neural detector — it exists so that the
full perception pipeline can be run end-to-end (including real video decoding)
on machines without model weights, network access, or a GPU. It detects
saturated colour blobs, which is exactly what ``scripts/generate_demo_video.py``
draws.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from ..detector import Detector
from ..types import Detection, DetectorInfo

# Hue ranges (OpenCV HSV, hue 0-179) mapped to the label they stand in for.
DEFAULT_HUE_CLASSES: Sequence[Tuple[int, int, int, str]] = (
    (0, 10, 0, "person"),      # red
    (170, 179, 0, "person"),   # red wraps around the hue circle
    (100, 130, 7, "truck"),    # blue
    (35, 85, 2, "car"),        # green
)


class BlobDetector(Detector):
    """Detects coloured blobs and labels them by hue band."""

    name = "blob"

    def __init__(
        self,
        min_area: float = 400.0,
        min_saturation: int = 120,
        min_value: int = 80,
        hue_classes: Sequence[Tuple[int, int, int, str]] = DEFAULT_HUE_CLASSES,
    ) -> None:
        import cv2  # imported here so the module only hard-requires cv2 when used
        import numpy as np

        self._cv2 = cv2
        self._np = np
        self.min_area = float(min_area)
        self.min_saturation = int(min_saturation)
        self.min_value = int(min_value)
        self.hue_classes = tuple(hue_classes)

    def detect(self, frame: Any) -> List[Detection]:
        cv2, np = self._cv2, self._np
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Merge masks for hue bands that share a label (e.g. red wraps around).
        masks: Dict[Tuple[int, str], Any] = {}
        for lo, hi, class_id, class_name in self.hue_classes:
            mask = cv2.inRange(
                hsv,
                np.array([lo, self.min_saturation, self.min_value], dtype=np.uint8),
                np.array([hi, 255, 255], dtype=np.uint8),
            )
            key = (class_id, class_name)
            masks[key] = mask if key not in masks else cv2.bitwise_or(masks[key], mask)

        detections: List[Detection] = []
        kernel = np.ones((3, 3), np.uint8)
        for (class_id, class_name), mask in masks.items():
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < self.min_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                # Fill ratio doubles as a confidence proxy: solid rectangles score high.
                fill = area / float(max(1, w * h))
                detections.append(
                    Detection(
                        bbox=(float(x), float(y), float(x + w), float(y + h)),
                        confidence=round(min(0.99, 0.5 + 0.49 * fill), 4),
                        class_id=class_id,
                        class_name=class_name,
                    )
                )
        detections.sort(key=lambda d: (-d.confidence, d.bbox[0]))
        return detections

    def info(self) -> DetectorInfo:
        return DetectorInfo(
            backend="blob",
            model="hsv-contours",
            device="cpu",
            accelerator="cpu",
            classes=sorted({name for _, _, _, name in self.hue_classes}),
        )
