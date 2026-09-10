"""YOLOv8 detection backend (Ultralytics).

This is the only file in the repository that imports ``ultralytics`` or
``torch``. Everything else talks to the :class:`~app.perception.detector.Detector`
interface, so replacing this backend with an ONNX Runtime / MIGraphX one on
AMD hardware is an additive change, not a refactor.

AMD/ROCm note
-------------
ROCm builds of PyTorch expose the HIP runtime through the *CUDA* API surface:
``torch.cuda.is_available()`` returns ``True`` and the device string is still
``"cuda"``. So ``device="auto"`` resolves correctly on both NVIDIA and AMD with
no branching; we only inspect ``torch.version.hip`` to *report* which stack is
actually live.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..detector import Detector
from ..types import Detection, DetectorInfo


def resolve_device(requested: str = "auto") -> str:
    """Map ``auto`` onto the best available torch device.

    Returns ``"cuda"`` for both NVIDIA CUDA and AMD ROCm builds, which is what
    torch itself expects in either case.
    """
    if requested and requested != "auto":
        return requested
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is an optional dependency
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe_accelerator() -> str:
    """Return ``"rocm"``, ``"cuda"``, ``"mps"`` or ``"cpu"`` for reporting."""
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"
    if torch.cuda.is_available():
        # torch.version.hip is set only on ROCm builds.
        return "rocm" if getattr(torch.version, "hip", None) else "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class UltralyticsYOLODetector(Detector):
    """Wraps an Ultralytics YOLO model behind Sentinel's detector interface."""

    name = "yolo"

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        device: str = "auto",
        confidence: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        half: bool = False,
        verbose: bool = False,
    ) -> None:
        try:  # deferred heavy import: keeps torch off the critical path
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "The 'yolo' backend requires ultralytics and torch. Install them with "
                "'pip install -r requirements-yolo.txt' (see README for the AMD ROCm "
                "wheel index), or run with --detector blob / --detector mock."
            ) from exc

        self.weights = weights
        self.device = resolve_device(device)
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        # FP16 is a win on AMD CDNA and NVIDIA tensor cores, but is not
        # supported on CPU, so it is silently ignored there.
        self.half = bool(half) and self.device not in ("cpu", "mps")
        self.verbose = verbose
        self._model = YOLO(weights)
        self._names = dict(getattr(self._model, "names", {}) or {})

    def detect(self, frame: Any) -> List[Detection]:
        kwargs: Dict[str, Any] = {
            "conf": self.confidence,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "device": self.device,
            "verbose": self.verbose,
        }
        # Only pass ``half`` when it is actually wanted: recent Ultralytics
        # releases warn on the argument even when it is False.
        if self.half:
            kwargs["half"] = True
        results = self._model.predict(frame, **kwargs)
        detections: List[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            names = dict(getattr(result, "names", {}) or self._names)
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            for box, conf, class_id in zip(xyxy, confs, classes):
                detections.append(
                    Detection(
                        bbox=tuple(float(v) for v in box),
                        confidence=float(conf),
                        class_id=int(class_id),
                        class_name=str(names.get(int(class_id), int(class_id))),
                    )
                )
        return detections

    def warmup(self, frame: Any = None) -> None:
        import numpy as np

        if frame is None:
            frame = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.detect(frame)

    def info(self) -> DetectorInfo:
        return DetectorInfo(
            backend="ultralytics-yolo",
            model=self.weights,
            device=self.device,
            accelerator=describe_accelerator(),
            classes=[str(v) for v in self._names.values()] or None,
        )

    def close(self) -> None:
        self._model = None
