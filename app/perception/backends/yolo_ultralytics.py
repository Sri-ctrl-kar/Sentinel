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
no branching; ``torch.version.hip`` is what says which stack is actually live.

Since M0.8 that logic lives in :mod:`app.accel`, shared with the benchmark, so
there is exactly one place that decides what a device is and exactly one place
that decides what to call it — an AMD GPU is never reported as CUDA.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...accel import KIND_CPU, KIND_MPS, gpu_kind, mps_available
from ...accel import resolve_device as accel_resolve
from ..detector import Detector
from ..types import Detection, DetectorInfo


def resolve_device(requested: str = "auto") -> str:
    """Map a device request onto the string torch expects.

    Returns ``"cuda"`` for both NVIDIA CUDA and AMD ROCm builds, which is what
    torch itself wants in either case; :func:`describe_accelerator` is what
    tells the two apart for reporting. A thin wrapper over
    :func:`app.accel.resolve_device`, kept for the detector's own callers.
    """
    return accel_resolve(requested).torch_device


def describe_accelerator() -> str:
    """Return ``"rocm"``, ``"cuda"``, ``"mps"`` or ``"cpu"`` for reporting."""
    return gpu_kind() or (KIND_MPS if mps_available() else KIND_CPU)


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
