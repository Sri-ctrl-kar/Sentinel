"""Model-agnostic object detection interface.

Everything downstream of this module talks to :class:`Detector`, never to a
concrete model library. That keeps the inference backend swappable: CPU today,
AMD ROCm (or ONNX Runtime / MIGraphX) later, with no changes to the pipeline,
the tracker, or the event layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .types import Detection, DetectorInfo


class Detector(ABC):
    """Base class for every detection backend.

    Subclasses only need to implement :meth:`detect` and :meth:`info`.
    """

    #: Registry name used by :func:`create_detector`.
    name: str = "base"

    @abstractmethod
    def detect(self, frame: Any) -> List[Detection]:
        """Run detection on a single BGR frame and return :class:`Detection`s."""
        raise NotImplementedError

    @abstractmethod
    def info(self) -> DetectorInfo:
        """Describe the loaded model and the device it runs on."""
        raise NotImplementedError

    def warmup(self, frame: Any = None) -> None:
        """Optional: run a throwaway inference so the first real frame isn't slow."""
        return None

    def close(self) -> None:
        """Optional: release model/device resources."""
        return None

    def __enter__(self) -> "Detector":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class FilteredDetector(Detector):
    """Wraps another detector and drops detections we don't care about.

    Used to restrict COCO's 80 classes down to the entities Sentinel reasons
    about, and to enforce a confidence floor uniformly across backends.
    """

    name = "filtered"

    def __init__(
        self,
        inner: Detector,
        classes: Optional[Iterable[str]] = None,
        min_confidence: float = 0.0,
        min_area: float = 0.0,
    ) -> None:
        self.inner = inner
        self.classes = {c.lower() for c in classes} if classes else None
        self.min_confidence = float(min_confidence)
        self.min_area = float(min_area)

    def detect(self, frame: Any) -> List[Detection]:
        out = []
        for det in self.inner.detect(frame):
            if det.confidence < self.min_confidence:
                continue
            if self.classes is not None and det.class_name.lower() not in self.classes:
                continue
            if self.min_area and det.area < self.min_area:
                continue
            out.append(det)
        return out

    def info(self) -> DetectorInfo:
        return self.inner.info()

    def warmup(self, frame: Any = None) -> None:
        self.inner.warmup(frame)

    def close(self) -> None:
        self.inner.close()


# --------------------------------------------------------------------------
# Backend registry
# --------------------------------------------------------------------------

_REGISTRY: Dict[str, Callable[..., Detector]] = {}


def register_detector(name: str, factory: Callable[..., Detector]) -> None:
    """Register a detector backend under ``name``."""
    _REGISTRY[name.lower()] = factory


def available_detectors() -> List[str]:
    _ensure_builtin_backends()
    return sorted(_REGISTRY)


def create_detector(
    backend: str = "yolo",
    classes: Optional[Sequence[str]] = None,
    min_confidence: float = 0.0,
    **kwargs: Any,
) -> Detector:
    """Build a detector by backend name.

    ``classes`` / ``min_confidence`` wrap the backend in a
    :class:`FilteredDetector` so filtering behaves identically for every model.
    """
    _ensure_builtin_backends()
    key = backend.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown detector backend '{backend}'. Available: {available_detectors()}"
        )
    detector = _REGISTRY[key](**kwargs)
    if classes or min_confidence > 0:
        detector = FilteredDetector(
            detector, classes=classes, min_confidence=min_confidence
        )
    return detector


_BUILTINS_REGISTERED = False


def _ensure_builtin_backends() -> None:
    """Register the backends that ship with Sentinel, on first use.

    Imports are deferred and failures are swallowed so that a missing optional
    dependency (e.g. torch/ultralytics not installed) doesn't break the ones
    that are available. Registration is lazy so that importing a backend module
    directly cannot deadlock on a partially initialised parent module.
    """
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    _BUILTINS_REGISTERED = True

    from .backends.mock import MockDetector

    register_detector("mock", MockDetector)

    try:  # needs opencv, which is a core dependency but keep it non-fatal
        from .backends.blob import BlobDetector
    except Exception:  # noqa: BLE001 - optional backend
        pass
    else:
        register_detector("blob", BlobDetector)

    try:  # pragma: no cover - depends on optional heavy deps
        from .backends.yolo_ultralytics import UltralyticsYOLODetector
    except Exception:  # noqa: BLE001 - optional backend
        pass
    else:  # pragma: no cover
        register_detector("yolo", UltralyticsYOLODetector)
        register_detector("ultralytics", UltralyticsYOLODetector)

__all__ = [
    "Detector",
    "FilteredDetector",
    "Detection",
    "DetectorInfo",
    "create_detector",
    "register_detector",
    "available_detectors",
]
