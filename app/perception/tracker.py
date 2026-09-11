"""Model-agnostic multi-object tracking interface.

The tracker turns per-frame :class:`~app.perception.types.Detection` objects
into :class:`~app.perception.types.Track` objects carrying persistent IDs.
Persistent identity is the whole point: Sentinel's downstream layers reason
about *entities over time*, not about per-frame labels.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Sequence

from .types import Detection, Track


class Tracker(ABC):
    """Base class for every tracking strategy."""

    name: str = "base"

    @abstractmethod
    def update(self, detections: Sequence[Detection], timestamp: float) -> List[Track]:
        """Associate ``detections`` with existing tracks.

        Returns the list of tracks that are *confirmed and visible* in the
        current frame.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Drop all state (used between videos and in tests)."""
        raise NotImplementedError

    @property
    def lost_tracks(self) -> List[Track]:
        """Tracks removed since the previous :meth:`update` call."""
        return []


_REGISTRY: Dict[str, Callable[..., Tracker]] = {}


def register_tracker(name: str, factory: Callable[..., Tracker]) -> None:
    _REGISTRY[name.lower()] = factory


def available_trackers() -> List[str]:
    _ensure_builtin_trackers()
    return sorted(_REGISTRY)


def create_tracker(strategy: str = "byte_iou", **kwargs: Any) -> Tracker:
    _ensure_builtin_trackers()
    key = strategy.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown tracker '{strategy}'. Available: {available_trackers()}"
        )
    return _REGISTRY[key](**kwargs)


_BUILTINS_REGISTERED = False


def _ensure_builtin_trackers() -> None:
    """Register the bundled trackers on first use.

    Registration is lazy rather than import-time so that importing a tracker
    module directly (``from .trackers.byte_iou import ByteIoUTracker``) cannot
    deadlock on a partially initialised parent module.
    """
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    _BUILTINS_REGISTERED = True

    from .trackers.byte_iou import ByteIoUTracker

    register_tracker("byte_iou", ByteIoUTracker)
    register_tracker("bytetrack", ByteIoUTracker)
    register_tracker("iou", ByteIoUTracker)

__all__ = [
    "Tracker",
    "Track",
    "create_tracker",
    "register_tracker",
    "available_trackers",
]
