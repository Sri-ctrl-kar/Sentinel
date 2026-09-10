"""Turn tracked objects into structured events.

The tracker produces a *state* (which entities exist, and where) once per
frame. Downstream layers need *changes* to that state, at a manageable rate.
This module is the bridge: it converts per-frame track state into a sparse,
semantically meaningful event stream.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from ..perception.types import Track
from .schema import (
    ACTION_APPEARED,
    ACTION_DETECTED,
    ACTION_DISAPPEARED,
    ACTION_MOVED,
    Event,
)

TrackLike = Union[Track, Mapping[str, Any]]


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class EventGenerator:
    """Converts tracker output into :class:`~app.events.schema.Event` objects.

    Emitted actions
    ---------------
    ``appeared``
        The first frame in which a track is confirmed.
    ``detected``
        A heartbeat re-stating an entity's state every ``sample_interval``
        seconds, so a consumer always has a recent position without needing to
        replay motion.
    ``moved``
        The entity's centre has travelled more than ``movement_threshold``
        pixels since the last motion anchor.
    ``disappeared``
        The tracker gave up on the entity.

    A frame-by-frame dump would be both enormous and mostly redundant; these
    four actions keep the log small enough to hand to a reasoning layer later
    while preserving when things started, moved and stopped.
    """

    def __init__(
        self,
        sample_interval: float = 1.0,
        movement_threshold: float = 40.0,
        emit_appeared: bool = True,
        emit_disappeared: bool = True,
        source: Optional[str] = None,
    ) -> None:
        self.sample_interval = float(sample_interval)
        self.movement_threshold = float(movement_threshold)
        self.emit_appeared = emit_appeared
        self.emit_disappeared = emit_disappeared
        self.source = source
        self._seen: Dict[str, Dict[str, Any]] = {}
        self._sequence = 0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._seen = {}
        self._sequence = 0

    # ------------------------------------------------------------------
    def process(
        self,
        tracks: Sequence[Track],
        timestamp: float,
        frame_index: Optional[int] = None,
        lost_tracks: Iterable[Track] = (),
    ) -> List[Event]:
        """Emit the events implied by this frame's track state."""
        events: List[Event] = []

        for track in tracks:
            entity_id = track.entity_id
            center = list(track.center)
            state = self._seen.get(entity_id)

            if state is None:
                self._seen[entity_id] = {
                    "first_seen": timestamp,
                    "last_sampled": timestamp,
                    "anchor": center,
                    "last_seen": timestamp,
                    "track": track,
                }
                if self.emit_appeared:
                    events.append(
                        self._build(
                            track,
                            ACTION_APPEARED,
                            timestamp,
                            frame_index,
                            extra={"first_seen": round(timestamp, 4)},
                        )
                    )
                continue

            state["last_seen"] = timestamp
            state["track"] = track

            travelled = _distance(center, state["anchor"])
            if travelled >= self.movement_threshold:
                events.append(
                    self._build(
                        track,
                        ACTION_MOVED,
                        timestamp,
                        frame_index,
                        extra={
                            "displacement_px": round(travelled, 2),
                            "from_position": [round(v, 2) for v in state["anchor"]],
                            "speed_px_per_frame": round(track.speed, 2),
                        },
                    )
                )
                state["anchor"] = center
                # A motion event already reports the current position, so it
                # also counts as a heartbeat.
                state["last_sampled"] = timestamp
                continue

            if timestamp - state["last_sampled"] >= self.sample_interval:
                state["last_sampled"] = timestamp
                events.append(
                    self._build(
                        track,
                        ACTION_DETECTED,
                        timestamp,
                        frame_index,
                        extra={
                            "visible_for_seconds": round(
                                timestamp - state["first_seen"], 3
                            )
                        },
                    )
                )

        if self.emit_disappeared:
            for track in lost_tracks:
                entity_id = track.entity_id
                state = self._seen.pop(entity_id, None)
                extra: Dict[str, Any] = {}
                if state is not None:
                    extra = {
                        "first_seen": round(state["first_seen"], 4),
                        "last_seen": round(state["last_seen"], 4),
                        "visible_for_seconds": round(
                            state["last_seen"] - state["first_seen"], 3
                        ),
                    }
                events.append(
                    self._build(
                        track, ACTION_DISAPPEARED, timestamp, frame_index, extra=extra
                    )
                )

        return events

    def flush(
        self,
        timestamp: float,
        tracks: Sequence[Track] = (),
        frame_index: Optional[int] = None,
    ) -> List[Event]:
        """Close out every entity still open when the stream ends.

        This covers two cases: entities visible in the final frame, and
        entities that vanished too close to the end for the tracker's
        ``max_age`` to expire before the video ran out. Without this, an
        entity's duration would be unknowable downstream.
        """
        if not self.emit_disappeared:
            self._seen.clear()
            return []

        by_id = {t.entity_id: t for t in tracks}
        pending: List[Track] = []
        for entity_id, state in self._seen.items():
            track = by_id.get(entity_id) or state.get("track")
            if track is not None:
                pending.append(track)

        events = self.process([], timestamp, frame_index=frame_index, lost_tracks=pending)
        self._seen.clear()
        return events

    # ------------------------------------------------------------------
    def _build(
        self,
        track: Track,
        action: str,
        timestamp: float,
        frame_index: Optional[int],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Event:
        self._sequence += 1
        cx, cy = track.center
        attributes: Dict[str, Any] = {
            "class_name": track.class_name,
            "class_id": track.class_id,
            "confidence": round(float(track.confidence), 4),
            "track_age": track.age,
        }
        if extra:
            attributes.update(extra)
        return Event(
            timestamp=round(float(timestamp), 4),
            entity_id=track.entity_id,
            action=action,
            attributes=attributes,
            position=[round(cx, 2), round(cy, 2)],
            bbox=[round(float(v), 2) for v in track.bbox],
            frame_index=frame_index,
            track_id=track.track_id,
            source=self.source,
            event_id=f"evt_{self._sequence:06d}",
        )

    # ------------------------------------------------------------------
    # Backwards-compatible stateless helper
    # ------------------------------------------------------------------
    def from_tracking(
        self, tracked_objects: Sequence[TrackLike], timestamp: float
    ) -> List[Event]:
        """Emit one ``detected`` event per tracked object, without state.

        Accepts either :class:`Track` objects or plain dicts with ``id`` /
        ``class_name`` / ``confidence`` / ``position`` keys.
        """
        events: List[Event] = []
        for obj in tracked_objects:
            if isinstance(obj, Track):
                events.append(
                    self._build(obj, ACTION_DETECTED, timestamp, frame_index=None)
                )
                continue
            self._sequence += 1
            events.append(
                Event(
                    timestamp=timestamp,
                    entity_id=str(obj.get("id")),
                    action=ACTION_DETECTED,
                    attributes={
                        "class_name": obj.get("class_name"),
                        "confidence": obj.get("confidence"),
                    },
                    position=obj.get("position"),
                    bbox=obj.get("bbox"),
                    source=self.source,
                    event_id=f"evt_{self._sequence:06d}",
                )
            )
        return events
