"""Turn tracked objects into structured events.

The tracker produces a *state* (which entities exist, and where) once per
frame. Downstream layers need *changes* to that state, at a manageable rate.
This module is the bridge: it converts per-frame track state into a sparse,
semantically meaningful event stream.

It sits between perception and temporal memory and belongs to neither:
it is the last place that sees a :class:`~app.perception.types.Track`, and the
first place that produces an :class:`~app.events.schema.Event`. Nothing here
knows which detector produced the tracks, and nothing here stores history
beyond what is needed to decide whether *this* frame is worth an event —
long-term recall is :mod:`app.memory`'s job.

All coordinates are image pixels; see :mod:`app.spatial`.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from ..perception.types import Track
from ..spatial import IMAGE_PIXELS, ZoneSet, pixel_distance
from .schema import (
    ACTION_APPEARED,
    ACTION_DETECTED,
    ACTION_DISAPPEARED,
    ACTION_ENTERED_ZONE,
    ACTION_EXITED_ZONE,
    ACTION_MOVED,
    ACTION_STATIONARY,
    Event,
)

TrackLike = Union[Track, Mapping[str, Any]]


class EventGenerator:
    """Converts tracker output into :class:`~app.events.schema.Event` objects.

    Emitted actions
    ---------------
    ``appeared``
        The first frame in which a track is confirmed.
    ``moved``
        The entity's centre has travelled more than ``movement_threshold``
        pixels since the last motion anchor.
    ``stationary``
        The entity has stayed within ``stationary_threshold`` pixels for at
        least ``stationary_duration`` seconds. Emitted once per stationary
        episode, and re-armed once the entity moves again.
    ``entered_zone`` / ``exited_zone``
        The entity's anchor point crossed into or out of a configured zone.
        One event per zone, since zones may overlap.
    ``detected``
        A heartbeat re-stating an entity's state every ``sample_interval``
        seconds, so a consumer always has a recent position without needing to
        replay motion.
    ``disappeared``
        The tracker gave up on the entity.

    A frame-by-frame dump would be both enormous and mostly redundant; these
    actions keep the log small enough to hand to a reasoning layer later while
    preserving when things started, moved, settled and stopped.
    """

    def __init__(
        self,
        sample_interval: float = 1.0,
        movement_threshold: float = 40.0,
        stationary_threshold: float = 15.0,
        stationary_duration: float = 2.0,
        zones: Optional[ZoneSet] = None,
        emit_appeared: bool = True,
        emit_disappeared: bool = True,
        source: Optional[str] = None,
    ) -> None:
        self.sample_interval = float(sample_interval)
        self.movement_threshold = float(movement_threshold)
        self.stationary_threshold = float(stationary_threshold)
        self.stationary_duration = float(stationary_duration)
        self.zones = zones if zones is not None else ZoneSet()
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
        """Emit the events implied by this frame's track state.

        Events are returned in a stable order: for each track, presence first,
        then zone transitions, then motion, then the heartbeat. Closing events
        for lost tracks come last.
        """
        events: List[Event] = []

        for track in tracks:
            state = self._seen.get(track.entity_id)
            if state is None:
                events.extend(self._on_first_sighting(track, timestamp, frame_index))
            else:
                events.extend(
                    self._on_continuing(track, state, timestamp, frame_index)
                )

        if self.emit_disappeared:
            for track in lost_tracks:
                events.extend(self._on_lost(track, timestamp, frame_index))

        return events

    # ------------------------------------------------------------------
    def _on_first_sighting(
        self, track: Track, timestamp: float, frame_index: Optional[int]
    ) -> List[Event]:
        position = list(track.center)
        occupied = self._zones_for(track)
        self._seen[track.entity_id] = {
            "first_seen": timestamp,
            "last_seen": timestamp,
            "last_sampled": timestamp,
            "anchor": position,
            "still_anchor": position,
            "still_since": timestamp,
            "is_stationary": False,
            "zones": list(occupied),
            "track": track,
        }

        events: List[Event] = []
        if self.emit_appeared:
            events.append(
                self._build(
                    track,
                    ACTION_APPEARED,
                    timestamp,
                    frame_index,
                    zones=occupied,
                    extra={"first_seen": round(timestamp, 4)},
                )
            )
        # An entity that appears already inside a zone has entered it as far as
        # Sentinel can tell; without this, zone occupancy derived from the
        # event stream would miss it entirely.
        for zone_name in occupied:
            events.append(
                self._build(
                    track,
                    ACTION_ENTERED_ZONE,
                    timestamp,
                    frame_index,
                    zones=occupied,
                    extra={"zone": zone_name, "on_appearance": True},
                )
            )
        return events

    def _on_continuing(
        self,
        track: Track,
        state: Dict[str, Any],
        timestamp: float,
        frame_index: Optional[int],
    ) -> List[Event]:
        position = list(track.center)
        state["last_seen"] = timestamp
        state["track"] = track

        events: List[Event] = []
        occupied = self._zones_for(track)
        events.extend(
            self._zone_transitions(track, state, occupied, timestamp, frame_index)
        )
        state["zones"] = list(occupied)

        moved_event = self._motion(track, state, position, occupied, timestamp, frame_index)
        stationary_event = self._stationary(
            track, state, position, occupied, timestamp, frame_index
        )
        if moved_event is not None:
            events.append(moved_event)
        if stationary_event is not None:
            events.append(stationary_event)

        # A motion or stationary event already reports the current position,
        # so it counts as a heartbeat and suppresses a redundant `detected`.
        if moved_event is None and stationary_event is None:
            if timestamp - state["last_sampled"] >= self.sample_interval:
                state["last_sampled"] = timestamp
                events.append(
                    self._build(
                        track,
                        ACTION_DETECTED,
                        timestamp,
                        frame_index,
                        zones=occupied,
                        extra={
                            "visible_for_seconds": round(
                                timestamp - state["first_seen"], 3
                            )
                        },
                    )
                )
        return events

    def _on_lost(
        self, track: Track, timestamp: float, frame_index: Optional[int]
    ) -> List[Event]:
        state = self._seen.pop(track.entity_id, None)
        occupied = list(state["zones"]) if state else []

        events: List[Event] = []
        # Close out zone occupancy before the entity leaves, so that occupancy
        # reconstructed from events never leaks a departed entity.
        for zone_name in occupied:
            events.append(
                self._build(
                    track,
                    ACTION_EXITED_ZONE,
                    timestamp,
                    frame_index,
                    zones=[],
                    extra={"zone": zone_name, "on_disappearance": True},
                )
            )

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
                track, ACTION_DISAPPEARED, timestamp, frame_index, zones=[], extra=extra
            )
        )
        return events

    # ------------------------------------------------------------------
    def _zones_for(self, track: Track) -> List[str]:
        if not self.zones:
            return []
        return self.zones.containing_box(track.bbox)

    def _zone_transitions(
        self,
        track: Track,
        state: Dict[str, Any],
        occupied: Sequence[str],
        timestamp: float,
        frame_index: Optional[int],
    ) -> List[Event]:
        previous = set(state.get("zones", ()))
        current = set(occupied)
        events: List[Event] = []
        # Sorted so the event order is deterministic when several zones change
        # in the same frame.
        for zone_name in sorted(current - previous):
            events.append(
                self._build(
                    track,
                    ACTION_ENTERED_ZONE,
                    timestamp,
                    frame_index,
                    zones=occupied,
                    extra={"zone": zone_name},
                )
            )
        for zone_name in sorted(previous - current):
            events.append(
                self._build(
                    track,
                    ACTION_EXITED_ZONE,
                    timestamp,
                    frame_index,
                    zones=occupied,
                    extra={"zone": zone_name},
                )
            )
        return events

    def _motion(
        self,
        track: Track,
        state: Dict[str, Any],
        position: List[float],
        occupied: Sequence[str],
        timestamp: float,
        frame_index: Optional[int],
    ) -> Optional[Event]:
        travelled = pixel_distance(position, state["anchor"])
        if travelled < self.movement_threshold:
            return None

        event = self._build(
            track,
            ACTION_MOVED,
            timestamp,
            frame_index,
            zones=occupied,
            extra={
                "displacement_px": round(travelled, 2),
                "from_position": [round(v, 2) for v in state["anchor"]],
                "speed_px_per_frame": round(track.speed, 2),
            },
        )
        state["anchor"] = position
        state["last_sampled"] = timestamp
        return event

    def _stationary(
        self,
        track: Track,
        state: Dict[str, Any],
        position: List[float],
        occupied: Sequence[str],
        timestamp: float,
        frame_index: Optional[int],
    ) -> Optional[Event]:
        drift = pixel_distance(position, state["still_anchor"])
        if drift > self.stationary_threshold:
            # Left the dwell radius: restart the clock and re-arm the event.
            state["still_anchor"] = position
            state["still_since"] = timestamp
            state["is_stationary"] = False
            return None

        if state["is_stationary"]:
            return None
        held_for = timestamp - state["still_since"]
        if held_for < self.stationary_duration:
            return None

        state["is_stationary"] = True
        state["last_sampled"] = timestamp
        return self._build(
            track,
            ACTION_STATIONARY,
            timestamp,
            frame_index,
            zones=occupied,
            extra={
                "stationary_for_seconds": round(held_for, 3),
                "since": round(state["still_since"], 4),
                "drift_px": round(drift, 2),
            },
        )

    # ------------------------------------------------------------------
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

        events: List[Event] = []
        for track in pending:
            events.extend(self._on_lost(track, timestamp, frame_index))
        self._seen.clear()
        return events

    # ------------------------------------------------------------------
    def _build(
        self,
        track: Track,
        action: str,
        timestamp: float,
        frame_index: Optional[int],
        zones: Sequence[str] = (),
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
            coordinate_space=IMAGE_PIXELS,
            zones=list(zones),
            frame_index=frame_index,
            track_id=track.track_id,
            source=self.source,
            event_id=f"evt_{self._sequence:06d}",
        )

    # ------------------------------------------------------------------
    # Backwards-compatible stateless helper (M0.1 API)
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
