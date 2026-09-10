"""Chronological, queryable memory of everything each entity has done.

Design notes
------------
* **Insertion keeps chronological order.** Events may be ingested out of
  order (a replayed log, a merged file, a future multi-camera source) and the
  stream still reads chronologically. Ordering is by
  ``(timestamp, event_id, ingest_sequence)``: ``event_id`` breaks ties within
  a frame in generation order, and the ingest counter is a final tiebreak so
  ordering is total and stable.
* **State is derived, never authored.** :class:`EntityState` is a fold over an
  entity's events, so it can always be rebuilt — which is what makes
  :meth:`TemporalEventMemory.state_at` (state as of an arbitrary moment)
  possible and what keeps out-of-order ingestion correct.
* **No model-specific code.** See the package docstring.

Coordinates stored here are image pixels. :mod:`app.spatial` explains why that
must not be read as real-world distance.
"""

from __future__ import annotations

import bisect
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from ..events.schema import (
    ACTION_DISAPPEARED,
    ACTION_ENTERED_ZONE,
    ACTION_EXITED_ZONE,
    SCHEMA_VERSION,
    Event,
)
from ..spatial import IMAGE_PIXELS
from .query import EventFilter


@dataclass
class EntityState:
    """The latest known state of one entity, folded from its events."""

    entity_id: str
    class_name: Optional[str] = None
    class_id: Optional[int] = None
    track_id: Optional[int] = None
    confidence: Optional[float] = None
    position: Optional[List[float]] = None
    bbox: Optional[List[float]] = None
    coordinate_space: str = IMAGE_PIXELS
    zones: List[str] = field(default_factory=list)
    first_seen: Optional[float] = None
    last_seen: Optional[float] = None
    last_action: Optional[str] = None
    present: bool = False
    event_count: int = 0

    @property
    def duration(self) -> float:
        """Seconds between first and last observation."""
        if self.first_seen is None or self.last_seen is None:
            return 0.0
        return round(self.last_seen - self.first_seen, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "track_id": self.track_id,
            "confidence": self.confidence,
            "position": self.position,
            "bbox": self.bbox,
            "coordinate_space": self.coordinate_space,
            "zones": list(self.zones),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "duration_seconds": self.duration,
            "last_action": self.last_action,
            "present": self.present,
            "event_count": self.event_count,
        }


@dataclass
class EntityTimeline:
    """One entity's full history plus its folded state."""

    entity_id: str
    events: List[Event]
    state: EntityState

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self.events)

    @property
    def actions(self) -> List[str]:
        return [e.action for e in self.events]

    @property
    def zones_visited(self) -> List[str]:
        """Every zone this entity entered, in order of first entry."""
        seen: Dict[str, None] = {}
        for event in self.events:
            if event.action == ACTION_ENTERED_ZONE:
                zone = event.attributes.get("zone")
                if zone:
                    seen.setdefault(zone, None)
        return list(seen)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "state": self.state.to_dict(),
            "zones_visited": self.zones_visited,
            "events": [e.to_dict() for e in self.events],
        }


# Sort key: timestamp, then generation order within the frame, then arrival.
_SortKey = Tuple[float, str, int]


class TemporalEventMemory:
    """A persistent, chronological, queryable event history.

    This is the M0.2 deliverable: perception decides what is on screen now,
    the event generator decides what is worth recording, and this class is
    what Sentinel actually reasons over later.
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self._keys: List[_SortKey] = []
        self._events: List[Event] = []
        # Per-entity index, kept as (sort key, event) so an out-of-order
        # arrival can be placed without re-deriving keys or relying on event
        # equality (two events may legitimately compare equal).
        self._by_entity: Dict[str, List[Tuple[_SortKey, Event]]] = {}
        self._states: Dict[str, EntityState] = {}
        self._ingest_count = 0

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest(self, event: Event) -> None:
        """Add one event, preserving chronological order."""
        key: _SortKey = (
            float(event.timestamp),
            event.event_id or "",
            self._ingest_count,
        )
        self._ingest_count += 1

        index = bisect.bisect_right(self._keys, key)
        self._keys.insert(index, key)
        self._events.insert(index, event)

        history = self._by_entity.setdefault(event.entity_id, [])
        entity_index = bisect.bisect_right([k for k, _ in history], key)
        in_order = entity_index == len(history)
        history.insert(entity_index, (key, event))

        if in_order:
            self._apply(event)
        else:
            # Out-of-order arrival: a fold cannot be repaired in place, so
            # replay this entity's history from the start.
            self._rebuild_state(event.entity_id)

    def ingest_many(self, events: Iterable[Event]) -> None:
        for event in events:
            self.ingest(event)

    @classmethod
    def from_events(
        cls, events: Iterable[Event], metadata: Optional[Dict[str, Any]] = None
    ) -> "TemporalEventMemory":
        memory = cls(metadata=metadata)
        memory.ingest_many(events)
        return memory

    def _entity_events(self, entity_id: str) -> List[Event]:
        return [event for _key, event in self._by_entity.get(entity_id, [])]

    # ------------------------------------------------------------------
    # State folding
    # ------------------------------------------------------------------
    def _apply(self, event: Event, state: Optional[EntityState] = None) -> EntityState:
        """Fold one event into an entity's state."""
        if state is None:
            state = self._states.get(event.entity_id)
            if state is None:
                state = EntityState(entity_id=event.entity_id)
                self._states[event.entity_id] = state

        state.event_count += 1
        state.last_action = event.action
        state.last_seen = event.timestamp
        if state.first_seen is None:
            state.first_seen = event.timestamp

        if event.position is not None:
            state.position = list(event.position)
        if event.bbox is not None:
            state.bbox = list(event.bbox)
        if event.track_id is not None:
            state.track_id = event.track_id
        state.coordinate_space = event.coordinate_space

        class_name = event.attributes.get("class_name")
        if class_name is not None:
            state.class_name = class_name
        class_id = event.attributes.get("class_id")
        if class_id is not None:
            state.class_id = class_id
        confidence = event.attributes.get("confidence")
        if confidence is not None:
            state.confidence = confidence

        if event.action == ACTION_DISAPPEARED:
            state.present = False
            state.zones = []
        else:
            state.present = True
            # `zones` on the event is the authoritative occupancy snapshot;
            # zone events additionally name the zone that changed.
            if event.zones:
                state.zones = list(event.zones)
            elif event.action == ACTION_EXITED_ZONE:
                zone = event.attributes.get("zone")
                state.zones = [z for z in state.zones if z != zone]
            elif event.action == ACTION_ENTERED_ZONE:
                zone = event.attributes.get("zone")
                if zone and zone not in state.zones:
                    state.zones = state.zones + [zone]
            else:
                state.zones = []
        return state

    def _rebuild_state(self, entity_id: str) -> EntityState:
        state = EntityState(entity_id=entity_id)
        self._states[entity_id] = state
        for event in self._entity_events(entity_id):
            self._apply(event, state)
        return state

    # ------------------------------------------------------------------
    # Basic access
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    def __contains__(self, entity_id: object) -> bool:
        return entity_id in self._by_entity

    @property
    def events(self) -> List[Event]:
        """Every event, chronologically."""
        return list(self._events)

    @property
    def span(self) -> Tuple[Optional[float], Optional[float]]:
        """``(first_timestamp, last_timestamp)`` across the whole memory."""
        if not self._events:
            return (None, None)
        return (self._events[0].timestamp, self._events[-1].timestamp)

    def entities(self) -> List[str]:
        """Every entity ID ever seen, in order of first appearance."""
        return [
            state.entity_id
            for state in sorted(
                self._states.values(),
                key=lambda s: (s.first_seen if s.first_seen is not None else 0.0),
            )
        ]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def stream(
        self,
        filter: Optional[EventFilter] = None,
        limit: Optional[int] = None,
        **criteria: Any,
    ) -> List[Event]:
        """The chronological event stream, optionally filtered.

        ``criteria`` are forwarded to :class:`~app.memory.query.EventFilter`,
        so ``memory.stream(entity_id="person_1", actions=["moved"])`` works
        without building a filter by hand.
        """
        active = filter or EventFilter(**criteria)
        # Entity-scoped queries read the per-entity index instead of scanning.
        if active.entity_id is not None:
            source: Sequence[Event] = self._entity_events(active.entity_id)
        else:
            source = self._events

        results = [e for e in source if active.matches(e)]
        return results[:limit] if limit is not None else results

    def entity_history(self, entity_id: str, **criteria: Any) -> List[Event]:
        """Every event for one entity, chronologically."""
        return self.stream(entity_id=entity_id, **criteria)

    def between(self, start: float, end: float, **criteria: Any) -> List[Event]:
        """Events in the interval ``[start, end)`` — start inclusive, end exclusive."""
        return self.stream(start=start, end=end, **criteria)

    def latest_state(self, entity_id: str) -> Optional[EntityState]:
        """The most recent known state of an entity, or ``None`` if unknown."""
        return self._states.get(entity_id)

    def states(self) -> Dict[str, EntityState]:
        return dict(self._states)

    def state_at(self, entity_id: str, at: float) -> Optional[EntityState]:
        """Reconstruct an entity's state as of time ``at`` (inclusive).

        Returns ``None`` if the entity had not been observed by then.
        """
        history = [e for e in self._entity_events(entity_id) if e.timestamp <= at]
        if not history:
            return None
        state = EntityState(entity_id=entity_id)
        for event in history:
            self._apply(event, state)
        return state

    def present_entities(self, at: Optional[float] = None) -> List[str]:
        """Entities currently on screen — appeared and not yet disappeared.

        With ``at`` given, answers as of that moment instead of now.
        """
        if at is None:
            return [
                state.entity_id
                for state in self._states.values()
                if state.present
            ]
        present: List[str] = []
        for entity_id in self._by_entity:
            state = self.state_at(entity_id, at)
            if state is not None and state.present:
                present.append(entity_id)
        return sorted(
            present,
            key=lambda eid: self._states[eid].first_seen
            if self._states[eid].first_seen is not None
            else 0.0,
        )

    def zone_occupancy(self, zone: str, at: Optional[float] = None) -> List[str]:
        """Which entities are inside ``zone`` (now, or as of ``at``)."""
        if at is None:
            return [
                state.entity_id
                for state in self._states.values()
                if state.present and zone in state.zones
            ]
        occupants: List[str] = []
        for entity_id in self._by_entity:
            state = self.state_at(entity_id, at)
            if state is not None and state.present and zone in state.zones:
                occupants.append(entity_id)
        return occupants

    def timeline(self, entity_id: str) -> Optional[EntityTimeline]:
        """One entity's events plus its folded state."""
        if entity_id not in self._by_entity:
            return None
        return EntityTimeline(
            entity_id=entity_id,
            events=self._entity_events(entity_id),
            state=self._states[entity_id],
        )

    def timelines(self) -> List[EntityTimeline]:
        return [t for t in (self.timeline(e) for e in self.entities()) if t is not None]

    # ------------------------------------------------------------------
    # Reporting / persistence
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        actions: Dict[str, int] = {}
        classes: Dict[str, int] = {}
        for event in self._events:
            actions[event.action] = actions.get(event.action, 0) + 1
            name = event.attributes.get("class_name")
            if name:
                classes[name] = classes.get(name, 0) + 1
        first, last = self.span
        return {
            "total_events": len(self._events),
            "unique_entities": len(self._states),
            "entities_present": len(self.present_entities()),
            "events_by_action": actions,
            "events_by_class": classes,
            "time_span_seconds": round(last - first, 3)
            if first is not None and last is not None
            else 0.0,
            "coordinate_space": IMAGE_PIXELS,
        }

    def to_dict(self, include_events: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "coordinate_space": IMAGE_PIXELS,
            "metadata": {**self.metadata, "summary": self.summary()},
            "entities": {
                entity_id: state.to_dict() for entity_id, state in self._states.items()
            },
            "event_count": len(self._events),
        }
        if include_events:
            payload["events"] = [e.to_dict() for e in self._events]
        return payload

    def save_json(self, path: str, indent: int = 2) -> str:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=indent)
            handle.write("\n")
        return path

    @classmethod
    def load_json(cls, path: str) -> "TemporalEventMemory":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        memory = cls(metadata=payload.get("metadata", {}))
        memory.ingest_many(Event.from_dict(e) for e in payload.get("events", []))
        return memory
