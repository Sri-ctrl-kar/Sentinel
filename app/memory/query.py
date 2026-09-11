"""Composable filters over an event stream.

One filter type is shared by every query entry point on
:class:`~app.memory.temporal.TemporalEventMemory`, so "events for this entity",
"events in this interval" and "events of this kind" combine freely instead of
each needing its own method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Sequence

from ..events.schema import Event


@dataclass(frozen=True)
class EventFilter:
    """Predicate over events.

    All supplied criteria must match (logical AND). Any criterion left as
    ``None`` is ignored.

    Time interval semantics
    -----------------------
    ``start`` is **inclusive** and ``end`` is **exclusive**, so adjacent
    windows tile the timeline without double-counting an event that lands
    exactly on a boundary.
    """

    entity_id: Optional[str] = None
    entity_ids: Optional[Sequence[str]] = None
    actions: Optional[Sequence[str]] = None
    class_names: Optional[Sequence[str]] = None
    zone: Optional[str] = None
    track_id: Optional[int] = None
    start: Optional[float] = None
    end: Optional[float] = None
    min_confidence: Optional[float] = None

    def matches(self, event: Event) -> bool:
        if self.entity_id is not None and event.entity_id != self.entity_id:
            return False
        if self.entity_ids is not None and event.entity_id not in self.entity_ids:
            return False
        if self.actions is not None and event.action not in self.actions:
            return False
        if self.track_id is not None and event.track_id != self.track_id:
            return False
        if self.start is not None and event.timestamp < self.start:
            return False
        if self.end is not None and event.timestamp >= self.end:
            return False
        if self.class_names is not None:
            if event.attributes.get("class_name") not in self.class_names:
                return False
        if self.min_confidence is not None:
            confidence = event.attributes.get("confidence")
            if confidence is None or confidence < self.min_confidence:
                return False
        if self.zone is not None:
            # A zone matches either because the entity was inside it when the
            # event fired, or because the event is about that zone.
            if self.zone not in event.zones and event.attributes.get("zone") != self.zone:
                return False
        return True

    def apply(self, events: Iterable[Event]) -> Iterator[Event]:
        return (e for e in events if self.matches(e))

    @property
    def is_empty(self) -> bool:
        """True when this filter would accept every event."""
        return all(
            getattr(self, f) is None
            for f in (
                "entity_id",
                "entity_ids",
                "actions",
                "class_names",
                "zone",
                "track_id",
                "start",
                "end",
                "min_confidence",
            )
        )
