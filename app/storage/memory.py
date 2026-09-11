"""In-memory event store with JSON persistence.

M0.1 keeps the whole run in memory and writes a single JSON document at the
end. That is deliberate: it makes the output trivially inspectable and
diffable while the schema is still moving. Swapping in a streaming writer or a
real database later is a change behind this same small interface.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

from ..events.schema import Event, EventLog


class EventMemory:
    """Append-only collection of events for a single run."""

    def __init__(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.events: List[Event] = []
        self.metadata: Dict[str, Any] = dict(metadata or {})

    def add(self, event: Event) -> None:
        self.events.append(event)

    def extend(self, events: Iterable[Event]) -> None:
        self.events.extend(events)

    def all(self) -> List[Event]:
        return list(self.events)

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    # ------------------------------------------------------------------
    def by_entity(self, entity_id: str) -> List[Event]:
        return [e for e in self.events if e.entity_id == entity_id]

    def by_action(self, action: str) -> List[Event]:
        return [e for e in self.events if e.action == action]

    def entities(self) -> List[str]:
        """Distinct entity IDs, in order of first appearance."""
        seen: Dict[str, None] = {}
        for event in self.events:
            seen.setdefault(event.entity_id, None)
        return list(seen)

    def summary(self) -> Dict[str, Any]:
        actions: Dict[str, int] = {}
        classes: Dict[str, int] = {}
        for event in self.events:
            actions[event.action] = actions.get(event.action, 0) + 1
            name = event.attributes.get("class_name")
            if name:
                classes[name] = classes.get(name, 0) + 1
        return {
            "total_events": len(self.events),
            "unique_entities": len(self.entities()),
            "events_by_action": actions,
            "events_by_class": classes,
        }

    # ------------------------------------------------------------------
    def to_log(self) -> EventLog:
        metadata = dict(self.metadata)
        metadata["summary"] = self.summary()
        return EventLog(events=self.all(), metadata=metadata)

    def to_dict(self) -> Dict[str, Any]:
        return self.to_log().to_dict()

    def save_json(self, path: str, indent: int = 2) -> str:
        """Write the run to ``path`` as a single JSON document."""
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=indent)
            handle.write("\n")
        return path

    @classmethod
    def load_json(cls, path: str) -> "EventMemory":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        store = cls(metadata=payload.get("metadata", {}))
        store.extend(Event.from_dict(e) for e in payload.get("events", []))
        return store
