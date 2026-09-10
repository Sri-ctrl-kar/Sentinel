"""Structured event schema.

Events are Sentinel's durable output: the perception layer's job ends here, and
every later layer (risk engine, reasoning, UI) consumes *only* these. Keeping
the schema flat and JSON-native means no downstream component ever needs to
import a model library to read them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Event action vocabulary. Deliberately small at M0.1 — richer semantics
# (interactions, zones, intent) belong to later milestones.
ACTION_APPEARED = "appeared"
ACTION_DETECTED = "detected"
ACTION_MOVED = "moved"
ACTION_DISAPPEARED = "disappeared"

ACTIONS = (ACTION_APPEARED, ACTION_DETECTED, ACTION_MOVED, ACTION_DISAPPEARED)

SCHEMA_VERSION = "0.1"


@dataclass
class Event:
    """One structured observation about one entity at one moment in time.

    Attributes
    ----------
    timestamp:
        Seconds since the start of the video.
    entity_id:
        Persistent identity assigned by the tracker (e.g. ``person_3``).
    action:
        One of :data:`ACTIONS`.
    attributes:
        Free-form payload; always carries ``class_name`` and ``confidence``.
    position:
        ``[cx, cy]`` centre point of the entity, in pixels.
    bbox:
        ``[x1, y1, x2, y2]`` bounding box, in pixels.
    """

    timestamp: float
    entity_id: str
    action: str
    attributes: Dict[str, Any]
    position: Optional[List[float]] = None
    bbox: Optional[List[float]] = None
    frame_index: Optional[int] = None
    track_id: Optional[int] = None
    source: Optional[str] = None
    event_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable representation, omitting unset optional fields."""
        payload: Dict[str, Any] = {
            "event_id": self.event_id,
            "timestamp": round(float(self.timestamp), 4),
            "frame_index": self.frame_index,
            "entity_id": self.entity_id,
            "track_id": self.track_id,
            "action": self.action,
            "position": [round(float(v), 2) for v in self.position]
            if self.position
            else None,
            "bbox": [round(float(v), 2) for v in self.bbox] if self.bbox else None,
            "attributes": self.attributes,
        }
        if self.source is not None:
            payload["source"] = self.source
        return {k: v for k, v in payload.items() if v is not None}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Event":
        return cls(
            timestamp=payload["timestamp"],
            entity_id=payload["entity_id"],
            action=payload["action"],
            attributes=payload.get("attributes", {}),
            position=payload.get("position"),
            bbox=payload.get("bbox"),
            frame_index=payload.get("frame_index"),
            track_id=payload.get("track_id"),
            source=payload.get("source"),
            event_id=payload.get("event_id"),
        )


@dataclass
class EventLog:
    """An ordered set of events plus the context needed to interpret them."""

    events: List[Event] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "metadata": self.metadata,
            "event_count": len(self.events),
            "events": [e.to_dict() for e in self.events],
        }
