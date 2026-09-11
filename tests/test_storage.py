"""Tests for the in-memory event store and its JSON persistence."""

import json
import os

from app.events.generator import EventGenerator
from app.events.schema import Event
from app.storage.memory import EventMemory
from app.perception.types import Track


def make_event(entity_id="person_1", action="detected", timestamp=0.0, class_name="person"):
    return Event(
        timestamp=timestamp,
        entity_id=entity_id,
        action=action,
        attributes={"class_name": class_name, "confidence": 0.9},
        position=[10.0, 20.0],
        bbox=[0.0, 0.0, 20.0, 40.0],
        event_id=f"evt_{entity_id}_{action}_{timestamp}",
    )


def test_add_and_all():
    memory = EventMemory()
    memory.add(make_event())
    assert len(memory) == 1
    assert memory.all()[0].entity_id == "person_1"


def test_extend_and_iterate():
    memory = EventMemory()
    memory.extend([make_event(timestamp=t) for t in (0.0, 1.0, 2.0)])
    assert [e.timestamp for e in memory] == [0.0, 1.0, 2.0]


def test_query_by_entity_and_action():
    memory = EventMemory()
    memory.extend(
        [
            make_event(entity_id="person_1", action="appeared"),
            make_event(entity_id="truck_2", action="moved"),
            make_event(entity_id="person_1", action="moved"),
        ]
    )
    assert len(memory.by_entity("person_1")) == 2
    assert len(memory.by_action("moved")) == 2


def test_entities_are_in_first_appearance_order():
    memory = EventMemory()
    memory.extend(
        [
            make_event(entity_id="truck_2"),
            make_event(entity_id="person_1"),
            make_event(entity_id="truck_2"),
        ]
    )
    assert memory.entities() == ["truck_2", "person_1"]


def test_summary_counts_actions_and_classes():
    memory = EventMemory()
    memory.extend(
        [
            make_event(entity_id="person_1", action="appeared"),
            make_event(entity_id="person_1", action="moved"),
            make_event(entity_id="truck_2", action="appeared", class_name="truck"),
        ]
    )
    summary = memory.summary()
    assert summary["total_events"] == 3
    assert summary["unique_entities"] == 2
    assert summary["events_by_action"] == {"appeared": 2, "moved": 1}
    assert summary["events_by_class"] == {"person": 2, "truck": 1}


def test_save_json_writes_valid_document(tmp_path):
    memory = EventMemory(metadata={"video": {"path": "clip.mp4"}})
    memory.add(make_event())
    path = memory.save_json(str(tmp_path / "events.json"))

    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["schema_version"]
    assert payload["event_count"] == 1
    assert payload["metadata"]["video"]["path"] == "clip.mp4"
    assert payload["metadata"]["summary"]["total_events"] == 1
    event = payload["events"][0]
    assert event["entity_id"] == "person_1"
    assert event["attributes"]["confidence"] == 0.9
    assert event["bbox"] == [0.0, 0.0, 20.0, 40.0]


def test_save_json_creates_missing_directories(tmp_path):
    memory = EventMemory()
    memory.add(make_event())
    path = memory.save_json(str(tmp_path / "nested" / "deeper" / "events.json"))
    assert os.path.exists(path)


def test_json_round_trip(tmp_path):
    memory = EventMemory(metadata={"video": {"path": "clip.mp4"}})
    memory.extend([make_event(timestamp=t) for t in (0.0, 1.0)])
    path = memory.save_json(str(tmp_path / "events.json"))

    reloaded = EventMemory.load_json(path)
    assert len(reloaded) == 2
    assert [e.timestamp for e in reloaded] == [0.0, 1.0]
    assert reloaded.events[0].bbox == [0.0, 0.0, 20.0, 40.0]


def test_generated_events_serialise_cleanly(tmp_path):
    """Guard against a non-JSON-serialisable field sneaking into an event."""
    track = Track(
        track_id=1,
        bbox=(10, 20, 50, 110),
        confidence=0.91,
        class_id=0,
        class_name="person",
    )
    generator = EventGenerator(source="clip.mp4")
    memory = EventMemory()
    memory.extend(generator.process([track], 0.0, frame_index=0))
    memory.extend(generator.flush(1.0, [track]))

    path = memory.save_json(str(tmp_path / "events.json"))
    payload = json.loads(open(path, encoding="utf-8").read())
    assert payload["event_count"] == 2
