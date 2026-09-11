"""Persistence and reporting for the temporal memory."""

import json

import pytest

from app.events.schema import ACTION_MOVED, SCHEMA_VERSION, Event
from app.memory import TemporalEventMemory
from app.spatial import IMAGE_PIXELS, Zone, ZoneSet

from helpers import hold, make_detection, run_scenario, walk

ZONES = ZoneSet([Zone.from_rect("loading_bay", (200, 0, 400, 400))])


def test_summary_reports_actions_classes_and_span():
    memory = run_scenario(walk(0, 20, 16), zones=ZONES)
    summary = memory.summary()

    assert summary["total_events"] == len(memory)
    assert summary["unique_entities"] == 1
    assert summary["events_by_class"]["person"] == len(memory)
    assert summary["coordinate_space"] == IMAGE_PIXELS
    assert summary["time_span_seconds"] > 0


def test_summary_records_the_coordinate_space_explicitly():
    """Pixel space must be stated, never assumed by a consumer."""
    memory = run_scenario(walk(0, 20, 6))
    assert memory.summary()["coordinate_space"] == "image_pixels"
    assert all(e.coordinate_space == "image_pixels" for e in memory)


def test_span_of_an_empty_memory():
    memory = TemporalEventMemory()
    assert memory.span == (None, None)
    assert memory.entities() == []
    assert memory.present_entities() == []
    assert memory.summary()["total_events"] == 0


def test_save_json_includes_events_and_folded_state(tmp_path):
    memory = run_scenario(walk(0, 20, 16), zones=ZONES)
    path = memory.save_json(str(tmp_path / "memory.json"))

    payload = json.loads(open(path, encoding="utf-8").read())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["coordinate_space"] == IMAGE_PIXELS
    assert payload["event_count"] == len(memory)
    assert "person_1" in payload["entities"]

    state = payload["entities"]["person_1"]
    assert state["class_name"] == "person"
    assert state["coordinate_space"] == IMAGE_PIXELS
    assert state["event_count"] == len(memory.entity_history("person_1"))


def test_json_round_trip_preserves_history_and_state(tmp_path):
    memory = run_scenario(walk(0, 20, 16), zones=ZONES)
    path = memory.save_json(str(tmp_path / "memory.json"))

    reloaded = TemporalEventMemory.load_json(path)
    assert [e.event_id for e in reloaded] == [e.event_id for e in memory]
    assert reloaded.latest_state("person_1").to_dict() == (
        memory.latest_state("person_1").to_dict()
    )
    assert reloaded.entities() == memory.entities()


def test_json_round_trip_preserves_zone_membership(tmp_path):
    memory = run_scenario(hold(250, 6), zones=ZONES)
    path = memory.save_json(str(tmp_path / "memory.json"))
    reloaded = TemporalEventMemory.load_json(path)

    original_zones = [e.zones for e in memory]
    assert [e.zones for e in reloaded] == original_zones


def test_to_dict_can_omit_the_event_list():
    memory = run_scenario(walk(0, 20, 8))
    payload = memory.to_dict(include_events=False)
    assert "events" not in payload
    assert payload["event_count"] == len(memory)


def test_memory_can_be_built_from_a_plain_event_list():
    """The memory is an index over events, so any event source works."""
    events = [
        Event(0.0, "person_1", "appeared", {"class_name": "person"}, event_id="evt_1"),
        Event(1.0, "person_1", ACTION_MOVED, {"class_name": "person"}, event_id="evt_2"),
    ]
    memory = TemporalEventMemory.from_events(events)
    assert len(memory) == 2
    assert memory.latest_state("person_1").present is True


def test_membership_check():
    memory = run_scenario(walk(0, 20, 6))
    assert "person_1" in memory
    assert "ghost_1" not in memory


def test_state_at_returns_none_before_the_entity_existed():
    memory = run_scenario(walk(0, 20, 10))
    assert memory.state_at("person_1", -1.0) is None
    assert memory.state_at("person_1", 0.0) is not None


def test_state_at_is_a_point_in_time_snapshot():
    memory = run_scenario(walk(0, 20, 21), movement_threshold=40.0)
    early = memory.state_at("person_1", 0.4)
    late = memory.state_at("person_1", 1.6)
    assert early.position[0] < late.position[0]
    assert early.event_count < late.event_count


def test_timelines_covers_every_entity():
    frames = [
        [
            make_detection(i * 20, 100, class_id=0, class_name="person"),
            make_detection(600 - i * 20, 250, class_id=7, class_name="truck"),
        ]
        for i in range(12)
    ]
    memory = run_scenario(frames)
    timelines = memory.timelines()
    assert {t.entity_id for t in timelines} == set(memory.entities())
    assert sum(len(t) for t in timelines) == len(memory)
