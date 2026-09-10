"""Tests for structured event generation."""

import pytest

from app.events.generator import EventGenerator
from app.events.schema import (
    ACTION_APPEARED,
    ACTION_DETECTED,
    ACTION_DISAPPEARED,
    ACTION_ENTERED_ZONE,
    ACTION_EXITED_ZONE,
    ACTION_MOVED,
    ACTION_STATIONARY,
    Event,
    EventLog,
)
from app.perception.types import Track
from app.spatial import Zone, ZoneSet


def make_track(track_id=1, x=100.0, y=200.0, class_name="person", class_id=0, confidence=0.9):
    return Track(
        track_id=track_id,
        bbox=(x, y, x + 40, y + 90),
        confidence=confidence,
        class_id=class_id,
        class_name=class_name,
        age=1,
        hits=3,
    )


# ---------------------------------------------------------------------------
# Backwards-compatible helper
# ---------------------------------------------------------------------------
def test_event_generator():
    events = EventGenerator().from_tracking(
        [{"id": "worker_01", "class_name": "person", "confidence": 0.95, "position": [100, 200]}],
        1.5,
    )
    assert len(events) == 1
    assert events[0].entity_id == "worker_01"
    assert events[0].action == "detected"


def test_from_tracking_accepts_track_objects():
    events = EventGenerator().from_tracking([make_track()], 2.0)
    assert events[0].entity_id == "person_1"
    assert events[0].action == ACTION_DETECTED
    assert events[0].bbox == [100.0, 200.0, 140.0, 290.0]


# ---------------------------------------------------------------------------
# Required event content
# ---------------------------------------------------------------------------
def test_events_carry_timestamp_class_confidence_and_position():
    events = EventGenerator().process([make_track(confidence=0.87)], timestamp=1.25, frame_index=25)
    event = events[0]
    assert event.timestamp == 1.25
    assert event.frame_index == 25
    assert event.attributes["class_name"] == "person"
    assert event.attributes["confidence"] == 0.87
    assert event.position == [120.0, 245.0]
    assert event.bbox == [100.0, 200.0, 140.0, 290.0]
    assert event.track_id == 1


def test_first_sighting_emits_appeared():
    events = EventGenerator().process([make_track()], timestamp=0.0)
    assert [e.action for e in events] == [ACTION_APPEARED]


def test_appeared_is_emitted_only_once_per_entity():
    generator = EventGenerator(sample_interval=999, movement_threshold=999)
    generator.process([make_track()], 0.0)
    later = generator.process([make_track()], 0.1)
    assert all(e.action != ACTION_APPEARED for e in later)


def test_movement_beyond_threshold_emits_moved():
    generator = EventGenerator(movement_threshold=30.0, sample_interval=999)
    generator.process([make_track(x=100)], 0.0)
    events = generator.process([make_track(x=200)], 0.5)
    assert [e.action for e in events] == [ACTION_MOVED]
    assert events[0].attributes["displacement_px"] == 100.0
    assert events[0].attributes["from_position"] == [120.0, 245.0]


def test_movement_below_threshold_emits_nothing():
    generator = EventGenerator(movement_threshold=100.0, sample_interval=999)
    generator.process([make_track(x=100)], 0.0)
    assert generator.process([make_track(x=105)], 0.1) == []


def test_stationary_entity_still_heartbeats():
    generator = EventGenerator(sample_interval=1.0, movement_threshold=1000.0)
    generator.process([make_track()], 0.0)
    assert generator.process([make_track()], 0.5) == []
    events = generator.process([make_track()], 1.2)
    assert [e.action for e in events] == [ACTION_DETECTED]
    assert events[0].attributes["visible_for_seconds"] == 1.2


def test_lost_track_emits_disappeared_with_duration():
    generator = EventGenerator()
    track = make_track()
    generator.process([track], 0.0)
    generator.process([track], 2.0)
    events = generator.process([], 3.0, lost_tracks=[track])
    assert [e.action for e in events] == [ACTION_DISAPPEARED]
    assert events[0].attributes["visible_for_seconds"] == 2.0


def test_flush_closes_entities_still_visible_at_end_of_stream():
    generator = EventGenerator()
    track = make_track()
    generator.process([track], 0.0)
    events = generator.flush(5.0, [track])
    assert [e.action for e in events] == [ACTION_DISAPPEARED]


def test_flush_closes_entities_that_vanished_near_the_end():
    # The tracker's max_age may not expire before the video runs out; the
    # entity must still be closed out.
    generator = EventGenerator()
    generator.process([make_track()], 0.0)
    events = generator.flush(5.0, tracks=[])
    assert [e.action for e in events] == [ACTION_DISAPPEARED]


def test_flush_is_idempotent():
    generator = EventGenerator()
    track = make_track()
    generator.process([track], 0.0)
    generator.flush(1.0, [track])
    assert generator.flush(2.0, [track]) == []


def test_multiple_entities_are_tracked_independently():
    generator = EventGenerator(movement_threshold=30.0, sample_interval=999)
    a = make_track(track_id=1, x=100)
    b = make_track(track_id=2, x=400, class_name="truck", class_id=7)
    generator.process([a, b], 0.0)
    events = generator.process([make_track(track_id=1, x=200), b], 0.5)
    assert [(e.entity_id, e.action) for e in events] == [("person_1", ACTION_MOVED)]


def test_event_ids_are_unique_and_ordered():
    generator = EventGenerator(movement_threshold=1.0)
    events = []
    for i in range(5):
        events.extend(generator.process([make_track(x=100 + i * 50)], i * 0.1))
    ids = [e.event_id for e in events]
    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids)


def test_reset_clears_generator_state():
    generator = EventGenerator()
    generator.process([make_track()], 0.0)
    generator.reset()
    events = generator.process([make_track()], 0.0)
    assert [e.action for e in events] == [ACTION_APPEARED]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def test_event_to_dict_round_trips():
    original = EventGenerator(source="clip.mp4").process([make_track()], 1.0, frame_index=20)[0]
    restored = Event.from_dict(original.to_dict())
    assert restored == original


def test_event_to_dict_omits_unset_fields():
    event = Event(timestamp=1.0, entity_id="person_1", action="detected", attributes={})
    payload = event.to_dict()
    assert "bbox" not in payload
    assert "source" not in payload
    assert payload["entity_id"] == "person_1"


def test_event_log_to_dict_has_schema_version():
    log = EventLog(events=[], metadata={"video": "x.mp4"})
    payload = log.to_dict()
    assert payload["schema_version"]
    assert payload["event_count"] == 0


# ---------------------------------------------------------------------------
# M0.2: stationary behaviour
# ---------------------------------------------------------------------------
def test_stationary_fires_once_after_the_dwell_time():
    generator = EventGenerator(
        sample_interval=999, movement_threshold=999, stationary_duration=1.0
    )
    track = make_track()
    assert generator.process([track], 0.0)[0].action == ACTION_APPEARED
    assert generator.process([track], 0.5) == []

    events = generator.process([track], 1.0)
    assert [e.action for e in events] == [ACTION_STATIONARY]
    assert events[0].attributes["stationary_for_seconds"] == 1.0

    # Already stationary: no repeat.
    assert generator.process([track], 1.5) == []


def test_small_drift_does_not_break_stationary():
    generator = EventGenerator(
        sample_interval=999, movement_threshold=999,
        stationary_duration=1.0, stationary_threshold=20.0,
    )
    generator.process([make_track(x=100)], 0.0)
    events = generator.process([make_track(x=110)], 1.0)
    assert [e.action for e in events] == [ACTION_STATIONARY]


def test_leaving_the_dwell_radius_restarts_the_stationary_clock():
    generator = EventGenerator(
        sample_interval=999, movement_threshold=999,
        stationary_duration=1.0, stationary_threshold=10.0,
    )
    generator.process([make_track(x=100)], 0.0)
    assert generator.process([make_track(x=300)], 0.9) == []
    # The clock restarted at 0.9, so 1.5 is only 0.6s of dwell.
    assert generator.process([make_track(x=300)], 1.5) == []
    events = generator.process([make_track(x=300)], 1.95)
    assert [e.action for e in events] == [ACTION_STATIONARY]


def test_a_moved_event_suppresses_the_heartbeat_in_the_same_frame():
    generator = EventGenerator(sample_interval=0.1, movement_threshold=30.0)
    generator.process([make_track(x=100)], 0.0)
    events = generator.process([make_track(x=200)], 1.0)
    assert [e.action for e in events] == [ACTION_MOVED]


# ---------------------------------------------------------------------------
# M0.2: zones
# ---------------------------------------------------------------------------
BAY = ZoneSet([Zone.from_rect("bay", (200, 0, 400, 1000))])


def test_zone_entry_and_exit_events_name_the_zone():
    generator = EventGenerator(sample_interval=999, movement_threshold=999, zones=BAY)
    generator.process([make_track(x=0)], 0.0)

    entered = generator.process([make_track(x=250)], 1.0)
    assert [e.action for e in entered] == [ACTION_ENTERED_ZONE]
    assert entered[0].attributes["zone"] == "bay"
    assert entered[0].zones == ["bay"]

    exited = generator.process([make_track(x=500)], 2.0)
    assert [e.action for e in exited] == [ACTION_EXITED_ZONE]
    assert exited[0].attributes["zone"] == "bay"
    assert exited[0].zones == []


def test_no_zone_events_without_configured_zones():
    generator = EventGenerator(sample_interval=999, movement_threshold=999)
    generator.process([make_track(x=0)], 0.0)
    events = generator.process([make_track(x=250)], 1.0)
    assert all(e.action not in (ACTION_ENTERED_ZONE, ACTION_EXITED_ZONE) for e in events)


def test_zone_events_are_deterministic_when_several_change_at_once():
    zones = ZoneSet(
        [
            Zone.from_rect("zulu", (200, 0, 400, 1000)),
            Zone.from_rect("alpha", (210, 0, 400, 1000)),
        ]
    )
    generator = EventGenerator(sample_interval=999, movement_threshold=999, zones=zones)
    generator.process([make_track(x=0)], 0.0)
    events = generator.process([make_track(x=250)], 1.0)
    # Sorted by zone name, not by config order, so replays match.
    assert [e.attributes["zone"] for e in events] == ["alpha", "zulu"]


def test_disappearing_inside_a_zone_emits_exit_then_disappeared():
    generator = EventGenerator(zones=BAY)
    track = make_track(x=250)
    generator.process([track], 0.0)
    events = generator.flush(3.0, [track])
    assert [e.action for e in events] == [ACTION_EXITED_ZONE, ACTION_DISAPPEARED]


def test_events_record_their_coordinate_space():
    generator = EventGenerator(zones=BAY)
    event = generator.process([make_track(x=250)], 0.0)[0]
    assert event.coordinate_space == "image_pixels"
