"""M0.2 temporal event memory — the twelve required behaviours.

Every test here is driven by scripted synthetic detections through the real
tracker, the real event generator and the real memory. No video, no model, no
randomness, so results are exactly reproducible.

Reminder: all coordinates and thresholds are IMAGE PIXELS, never real-world
units. See app/spatial.py.
"""

import pytest

from app.events.schema import (
    ACTION_APPEARED,
    ACTION_DISAPPEARED,
    ACTION_ENTERED_ZONE,
    ACTION_EXITED_ZONE,
    ACTION_MOVED,
    ACTION_STATIONARY,
)
from app.memory import EventFilter, TemporalEventMemory
from app.spatial import Zone, ZoneSet

from helpers import hold, make_detection, run_scenario, walk


# ===========================================================================
# 1. Entity appearance
# ===========================================================================
def test_1_entity_appearance():
    memory = run_scenario(walk(0, 20, 6))

    appeared = memory.stream(actions=[ACTION_APPEARED])
    assert len(appeared) == 1

    event = appeared[0]
    assert event.entity_id == "person_1"
    assert event.timestamp == 0.0
    assert event.attributes["class_name"] == "person"
    assert event.attributes["confidence"] == 0.9
    assert event.position == [20.0, 145.0]
    assert event.bbox == [0.0, 100.0, 40.0, 190.0]
    assert event.coordinate_space == "image_pixels"

    # Appearance opens the entity's presence and its state.
    state = memory.latest_state("person_1")
    assert state.first_seen == 0.0
    assert state.class_name == "person"
    assert memory.entity_history("person_1")[0].action == ACTION_APPEARED


def test_1b_appearance_is_emitted_exactly_once():
    memory = run_scenario(walk(0, 20, 20))
    assert len(memory.stream(entity_id="person_1", actions=[ACTION_APPEARED])) == 1


# ===========================================================================
# 2. Movement
# ===========================================================================
def test_2_movement():
    # 20px per frame, 40px threshold: a `moved` event every second frame.
    memory = run_scenario(walk(0, 20, 11), movement_threshold=40.0)

    moved = memory.stream(actions=[ACTION_MOVED])
    assert len(moved) == 5

    first = moved[0]
    assert first.attributes["displacement_px"] == 40.0
    assert first.attributes["from_position"] == [20.0, 145.0]
    assert first.position == [60.0, 145.0]

    # Positions advance monotonically in the direction of travel.
    xs = [e.position[0] for e in moved]
    assert xs == sorted(xs)


def test_2b_movement_below_threshold_emits_nothing():
    # 2px per frame never accumulates to the 40px threshold within the clip.
    memory = run_scenario(
        walk(0, 2, 10), movement_threshold=40.0, sample_interval=999, stationary_duration=999
    )
    assert memory.stream(actions=[ACTION_MOVED]) == []


# ===========================================================================
# 3. Stationary behaviour
# ===========================================================================
def test_3_stationary_behavior():
    # Perfectly still for 10 frames at 10fps = 0.9s; threshold is 0.5s.
    memory = run_scenario(hold(100, 10), stationary_duration=0.5, sample_interval=999)

    stationary = memory.stream(actions=[ACTION_STATIONARY])
    assert len(stationary) == 1, "stationary must fire once per episode, not per frame"

    event = stationary[0]
    assert event.timestamp == pytest.approx(0.5)
    assert event.attributes["stationary_for_seconds"] == pytest.approx(0.5)
    assert event.attributes["drift_px"] == 0.0


def test_3b_stationary_does_not_fire_before_the_dwell_time():
    memory = run_scenario(hold(100, 3), stationary_duration=5.0, sample_interval=999)
    assert memory.stream(actions=[ACTION_STATIONARY]) == []


def test_3c_stationary_rearms_after_the_entity_moves_again():
    # still, then a large jump, then still again -> two stationary episodes.
    frames = hold(100, 8) + walk(400, 0, 8)
    memory = run_scenario(
        frames,
        stationary_duration=0.5,
        stationary_threshold=15.0,
        sample_interval=999,
        tracker_kwargs={"min_hits": 1, "max_age": 5, "iou_threshold": 0.0},
    )
    stationary = memory.stream(actions=[ACTION_STATIONARY])
    assert len(stationary) == 2
    assert stationary[0].position[0] != stationary[1].position[0]


def test_3d_a_moving_entity_is_never_stationary():
    memory = run_scenario(walk(0, 30, 12), stationary_duration=0.2, stationary_threshold=5.0)
    assert memory.stream(actions=[ACTION_STATIONARY]) == []


# ===========================================================================
# 4. Zone entry
# ===========================================================================
ZONES = ZoneSet([Zone.from_rect("loading_bay", (200, 0, 400, 400))])


def test_4_zone_entry():
    # Anchor is bottom-centre: x = left + 20. Crosses x=200 at frame 10.
    memory = run_scenario(walk(0, 20, 16), zones=ZONES)

    entered = memory.stream(actions=[ACTION_ENTERED_ZONE])
    assert len(entered) == 1

    event = entered[0]
    assert event.attributes["zone"] == "loading_bay"
    assert event.attributes.get("on_appearance") is None
    assert "loading_bay" in event.zones
    assert event.position[0] >= 200

    # Occupancy is queryable from the memory, not just the event.
    assert memory.zone_occupancy("loading_bay", at=event.timestamp) == ["person_1"]


def test_4b_entity_that_appears_inside_a_zone_is_recorded_as_entering():
    memory = run_scenario(hold(250, 4), zones=ZONES)
    entered = memory.stream(actions=[ACTION_ENTERED_ZONE])
    assert len(entered) == 1
    assert entered[0].attributes["on_appearance"] is True


def test_4c_an_entity_outside_every_zone_produces_no_zone_events():
    memory = run_scenario(walk(0, 5, 10), zones=ZONES)
    assert memory.stream(actions=[ACTION_ENTERED_ZONE, ACTION_EXITED_ZONE]) == []


# ===========================================================================
# 5. Zone exit
# ===========================================================================
def test_5_zone_exit():
    # Starts inside the bay, walks out to the right past x=400.
    memory = run_scenario(walk(200, 20, 16), zones=ZONES)

    exited = memory.stream(actions=[ACTION_EXITED_ZONE])
    assert len(exited) == 1

    event = exited[0]
    assert event.attributes["zone"] == "loading_bay"
    assert "loading_bay" not in event.zones
    assert event.position[0] > 400

    entered = memory.stream(actions=[ACTION_ENTERED_ZONE])
    assert entered[0].timestamp < event.timestamp
    assert memory.zone_occupancy("loading_bay") == []


def test_5b_disappearing_inside_a_zone_still_closes_occupancy():
    """Otherwise zone occupancy would leak an entity that is long gone."""
    memory = run_scenario(hold(250, 6), zones=ZONES)

    exited = memory.stream(actions=[ACTION_EXITED_ZONE])
    assert len(exited) == 1
    assert exited[0].attributes["on_disappearance"] is True
    assert memory.zone_occupancy("loading_bay") == []


def test_5c_overlapping_zones_each_get_their_own_events():
    zones = ZoneSet(
        [
            Zone.from_rect("bay", (200, 0, 500, 400)),
            Zone.from_rect("aisle", (300, 0, 600, 400)),
        ]
    )
    memory = run_scenario(walk(0, 20, 20), zones=zones)
    entered = {e.attributes["zone"] for e in memory.stream(actions=[ACTION_ENTERED_ZONE])}
    assert entered == {"bay", "aisle"}


# ===========================================================================
# 6. Disappearance
# ===========================================================================
def test_6_disappearance():
    # Present for 6 frames, then gone for long enough to exceed max_age.
    frames = walk(0, 20, 6) + [[] for _ in range(10)]
    memory = run_scenario(frames, tracker_kwargs={"min_hits": 1, "max_age": 3})

    disappeared = memory.stream(actions=[ACTION_DISAPPEARED])
    assert len(disappeared) == 1

    event = disappeared[0]
    assert event.entity_id == "person_1"
    assert event.attributes["first_seen"] == 0.0
    assert event.attributes["visible_for_seconds"] == pytest.approx(0.5)

    state = memory.latest_state("person_1")
    assert state.present is False
    assert memory.present_entities() == []


def test_6b_disappearance_is_the_last_event_for_an_entity():
    frames = walk(0, 20, 6) + [[] for _ in range(10)]
    memory = run_scenario(frames, tracker_kwargs={"min_hits": 1, "max_age": 3})
    assert memory.entity_history("person_1")[-1].action == ACTION_DISAPPEARED


def test_6c_entities_still_visible_at_the_end_are_closed_out():
    memory = run_scenario(walk(0, 20, 6), close_at_end=True)
    assert len(memory.stream(actions=[ACTION_DISAPPEARED])) == 1


# ===========================================================================
# 7. Chronological ordering
# ===========================================================================
def test_7_chronological_ordering():
    frames = []
    for i in range(20):
        frame = [make_detection(i * 20, 100)]
        if i >= 5:
            frame.append(make_detection(600 - i * 15, 250, class_id=7, class_name="truck"))
        frames.append(frame)
    memory = run_scenario(frames, zones=ZONES)

    timestamps = [e.timestamp for e in memory]
    assert timestamps == sorted(timestamps)
    assert [e.timestamp for e in memory.stream()] == sorted(timestamps)


def test_7b_out_of_order_ingestion_still_reads_chronologically():
    """A replayed or merged log must not corrupt the timeline."""
    memory = run_scenario(walk(0, 20, 8))
    shuffled = list(reversed(memory.events))

    rebuilt = TemporalEventMemory.from_events(shuffled)
    assert [e.event_id for e in rebuilt] == [e.event_id for e in memory]
    assert [e.timestamp for e in rebuilt] == sorted(e.timestamp for e in rebuilt)


def test_7c_out_of_order_ingestion_still_folds_state_correctly():
    memory = run_scenario(walk(0, 20, 8))
    rebuilt = TemporalEventMemory.from_events(reversed(memory.events))
    assert rebuilt.latest_state("person_1").to_dict() == (
        memory.latest_state("person_1").to_dict()
    )


# ===========================================================================
# 8. Multiple entities
# ===========================================================================
def test_8_multiple_entities():
    frames = []
    for i in range(16):
        frames.append(
            [
                make_detection(i * 20, 100, class_id=0, class_name="person"),
                make_detection(600 - i * 20, 250, class_id=7, class_name="truck"),
                make_detection(300, 50, class_id=2, class_name="car"),
            ]
        )
    memory = run_scenario(frames)

    assert sorted(memory.entities()) == ["car_3", "person_1", "truck_2"]

    # Each entity's history is complete and independent.
    for entity_id in memory.entities():
        history = memory.entity_history(entity_id)
        assert history[0].action == ACTION_APPEARED
        assert history[-1].action == ACTION_DISAPPEARED
        assert {e.entity_id for e in history} == {entity_id}

    # The static car never moves; the other two do.
    movers = {e.entity_id for e in memory.stream(actions=[ACTION_MOVED])}
    assert movers == {"person_1", "truck_2"}


def test_8b_entities_are_filterable_by_class():
    frames = [
        [
            make_detection(i * 20, 100, class_id=0, class_name="person"),
            make_detection(600 - i * 20, 250, class_id=7, class_name="truck"),
        ]
        for i in range(12)
    ]
    memory = run_scenario(frames)
    trucks = memory.stream(class_names=["truck"])
    assert {e.entity_id for e in trucks} == {"truck_2"}


def test_8c_entities_currently_present_tracks_arrivals_and_departures():
    # person present throughout; truck only for frames 4..9.
    frames = []
    for i in range(20):
        frame = [make_detection(i * 5, 100)]
        if 4 <= i < 10:
            frame.append(make_detection(500, 250, class_id=7, class_name="truck"))
        frames.append(frame)
    memory = run_scenario(frames, tracker_kwargs={"min_hits": 1, "max_age": 2})

    assert memory.present_entities(at=0.0) == ["person_1"]
    assert sorted(memory.present_entities(at=0.5)) == ["person_1", "truck_2"]
    assert memory.present_entities(at=1.5) == ["person_1"]
    assert memory.present_entities() == []  # everything closed at end of stream


# ===========================================================================
# 9. Querying an entity's history
# ===========================================================================
def test_9_querying_entity_history():
    frames = [
        [
            make_detection(i * 20, 100, class_id=0, class_name="person"),
            make_detection(600 - i * 20, 250, class_id=7, class_name="truck"),
        ]
        for i in range(16)
    ]
    memory = run_scenario(frames, zones=ZONES)

    history = memory.entity_history("person_1")
    assert len(history) > 0
    assert all(e.entity_id == "person_1" for e in history)
    assert [e.timestamp for e in history] == sorted(e.timestamp for e in history)

    # The same query, narrowed by action.
    moves = memory.entity_history("person_1", actions=[ACTION_MOVED])
    assert all(e.action == ACTION_MOVED for e in moves)
    assert len(moves) < len(history)

    # An unknown entity is an empty history, not an error.
    assert memory.entity_history("ghost_99") == []
    assert memory.latest_state("ghost_99") is None
    assert memory.timeline("ghost_99") is None


def test_9b_timeline_exposes_history_and_state_together():
    memory = run_scenario(walk(0, 20, 16), zones=ZONES)
    timeline = memory.timeline("person_1")

    assert timeline.entity_id == "person_1"
    assert len(timeline) == len(memory.entity_history("person_1"))
    assert timeline.actions[0] == ACTION_APPEARED
    assert timeline.zones_visited == ["loading_bay"]
    assert timeline.state.class_name == "person"


def test_9c_latest_state_reflects_the_most_recent_observation():
    memory = run_scenario(walk(0, 20, 10), close_at_end=False)
    state = memory.latest_state("person_1")

    last_event = memory.entity_history("person_1")[-1]
    assert state.position == last_event.position
    assert state.last_seen == last_event.timestamp
    assert state.present is True


# ===========================================================================
# 10. Querying a time interval
# ===========================================================================
def test_10_querying_a_time_interval():
    memory = run_scenario(walk(0, 20, 21), movement_threshold=40.0)

    window = memory.between(0.5, 1.0)
    assert window, "expected events inside the window"
    assert all(0.5 <= e.timestamp < 1.0 for e in window)

    # start inclusive, end exclusive — adjacent windows tile without overlap.
    first = memory.between(0.0, 1.0)
    second = memory.between(1.0, 2.1)
    assert len(first) + len(second) == len(memory)
    assert not ({e.event_id for e in first} & {e.event_id for e in second})


def test_10b_time_interval_combines_with_other_criteria():
    frames = [
        [
            make_detection(i * 20, 100, class_id=0, class_name="person"),
            make_detection(600 - i * 20, 250, class_id=7, class_name="truck"),
        ]
        for i in range(21)
    ]
    memory = run_scenario(frames)

    window = memory.between(0.5, 1.5, entity_id="person_1", actions=[ACTION_MOVED])
    assert window
    for event in window:
        assert event.entity_id == "person_1"
        assert event.action == ACTION_MOVED
        assert 0.5 <= event.timestamp < 1.5


def test_10c_empty_interval_returns_nothing():
    memory = run_scenario(walk(0, 20, 10))
    assert memory.between(100.0, 200.0) == []


def test_10d_event_filter_is_reusable_across_queries():
    memory = run_scenario(walk(0, 20, 16), zones=ZONES)
    only_moves = EventFilter(actions=[ACTION_MOVED], start=0.2, end=1.0)
    assert memory.stream(filter=only_moves) == [
        e for e in memory if only_moves.matches(e)
    ]


# ===========================================================================
# 11. Temporary occlusion
# ===========================================================================
def test_11_temporary_occlusion():
    """An entity hidden briefly must not be reported as gone and reborn."""
    frames = []
    for i in range(20):
        hidden = 8 <= i < 12  # 4 frames behind an obstruction
        frames.append([] if hidden else [make_detection(i * 15, 100)])

    memory = run_scenario(frames, tracker_kwargs={"min_hits": 1, "max_age": 8})

    assert len(memory.entities()) == 1, "occlusion must not spawn a second entity"
    assert len(memory.stream(actions=[ACTION_APPEARED])) == 1
    assert len(memory.stream(actions=[ACTION_DISAPPEARED])) == 1

    # No disappearance during the gap: the entity was occluded, not gone.
    disappeared = memory.stream(actions=[ACTION_DISAPPEARED])[0]
    assert disappeared.timestamp > 1.1

    # And it is continuously "present" across the occlusion window.
    assert memory.present_entities(at=0.7) == ["person_1"]
    assert memory.present_entities(at=1.0) == ["person_1"]
    assert memory.present_entities(at=1.3) == ["person_1"]


def test_11b_occlusion_longer_than_max_age_does_report_a_disappearance():
    """The honest opposite case: beyond max_age the entity really is gone."""
    frames = walk(0, 15, 6) + [[] for _ in range(12)] + walk(300, 15, 6)
    memory = run_scenario(frames, tracker_kwargs={"min_hits": 1, "max_age": 3})

    assert len(memory.stream(actions=[ACTION_DISAPPEARED])) == 2
    assert len(memory.entities()) == 2


# ===========================================================================
# 12. Track ID persistence across a short occlusion
# ===========================================================================
def test_12_track_id_persists_after_short_occlusion():
    frames = []
    for i in range(20):
        hidden = 8 <= i < 12
        frames.append([] if hidden else [make_detection(i * 15, 100)])

    memory = run_scenario(frames, tracker_kwargs={"min_hits": 1, "max_age": 8})

    track_ids = {e.track_id for e in memory}
    assert track_ids == {1}, "the same object must keep one track ID"

    entity_ids = {e.entity_id for e in memory}
    assert entity_ids == {"person_1"}

    # Events exist on both sides of the gap, under the same identity.
    before = memory.between(0.0, 0.8, entity_id="person_1")
    after = memory.between(1.2, 99.0, entity_id="person_1")
    assert before and after
    assert {e.track_id for e in before} == {e.track_id for e in after} == {1}


def test_12b_identity_is_continuous_in_the_entity_timeline():
    frames = []
    for i in range(20):
        frames.append([] if 8 <= i < 12 else [make_detection(i * 15, 100)])

    memory = run_scenario(frames, tracker_kwargs={"min_hits": 1, "max_age": 8})
    timeline = memory.timeline("person_1")

    assert timeline.actions.count(ACTION_APPEARED) == 1
    assert timeline.actions.count(ACTION_DISAPPEARED) == 1
    assert timeline.state.track_id == 1
    # One continuous presence, not two fragments.
    assert timeline.state.duration > 1.5
