"""Each risk factor, tested on its own.

Every factor is exercised in isolation with a hand-built context, so a failure
points at one formula rather than at "the risk engine". This is the payoff of
keeping factors independent of the aggregation.
"""

import pytest

from app.events.schema import Event
from app.memory import TemporalEventMemory
from app.reasoning.config import RiskConfig
from app.reasoning.factors import (
    ClosingSpeedFactor,
    EscalationFactor,
    PersistenceFactor,
    ProximityFactor,
    RiskCandidate,
    RiskContext,
    TrajectoryFactor,
    ZoneFactor,
)
from app.reasoning.factors.base import linear_falloff
from app.reasoning.kinematics import MotionEstimator
from app.reasoning.models.risk import FactorScore
from app.spatial import Zone, ZoneSet

PERSON = "person_1"
FORKLIFT = "forklift_2"
BAY = Zone.from_rect("forklift_bay", (400, 200, 800, 500))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def moving(entity_id, class_name, start, velocity_px_per_frame, count=10, fps=10.0, y=300.0):
    """A straight-line event history for one entity."""
    events = []
    for i in range(count):
        x = start + velocity_px_per_frame * i
        events.append(
            Event(
                timestamp=round(i / fps, 4),
                entity_id=entity_id,
                action="appeared" if i == 0 else "moved",
                attributes={"class_name": class_name, "confidence": 0.9},
                position=[x, y],
                bbox=[x - 20, y - 45, x + 20, y + 45],
                event_id=f"evt_{entity_id}_{i:02d}",
            )
        )
    return events


def make_context(person_events, vehicle_events=None, at=0.9, config=None, zones=None):
    config = config or RiskConfig(vehicle_classes=["forklift"], zones=zones)
    events = list(person_events) + list(vehicle_events or [])
    memory = TemporalEventMemory.from_events(events)

    estimator = MotionEstimator()
    motions = {}
    motions[PERSON] = estimator.estimate(
        [e for e in person_events if e.timestamp <= at], at, PERSON
    )
    if vehicle_events:
        motions[FORKLIFT] = estimator.estimate(
            [e for e in vehicle_events if e.timestamp <= at], at, FORKLIFT
        )
    motions = {k: v for k, v in motions.items() if v is not None}

    candidate = RiskCandidate(
        entity_ids=[PERSON] + ([FORKLIFT] if vehicle_events else []),
        primary=PERSON,
        secondary=FORKLIFT if vehicle_events else None,
    )
    return RiskContext(
        memory=memory, at=at, config=config, candidate=candidate, motions=motions
    )


# ===========================================================================
# Shared falloff helper
# ===========================================================================
def test_linear_falloff_endpoints_and_midpoint():
    assert linear_falloff(70, full_at=70, zero_at=260) == 1.0
    assert linear_falloff(260, full_at=70, zero_at=260) == 0.0
    assert linear_falloff(165, full_at=70, zero_at=260) == pytest.approx(0.5)


def test_linear_falloff_clamps_outside_the_range():
    assert linear_falloff(0, full_at=70, zero_at=260) == 1.0
    assert linear_falloff(9999, full_at=70, zero_at=260) == 0.0


def test_linear_falloff_works_in_the_increasing_direction():
    # Speed-like: full at the high end.
    assert linear_falloff(160, full_at=160, zero_at=10) == 1.0
    assert linear_falloff(10, full_at=160, zero_at=10) == 0.0
    assert linear_falloff(-50, full_at=160, zero_at=10) == 0.0


# ===========================================================================
# ProximityFactor
# ===========================================================================
def test_proximity_scores_zero_beyond_the_interaction_radius():
    context = make_context(
        moving(PERSON, "person", 100, 0), moving(FORKLIFT, "forklift", 900, 0)
    )
    score = ProximityFactor().evaluate(context)
    assert score.score == 0.0
    assert "beyond" in score.rationale


def test_proximity_scores_one_inside_the_critical_radius():
    context = make_context(
        moving(PERSON, "person", 400, 0), moving(FORKLIFT, "forklift", 430, 0)
    )
    score = ProximityFactor().evaluate(context)
    assert score.score == 1.0


def test_proximity_is_linear_between_the_radii():
    config = RiskConfig(vehicle_classes=["forklift"])
    midpoint = (config.critical_radius_px + config.interaction_radius_px) / 2
    context = make_context(
        moving(PERSON, "person", 400, 0),
        moving(FORKLIFT, "forklift", 400 + midpoint, 0),
        config=config,
    )
    assert ProximityFactor().evaluate(context).score == pytest.approx(0.5, abs=0.01)


def test_proximity_reports_its_measurements_in_pixels():
    context = make_context(
        moving(PERSON, "person", 400, 0), moving(FORKLIFT, "forklift", 500, 0)
    )
    score = ProximityFactor().evaluate(context)
    assert score.details["separation_px"] == pytest.approx(100.0)
    assert score.coordinate_space == "image_pixels"
    assert "interaction_radius_px" in score.details


def test_proximity_alone_cannot_exceed_the_low_band():
    """30 points is the whole factor: being near something is not an incident."""
    context = make_context(
        moving(PERSON, "person", 400, 0), moving(FORKLIFT, "forklift", 410, 0)
    )
    score = ProximityFactor().evaluate(context)
    assert score.contribution <= 30.0


def test_proximity_needs_a_pair():
    context = make_context(moving(PERSON, "person", 400, 0))
    score = ProximityFactor().evaluate(context)
    assert score.score == 0.0
    assert "needs two" in score.rationale


# ===========================================================================
# ClosingSpeedFactor
# ===========================================================================
def test_closing_speed_scores_zero_when_separating():
    context = make_context(
        moving(PERSON, "person", 400, -10), moving(FORKLIFT, "forklift", 500, 10)
    )
    score = ClosingSpeedFactor().evaluate(context)
    assert score.score == 0.0
    assert "separating" in score.rationale


def test_closing_speed_saturates_at_the_reference():
    context = make_context(
        moving(PERSON, "person", 100, 10), moving(FORKLIFT, "forklift", 900, -10)
    )
    score = ClosingSpeedFactor().evaluate(context)
    assert score.score == 1.0
    assert score.details["closing_speed_px_per_s"] == pytest.approx(200.0, abs=1.0)


def test_closing_speed_is_zero_for_stationary_entities():
    context = make_context(
        moving(PERSON, "person", 400, 0), moving(FORKLIFT, "forklift", 500, 0)
    )
    assert ClosingSpeedFactor().evaluate(context).score == 0.0


def test_closing_speed_refuses_without_history():
    context = make_context(
        moving(PERSON, "person", 100, 10, count=2),
        moving(FORKLIFT, "forklift", 400, -10, count=2),
        at=0.1,
    )
    score = ClosingSpeedFactor().evaluate(context)
    assert score.score == 0.0
    assert score.details["insufficient_history"] is True


def test_closing_speed_units_are_pixels_per_second():
    context = make_context(
        moving(PERSON, "person", 100, 10), moving(FORKLIFT, "forklift", 900, -10)
    )
    details = ClosingSpeedFactor().evaluate(context).details
    assert "closing_speed_px_per_s" in details
    assert "reference_px_per_s" in details


# ===========================================================================
# TrajectoryFactor
# ===========================================================================
def test_trajectory_scores_high_for_an_imminent_head_on_conflict():
    context = make_context(
        moving(PERSON, "person", 300, 10, y=300),
        moving(FORKLIFT, "forklift", 600, -10, y=300),
    )
    score = TrajectoryFactor().evaluate(context)
    assert score.score > 0.7
    assert score.details["closest_approach_distance_px"] == pytest.approx(0.0, abs=2.0)


def test_trajectory_scores_zero_for_a_wide_miss():
    context = make_context(
        moving(PERSON, "person", 100, 10, y=100),
        moving(FORKLIFT, "forklift", 900, -10, y=500),
    )
    score = TrajectoryFactor().evaluate(context)
    assert score.score == 0.0
    assert "miss" in score.rationale


def test_trajectory_refuses_when_both_entities_are_stationary():
    """Scenario F: detector jitter must never be read as a heading."""
    context = make_context(
        moving(PERSON, "person", 400, 0), moving(FORKLIFT, "forklift", 480, 0)
    )
    score = TrajectoryFactor().evaluate(context)
    assert score.score == 0.0
    assert score.details["both_stationary"] is True


def test_trajectory_refuses_without_enough_history():
    """Scenario G: no velocity, no prediction."""
    context = make_context(
        moving(PERSON, "person", 300, 10, count=2),
        moving(FORKLIFT, "forklift", 500, -10, count=2),
        at=0.1,
    )
    score = TrajectoryFactor().evaluate(context)
    assert score.score == 0.0
    assert score.details["insufficient_history"] is True
    assert PERSON in score.details["entities_without_trajectory"]


def test_trajectory_scores_zero_when_already_separating():
    context = make_context(
        moving(PERSON, "person", 400, -10), moving(FORKLIFT, "forklift", 500, 10)
    )
    score = TrajectoryFactor().evaluate(context)
    assert score.score == 0.0
    assert score.details["is_converging"] is False


def test_trajectory_spatial_and_temporal_terms_are_multiplied():
    context = make_context(
        moving(PERSON, "person", 300, 10, y=300),
        moving(FORKLIFT, "forklift", 600, -10, y=300),
    )
    score = TrajectoryFactor().evaluate(context)
    expected = score.details["spatial_term"] * score.details["temporal_term"]
    assert score.score == pytest.approx(expected)


def test_a_distant_conflict_is_discounted_by_the_horizon():
    """Slow convergence over a long horizon must not read as urgent."""
    near = make_context(
        moving(PERSON, "person", 300, 10, y=300),
        moving(FORKLIFT, "forklift", 600, -10, y=300),
    )
    far = make_context(
        moving(PERSON, "person", 300, 1, y=300),
        moving(FORKLIFT, "forklift", 900, -1, y=300),
    )
    assert TrajectoryFactor().evaluate(near).score > TrajectoryFactor().evaluate(far).score


def test_predicted_time_to_incident_only_for_a_real_conflict():
    conflict = make_context(
        moving(PERSON, "person", 300, 10, y=300),
        moving(FORKLIFT, "forklift", 600, -10, y=300),
    )
    near_miss = make_context(
        moving(PERSON, "person", 100, 10, y=100),
        moving(FORKLIFT, "forklift", 900, -10, y=350),
    )
    assert TrajectoryFactor.predicted_time_to_incident(conflict) is not None
    assert TrajectoryFactor.predicted_time_to_incident(near_miss) is None


def test_trajectory_reports_velocities_in_pixels_per_second():
    context = make_context(
        moving(PERSON, "person", 300, 10, y=300),
        moving(FORKLIFT, "forklift", 600, -10, y=300),
    )
    details = TrajectoryFactor().evaluate(context).details
    assert "primary_velocity_px_per_s" in details
    assert "secondary_velocity_px_per_s" in details
    assert "closest_approach_distance_px" in details


# ===========================================================================
# ZoneFactor
# ===========================================================================
def zoned_context(person_x, vehicle_x=None, at=0.9):
    zones = ZoneSet([BAY])
    config = RiskConfig(
        vehicle_classes=["forklift"], operating_zones=["forklift_bay"], zones=zones
    )
    person_events = moving(PERSON, "person", person_x, 0, y=400)
    vehicle_events = moving(FORKLIFT, "forklift", vehicle_x, 0, y=400) if vehicle_x else None

    # The memory carries zone occupancy, exactly as M0.2 recorded it.
    for event in person_events:
        event.zones = ["forklift_bay"] if BAY.contains_box(event.bbox) else []
    if vehicle_events:
        for event in vehicle_events:
            event.zones = ["forklift_bay"] if BAY.contains_box(event.bbox) else []
    return make_context(person_events, vehicle_events, at=at, config=config)


def test_zone_scores_zero_outside_every_operating_zone():
    score = ZoneFactor().evaluate(zoned_context(person_x=100))
    assert score.score == 0.0
    assert "not inside" in score.rationale


def test_zone_scores_high_for_a_person_inside_with_no_vehicle():
    score = ZoneFactor().evaluate(zoned_context(person_x=600))
    assert score.score == ZoneFactor.SCORE_UNACCOMPANIED
    assert score.details["vehicle_present"] is False


def test_zone_maxes_out_when_the_vehicle_shares_the_zone():
    score = ZoneFactor().evaluate(zoned_context(person_x=600, vehicle_x=700))
    assert score.score == ZoneFactor.SCORE_VEHICLE_PRESENT
    assert score.details["vehicle_in_zone"] is True


def test_zone_score_rises_as_the_vehicle_closes():
    far = ZoneFactor().evaluate(zoned_context(person_x=600, vehicle_x=100))
    assert ZoneFactor.SCORE_VEHICLE_DISTANT <= far.score < ZoneFactor.SCORE_VEHICLE_PRESENT


def test_zone_is_inactive_when_no_operating_zones_are_configured():
    context = make_context(
        moving(PERSON, "person", 600, 0), moving(FORKLIFT, "forklift", 700, 0)
    )
    score = ZoneFactor().evaluate(context)
    assert score.score == 0.0
    assert "no operating zones configured" in score.rationale


def test_zone_names_the_breached_zone():
    score = ZoneFactor().evaluate(zoned_context(person_x=600))
    assert score.details["breached_zones"] == ["forklift_bay"]


# ===========================================================================
# PersistenceFactor
# ===========================================================================
def persistence_context(entry_count, at=10.0):
    zones = ZoneSet([BAY])
    config = RiskConfig(
        vehicle_classes=["forklift"], operating_zones=["forklift_bay"], zones=zones
    )
    events = moving(PERSON, "person", 600, 0, count=4)
    for i in range(entry_count):
        events.append(
            Event(
                timestamp=1.0 + i,
                entity_id=PERSON,
                action="entered_zone",
                attributes={"class_name": "person", "confidence": 0.9, "zone": "forklift_bay"},
                position=[600.0, 400.0],
                bbox=[580.0, 355.0, 620.0, 445.0],
                zones=["forklift_bay"],
                event_id=f"evt_entry_{i}",
            )
        )
    return make_context(events, at=at, config=config)


def test_persistence_scores_zero_without_violations():
    score = PersistenceFactor().evaluate(persistence_context(0))
    assert score.score == 0.0
    assert score.details["violation_count"] == 0


def test_a_single_entry_is_not_yet_a_pattern():
    score = PersistenceFactor().evaluate(persistence_context(1))
    assert score.score == pytest.approx(1 / 3)


def test_persistence_saturates_at_the_threshold():
    score = PersistenceFactor().evaluate(persistence_context(3))
    assert score.score == 1.0
    assert score.details["violation_count"] == 3


def test_persistence_does_not_exceed_one():
    assert PersistenceFactor().evaluate(persistence_context(9)).score == 1.0


def test_persistence_ignores_violations_outside_the_window():
    score = PersistenceFactor().evaluate(persistence_context(3, at=1000.0))
    assert score.score == 0.0


def test_persistence_cites_the_entry_events_as_evidence():
    score = PersistenceFactor().evaluate(persistence_context(3))
    assert score.evidence_event_ids == ["evt_entry_0", "evt_entry_1", "evt_entry_2"]


# ===========================================================================
# EscalationFactor
# ===========================================================================
def escalation_scores(*values):
    return [
        FactorScore(name=f"f{i}", score=v, weight=10.0) for i, v in enumerate(values)
    ]


def test_escalation_needs_at_least_two_signals():
    context = make_context(moving(PERSON, "person", 400, 0))
    score = EscalationFactor().evaluate_with(context, escalation_scores(0.9))
    assert score.details["multiplier"] == 1.0
    assert score.details["signal_count"] == 1


def test_escalation_multiplier_grows_with_corroboration():
    context = make_context(moving(PERSON, "person", 400, 0))
    factor = EscalationFactor()
    multipliers = [
        factor.evaluate_with(context, escalation_scores(*([0.5] * n))).details["multiplier"]
        for n in range(6)
    ]
    assert multipliers == [1.0, 1.0, 1.10, 1.20, 1.30, 1.35]


def test_weak_signals_below_the_threshold_do_not_corroborate():
    context = make_context(moving(PERSON, "person", 400, 0))
    score = EscalationFactor().evaluate_with(context, escalation_scores(0.1, 0.1, 0.1))
    assert score.details["signal_count"] == 0
    assert score.details["multiplier"] == 1.0


def test_escalation_contributes_no_points_of_its_own():
    context = make_context(moving(PERSON, "person", 400, 0))
    score = EscalationFactor().evaluate_with(context, escalation_scores(0.9, 0.9, 0.9))
    assert score.weight == 0.0
    assert score.contribution == 0.0


def test_escalation_names_the_corroborating_factors():
    context = make_context(moving(PERSON, "person", 400, 0))
    scores = [
        FactorScore(name="proximity", score=0.5, weight=30.0),
        FactorScore(name="trajectory", score=0.5, weight=25.0),
    ]
    score = EscalationFactor().evaluate_with(context, scores)
    assert score.details["signal_names"] == ["proximity", "trajectory"]
