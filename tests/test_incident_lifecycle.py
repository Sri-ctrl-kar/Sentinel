"""Incident lifecycle states (M0.7 section H).

Lifecycle is derived from the deterministic prediction state, never from a
language model and never from the severity band. These tests pin both halves
of that claim: the derivation rules, and the independence from severity.
"""

from __future__ import annotations

import pytest

from app.intelligence.lifecycle import (
    ACTIVE_STATES,
    IMMINENT_HORIZON_SECONDS,
    INCIDENT_STATES,
    STATE_CURRENT,
    STATE_DEVELOPING,
    STATE_IMMINENT,
    STATE_OBSERVED,
    STATE_RESOLVED,
    LifecycleTracker,
    derive_incident_state,
    describe_state,
    is_active,
)
from app.reasoning import RiskEngine
from app.reasoning.models.risk import RiskAssessment, SEVERITIES, TimeToRisk
from app.reasoning.prediction import (
    CURRENTLY_UNSAFE_PROXIMITY,
    NO_PREDICTED_CONFLICT,
    PREDICTED_TRAJECTORY_CONFLICT,
    PREDICTION_UNAVAILABLE,
)
from app.scenarios import load_world


def assessment(
    risk_score: float = 50.0,
    outcome: str = NO_PREDICTED_CONFLICT,
    time_to_risk: TimeToRisk = None,
    severity: str = "medium",
) -> RiskAssessment:
    return RiskAssessment(
        risk_score=risk_score,
        severity=severity,
        incident_type="PERSON_VEHICLE_COLLISION_RISK",
        involved_entity_ids=["person_1", "forklift_2"],
        timestamp=1.2,
        prediction_outcome=outcome,
        time_to_risk=time_to_risk,
    )


# ---------------------------------------------------------------------------
# The five states
# ---------------------------------------------------------------------------
def test_states_are_exactly_the_five_specified():
    assert INCIDENT_STATES == (
        "observed",
        "developing",
        "imminent",
        "current",
        "resolved",
    )


def test_an_already_unsafe_pair_is_current_not_predicted():
    state = derive_incident_state(
        assessment(
            risk_score=90.0,
            outcome=CURRENTLY_UNSAFE_PROXIMITY,
            time_to_risk=TimeToRisk(status=TimeToRisk.STATUS_ALREADY_UNSAFE, seconds=0.0),
        )
    )
    assert state == STATE_CURRENT


def test_current_is_derived_from_the_outcome_when_no_time_to_risk_exists():
    state = derive_incident_state(assessment(outcome=CURRENTLY_UNSAFE_PROXIMITY))
    assert state == STATE_CURRENT


def test_a_crossing_inside_the_horizon_is_imminent():
    state = derive_incident_state(
        assessment(
            outcome=PREDICTED_TRAJECTORY_CONFLICT,
            time_to_risk=TimeToRisk(status=TimeToRisk.STATUS_PREDICTED, seconds=0.35),
        )
    )
    assert state == STATE_IMMINENT


def test_a_crossing_beyond_the_horizon_is_developing():
    state = derive_incident_state(
        assessment(
            outcome=PREDICTED_TRAJECTORY_CONFLICT,
            time_to_risk=TimeToRisk(
                status=TimeToRisk.STATUS_PREDICTED,
                seconds=IMMINENT_HORIZON_SECONDS + 1.0,
            ),
        )
    )
    assert state == STATE_DEVELOPING


def test_the_horizon_boundary_is_inclusive():
    state = derive_incident_state(
        assessment(
            outcome=PREDICTED_TRAJECTORY_CONFLICT,
            time_to_risk=TimeToRisk(
                status=TimeToRisk.STATUS_PREDICTED, seconds=IMMINENT_HORIZON_SECONDS
            ),
        )
    )
    assert state == STATE_IMMINENT


def test_a_predicted_conflict_without_a_crossing_time_is_developing():
    state = derive_incident_state(assessment(outcome=PREDICTED_TRAJECTORY_CONFLICT))
    assert state == STATE_DEVELOPING


def test_a_scored_but_quiet_situation_is_observed():
    state = derive_incident_state(assessment(risk_score=45.0))
    assert state == STATE_OBSERVED


def test_an_unpredictable_situation_is_observed_not_invented():
    """No prediction is not a prediction of safety, and not a resolution."""
    state = derive_incident_state(
        assessment(
            risk_score=30.0,
            outcome=PREDICTION_UNAVAILABLE,
            time_to_risk=TimeToRisk(
                status=TimeToRisk.STATUS_NOT_PREDICTED, reason="insufficient_history"
            ),
        )
    )
    assert state == STATE_OBSERVED


def test_resolution_requires_having_been_active():
    quiet = assessment(risk_score=5.0, severity="normal")
    assert derive_incident_state(quiet, previous=STATE_IMMINENT) == STATE_RESOLVED
    assert derive_incident_state(quiet, previous=None) == STATE_OBSERVED
    assert derive_incident_state(quiet, previous=STATE_OBSERVED) == STATE_OBSERVED


# ---------------------------------------------------------------------------
# Lifecycle is not severity
# ---------------------------------------------------------------------------
def test_lifecycle_is_independent_of_the_severity_band():
    """Same geometry, every severity label: the state never moves."""
    states = {
        derive_incident_state(
            assessment(
                risk_score=95.0,
                severity=severity,
                outcome=PREDICTED_TRAJECTORY_CONFLICT,
                time_to_risk=TimeToRisk(
                    status=TimeToRisk.STATUS_PREDICTED, seconds=0.5
                ),
            )
        )
        for severity in SEVERITIES
    }
    assert states == {STATE_IMMINENT}


def test_no_state_name_collides_with_a_severity_name():
    """Section H forbids duplicate concepts; overlapping names would be one."""
    assert set(INCIDENT_STATES).isdisjoint(set(SEVERITIES))


def test_active_states_exclude_observed_and_resolved():
    assert set(ACTIVE_STATES) == {STATE_DEVELOPING, STATE_IMMINENT, STATE_CURRENT}
    assert is_active(STATE_CURRENT)
    assert not is_active(STATE_OBSERVED)
    assert not is_active(STATE_RESOLVED)


def test_every_state_has_a_description():
    for state in INCIDENT_STATES:
        assert describe_state(state) != state


# ---------------------------------------------------------------------------
# Tracking across time
# ---------------------------------------------------------------------------
def test_the_tracker_reports_resolution_after_an_active_period():
    tracker = LifecycleTracker()
    imminent = assessment(
        outcome=PREDICTED_TRAJECTORY_CONFLICT,
        time_to_risk=TimeToRisk(status=TimeToRisk.STATUS_PREDICTED, seconds=0.4),
    )
    assert tracker.update(imminent) == STATE_IMMINENT
    assert tracker.update(assessment(risk_score=2.0)) == STATE_RESOLVED


def test_the_tracker_keys_situations_by_entity_pair_regardless_of_order():
    tracker = LifecycleTracker()
    first = assessment(
        outcome=PREDICTED_TRAJECTORY_CONFLICT,
        time_to_risk=TimeToRisk(status=TimeToRisk.STATUS_PREDICTED, seconds=0.4),
    )
    tracker.update(first)
    reversed_pair = assessment(risk_score=2.0)
    reversed_pair.involved_entity_ids = ["forklift_2", "person_1"]
    assert tracker.update(reversed_pair) == STATE_RESOLVED


def test_the_tracker_is_deterministic():
    sequence = [
        assessment(risk_score=10.0),
        assessment(
            outcome=PREDICTED_TRAJECTORY_CONFLICT,
            time_to_risk=TimeToRisk(status=TimeToRisk.STATUS_PREDICTED, seconds=3.0),
        ),
        assessment(
            outcome=CURRENTLY_UNSAFE_PROXIMITY,
            time_to_risk=TimeToRisk(
                status=TimeToRisk.STATUS_ALREADY_UNSAFE, seconds=0.0
            ),
        ),
        assessment(risk_score=1.0),
    ]
    first = LifecycleTracker().update_all(sequence)
    second = LifecycleTracker().update_all(sequence)
    assert first == second == [
        STATE_OBSERVED,
        STATE_DEVELOPING,
        STATE_CURRENT,
        STATE_RESOLVED,
    ]


# ---------------------------------------------------------------------------
# Against the real engine
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "scenario_name,expected", [("D", STATE_IMMINENT), ("A", STATE_OBSERVED)]
)
def test_real_scenarios_land_in_the_expected_state(scenario_name, expected):
    scenario = load_world(scenario_name)
    engine = RiskEngine(scenario.config, calibration=scenario.calibration)
    top = engine.assess(scenario.memory, at=scenario.at).top
    assert derive_incident_state(top) == expected


def test_a_real_scenario_progresses_from_observed_to_imminent_to_current():
    """The states follow the geometry over time, in order.

    Scenario D never reports ``developing``: by the time its predictor has
    enough history, the pair is already closing fast enough that the crossing
    is inside the two-second imminent horizon. That is the engine's geometry
    showing through, not a gap in the state machine — the synthetic tests
    above pin ``developing`` directly.
    """
    scenario = load_world("D")
    engine = RiskEngine(scenario.config, calibration=scenario.calibration)
    tracker = LifecycleTracker()
    seen = []
    for report in engine.assess_timeline(scenario.memory, step=0.1):
        pairs = [a for a in report.assessments if len(a.involved_entity_ids) == 2]
        if pairs:
            seen.append(tracker.update(max(pairs, key=lambda a: a.risk_score)))

    assert seen[0] == STATE_OBSERVED
    assert STATE_IMMINENT in seen
    assert STATE_CURRENT in seen
    assert seen.index(STATE_IMMINENT) < seen.index(STATE_CURRENT)


def test_the_lifecycle_never_calls_a_prediction_current():
    """Across every scenario, ``current`` appears only when the engine says so."""
    from app.scenarios import ALL_WORLD_SCENARIOS

    for name in ALL_WORLD_SCENARIOS:
        scenario = load_world(name)
        engine = RiskEngine(scenario.config, calibration=scenario.calibration)
        for report in engine.assess_timeline(scenario.memory, step=0.1):
            for a in report.assessments:
                if derive_incident_state(a) == STATE_CURRENT:
                    unsafe_now = (
                        a.time_to_risk is not None
                        and a.time_to_risk.status == TimeToRisk.STATUS_ALREADY_UNSAFE
                    ) or a.prediction_outcome == CURRENTLY_UNSAFE_PROXIMITY
                    assert unsafe_now, (
                        f"{name}: state 'current' without a present-tense unsafe "
                        f"signal (outcome={a.prediction_outcome})"
                    )
