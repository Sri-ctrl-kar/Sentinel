"""The incident evidence contract (M0.7 section A).

Evidence is the seam between the deterministic system and the AI layer, so
these tests care about two things above all: that nothing in it is invented,
and that every number carries the unit it was measured in.
"""

from __future__ import annotations

import json

import pytest

from app.intelligence import IncidentEvidence, evidence_from_assessment
from app.intelligence.evidence import EVIDENCE_SCHEMA_VERSION, Quantity
from app.reasoning import RiskEngine
from app.scenarios import ALL_WORLD_SCENARIOS, load_world
from app.spatial import GROUND_PLANE_METERS, IMAGE_PIXELS
from incident_helpers import make_evidence, scenario_evidence


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------
def test_evidence_round_trips_through_a_dict():
    evidence = scenario_evidence("D")
    restored = IncidentEvidence.from_dict(evidence.to_dict())
    assert restored.to_dict() == evidence.to_dict()


def test_evidence_round_trips_through_json():
    evidence = scenario_evidence("D")
    restored = IncidentEvidence.from_json(evidence.to_json())
    assert restored.incident_id == evidence.incident_id
    assert restored.risk_score == evidence.risk_score
    assert restored.entity_ids == evidence.entity_ids
    assert restored.prediction == evidence.prediction
    assert restored.time_to_risk == evidence.time_to_risk


def test_evidence_json_is_plain_json():
    payload = json.loads(scenario_evidence("D").to_json())
    assert payload["schema_version"] == EVIDENCE_SCHEMA_VERSION
    assert isinstance(payload["entities"], list)


def test_every_scenario_produces_serialisable_evidence():
    """Serialisation is canonical: a second round trip changes nothing.

    ``to_dict`` rounds for readability, so the first serialisation can shave
    the last digits off a factor contribution. What must not happen is drift —
    serialising the restored evidence has to produce byte-identical output.
    """
    for name in ALL_WORLD_SCENARIOS:
        scenario = load_world(name)
        engine = RiskEngine(scenario.config, calibration=scenario.calibration)
        for assessment in engine.assess(scenario.memory, at=scenario.at):
            evidence = evidence_from_assessment(assessment)
            restored = IncidentEvidence.from_json(evidence.to_json())
            assert restored.to_json() == evidence.to_json()
            assert IncidentEvidence.from_json(restored.to_json()) == restored


def test_evidence_is_immutable():
    """An explanation can never edit the evidence it was derived from."""
    evidence = make_evidence()
    with pytest.raises(Exception):
        evidence.risk_score = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# It copies, it does not compute
# ---------------------------------------------------------------------------
def test_evidence_mirrors_the_deterministic_assessment_exactly():
    scenario = load_world("D")
    engine = RiskEngine(scenario.config, calibration=scenario.calibration)
    assessment = engine.assess(scenario.memory, at=scenario.at).top

    evidence = evidence_from_assessment(assessment)

    assert evidence.risk_score == assessment.risk_score
    assert evidence.severity == assessment.severity
    assert evidence.incident_type == assessment.incident_type
    assert evidence.entity_ids == list(assessment.involved_entity_ids)
    assert evidence.timestamp == assessment.timestamp
    assert evidence.coordinate_space == assessment.coordinate_space
    assert evidence.confidence == assessment.confidence
    assert evidence.recommended_intervention == assessment.recommended_intervention
    assert evidence.prediction.outcome == assessment.prediction_outcome
    assert evidence.time_to_risk.status == assessment.time_to_risk.status
    assert evidence.time_to_risk.seconds == assessment.time_to_risk.seconds


def test_factor_contributions_are_carried_over_unchanged():
    scenario = load_world("D")
    engine = RiskEngine(scenario.config, calibration=scenario.calibration)
    assessment = engine.assess(scenario.memory, at=scenario.at).top
    evidence = evidence_from_assessment(assessment)

    by_name = {f.name: f for f in evidence.factors}
    for factor in assessment.contributing_factors:
        if factor.name == "escalation":
            continue
        assert by_name[factor.name].score == factor.score
        assert by_name[factor.name].weight == factor.weight
        assert by_name[factor.name].contribution == factor.contribution


def test_zone_membership_comes_from_the_zone_factor_not_from_prose():
    evidence = scenario_evidence("D")
    assert evidence.zone_membership == {
        "person_1": ["forklift_bay"],
        "forklift_2": ["forklift_bay"],
    }


def test_triggered_event_actions_are_empty_without_the_event_log():
    """Absent input produces absence, never a plausible-looking guess."""
    scenario = load_world("D")
    engine = RiskEngine(scenario.config, calibration=scenario.calibration)
    assessment = engine.assess(scenario.memory, at=scenario.at).top
    assert evidence_from_assessment(assessment).triggered_event_actions == []
    with_events = evidence_from_assessment(assessment, events=scenario.memory.events)
    assert "entered_zone" in with_events.triggered_event_actions


# ---------------------------------------------------------------------------
# Units travel with the numbers
# ---------------------------------------------------------------------------
def test_calibrated_evidence_is_labelled_in_metres():
    evidence = scenario_evidence("D", calibrated=True)
    assert evidence.coordinate_space == GROUND_PLANE_METERS
    assert evidence.distance_unit == "m"
    assert evidence.speed_unit == "m/s"
    assert evidence.is_metric
    assert all(q.is_metric for q in evidence.quantities())


def test_uncalibrated_evidence_is_labelled_in_pixels():
    evidence = scenario_evidence("D", calibrated=False)
    assert evidence.coordinate_space == IMAGE_PIXELS
    assert evidence.distance_unit == "px"
    assert evidence.speed_unit == "px/s"
    assert not evidence.is_metric
    assert not any(q.is_metric for q in evidence.quantities())


def test_the_same_scene_has_different_numbers_in_the_two_spaces():
    """Pixels and metres are different measurements, never relabellings."""
    metric = scenario_evidence("D", calibrated=True)
    pixels = scenario_evidence("D", calibrated=False)
    assert metric.current_separation != pixels.current_separation


def test_every_quantity_states_its_unit():
    for evidence in (scenario_evidence("D"), scenario_evidence("D", calibrated=False)):
        for quantity in evidence.quantities():
            assert quantity.unit in ("m", "m/s", "px", "px/s")
            assert quantity.coordinate_space == evidence.coordinate_space
            assert quantity.unit in quantity.describe()


def test_quantity_round_trips():
    quantity = Quantity("current separation", 3.9, "m", GROUND_PLANE_METERS)
    assert Quantity.from_dict(quantity.to_dict()) == quantity


# ---------------------------------------------------------------------------
# numeric_facts is the grounding allow-list
# ---------------------------------------------------------------------------
def test_numeric_facts_contain_every_reported_measurement():
    evidence = scenario_evidence("D")
    facts = evidence.numeric_facts()
    assert evidence.risk_score in facts
    assert evidence.current_separation in facts
    assert evidence.closing_speed in facts
    assert evidence.prediction.minimum_separation in facts
    assert evidence.time_to_risk.seconds in facts
    for entity in evidence.entities:
        assert entity.speed in facts


def test_numeric_facts_exclude_numbers_the_pipeline_never_produced():
    facts = scenario_evidence("D").numeric_facts()
    assert 9999.0 not in facts
    assert 42.4242 not in facts


# ---------------------------------------------------------------------------
# It is useful without an LLM
# ---------------------------------------------------------------------------
def test_evidence_is_readable_without_any_reasoner():
    evidence = scenario_evidence("D")
    text = evidence.to_json()
    assert "PERSON_VEHICLE_COLLISION_RISK" in text
    assert "ground_plane_meters" in text
    assert "NOT a calibrated probability" in evidence.to_dict()["score_interpretation"]


def test_insufficient_history_is_reported_rather_than_filled_in():
    evidence = make_evidence(
        prediction=type(make_evidence().prediction)(
            outcome="PREDICTION_UNAVAILABLE",
            unavailable_reason="insufficient_history",
        ),
        time_to_risk=None,
    )
    assert evidence.insufficient_history
    assert not evidence.has_usable_prediction
    assert evidence.prediction.minimum_separation is None
