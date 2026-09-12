"""Builders for incident-intelligence tests.

Evidence is built two ways here on purpose:

* :func:`scenario_evidence` runs the real deterministic pipeline, so the tests
  assert against values Sentinel actually produces;
* :func:`make_evidence` assembles evidence by hand, so a test can pin one
  awkward shape (no prediction, insufficient history, image pixels) without
  hunting for a scenario that happens to produce it.
"""

from __future__ import annotations

from typing import List, Optional

from app.intelligence import (
    EntityEvidence,
    FactorEvidence,
    IncidentEvidence,
    PredictionEvidence,
    TimeToRiskEvidence,
    evidence_from_assessment,
)
from app.intelligence.schema import IncidentExplanation
from app.reasoning import RiskEngine
from app.scenarios import load_world
from app.spatial import GROUND_PLANE_METERS, IMAGE_PIXELS


def scenario_evidence(
    name: str = "D", calibrated: bool = True, at: Optional[float] = None
) -> IncidentEvidence:
    """Top assessment of a world scenario, as incident evidence."""
    scenario = load_world(name)
    calibration = scenario.calibration if calibrated else None
    engine = RiskEngine(scenario.config, calibration=calibration)
    report = engine.assess(scenario.memory, at=scenario.at if at is None else at)
    assert report.assessments, f"scenario {name} produced no assessment"
    return evidence_from_assessment(
        report.top, events=scenario.memory.events
    )


def make_evidence(
    coordinate_space: str = GROUND_PLANE_METERS,
    prediction: Optional[PredictionEvidence] = None,
    time_to_risk: Optional[TimeToRiskEvidence] = None,
    entities: Optional[List[EntityEvidence]] = None,
    risk_score: float = 72.0,
    severity: str = "high",
    incident_state: str = "developing",
    zones: Optional[List[str]] = None,
    **overrides,
) -> IncidentEvidence:
    metric = coordinate_space == GROUND_PLANE_METERS
    zones = ["forklift_bay"] if zones is None else zones
    if entities is None:
        entities = [
            EntityEvidence(
                entity_id="person_1",
                class_name="person",
                position=[11.0, 5.0],
                velocity=[2.5, 0.0],
                speed=2.5,
                zones=list(zones),
                coordinate_space=coordinate_space,
            ),
            EntityEvidence(
                entity_id="forklift_2",
                class_name="forklift",
                position=[14.9, 5.0],
                velocity=[-3.0, 0.0],
                speed=3.0,
                zones=list(zones),
                coordinate_space=coordinate_space,
            ),
        ]
    if prediction is None:
        prediction = PredictionEvidence(
            outcome="PREDICTED_TRAJECTORY_CONFLICT",
            horizon_seconds=6.0,
            minimum_separation=0.0,
            seconds_to_minimum_separation=0.71,
            unsafe_separation_threshold=2.0,
            is_forward_looking=True,
        )
    if time_to_risk is None:
        time_to_risk = TimeToRiskEvidence(
            status="predicted", seconds=0.35, threshold=2.0
        )

    defaults = dict(
        incident_id="INC-1.20-person_1+forklift_2",
        timestamp=1.2,
        coordinate_space=coordinate_space,
        distance_unit="m" if metric else "px",
        speed_unit="m/s" if metric else "px/s",
        entities=entities,
        risk_score=risk_score,
        severity=severity,
        incident_type="PERSON_VEHICLE_COLLISION_RISK",
        incident_state=incident_state,
        confidence=1.0,
        factors=[
            FactorEvidence("proximity", 0.52, 30.0, 15.6, "3.90 apart", 1.0),
            FactorEvidence("zone", 1.0, 20.0, 20.0, "in the operating zone", 1.0),
        ],
        prediction=prediction,
        time_to_risk=time_to_risk,
        current_separation=3.9,
        closing_speed=5.5,
        triggered_event_ids=["evt-1"],
        triggered_event_actions=["entered_zone"],
        recommended_intervention="slow/stop the vehicle",
        calibration_active=metric,
    )
    defaults.update(overrides)
    return IncidentEvidence(**defaults)


def make_explanation(evidence: IncidentEvidence, **overrides) -> IncidentExplanation:
    """A minimal, fully grounded explanation that a test can then corrupt."""
    payload = dict(
        summary="Sentinel raised an incident involving person_1 and forklift_2.",
        severity_explanation="The risk score is an ordinal engineering signal.",
        evidence_points=["current separation: 3.90 m"],
        predicted_outcome="The pair is predicted to close further.",
        recommended_action="Recommend the operator slow the vehicle.",
        urgency="elevated",
        uncertainty="Prediction assumes constant velocity.",
        coordinate_space=evidence.coordinate_space,
    )
    payload.update(overrides)
    return IncidentExplanation.from_model_output(
        payload, incident_id=evidence.incident_id, provider="test", model="test"
    )
