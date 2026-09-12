"""The incident evidence contract.

This is the boundary between deterministic reasoning and AI reasoning. Above
it, Sentinel's risk engine has already decided everything that matters: whether
an incident exists, how severe it is, what will happen and when. Below it, a
language model may only *interpret* what is written here.

Three properties make that boundary safe:

1. **Nothing is added.** Every field is copied from a
   :class:`~app.reasoning.models.risk.RiskAssessment` the deterministic
   pipeline already produced. No measurement originates here.
2. **Every number carries its units.** Distances and speeds are tagged with the
   coordinate space they were measured in, and pixel values are never relabelled
   as metres.
3. **It is useful without an LLM.** The evidence serialises, round-trips and
   reads perfectly well on its own; the AI layer is optional on top of it.

:meth:`IncidentEvidence.numeric_facts` is what makes grounding checkable: it
enumerates every number an explanation is permitted to state. A figure that is
not in that set did not come from Sentinel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..reasoning.models.risk import RiskAssessment
from ..spatial import GROUND_PLANE_METERS, IMAGE_PIXELS

EVIDENCE_SCHEMA_VERSION = "0.7"


@dataclass(frozen=True)
class Quantity:
    """A number that knows what it is and what it is measured in.

    Exists so that a separation of ``8.0`` can never be read as metres when it
    was pixels. The unit travels with the value into the prompt, into the
    explanation, and into the grounding check.
    """

    name: str
    value: float
    unit: str
    coordinate_space: str

    @property
    def is_metric(self) -> bool:
        return self.coordinate_space == GROUND_PLANE_METERS

    def describe(self) -> str:
        return f"{self.name}: {self.value:.2f} {self.unit}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "unit": self.unit,
            "coordinate_space": self.coordinate_space,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Quantity":
        return cls(
            name=payload["name"],
            value=float(payload["value"]),
            unit=payload["unit"],
            coordinate_space=payload["coordinate_space"],
        )


@dataclass(frozen=True)
class EntityEvidence:
    """What is known about one entity involved in the incident."""

    entity_id: str
    class_name: Optional[str]
    position: Optional[List[float]]
    velocity: Optional[List[float]]
    speed: Optional[float]
    zones: List[str] = field(default_factory=list)
    coordinate_space: str = IMAGE_PIXELS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "class_name": self.class_name,
            "position": self.position,
            "velocity": self.velocity,
            "speed": self.speed,
            "zones": list(self.zones),
            "coordinate_space": self.coordinate_space,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EntityEvidence":
        return cls(
            entity_id=payload["entity_id"],
            class_name=payload.get("class_name"),
            position=payload.get("position"),
            velocity=payload.get("velocity"),
            speed=payload.get("speed"),
            zones=list(payload.get("zones", [])),
            coordinate_space=payload.get("coordinate_space", IMAGE_PIXELS),
        )


@dataclass(frozen=True)
class FactorEvidence:
    """One deterministic risk factor's contribution, as already computed."""

    name: str
    score: float
    weight: float
    contribution: float
    rationale: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "weight": self.weight,
            "contribution": round(self.contribution, 4),
            "rationale": self.rationale,
            "confidence": round(self.confidence, 4),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FactorEvidence":
        return cls(
            name=payload["name"],
            score=float(payload["score"]),
            weight=float(payload["weight"]),
            contribution=float(payload["contribution"]),
            rationale=payload.get("rationale", ""),
            confidence=float(payload.get("confidence", 0.0)),
        )


@dataclass(frozen=True)
class PredictionEvidence:
    """What the constant-velocity predictor said — no more, no less."""

    outcome: str
    horizon_seconds: Optional[float] = None
    minimum_separation: Optional[float] = None
    seconds_to_minimum_separation: Optional[float] = None
    unsafe_separation_threshold: Optional[float] = None
    unavailable_reason: Optional[str] = None
    is_forward_looking: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome,
            "horizon_seconds": self.horizon_seconds,
            "minimum_separation": self.minimum_separation,
            "seconds_to_minimum_separation": self.seconds_to_minimum_separation,
            "unsafe_separation_threshold": self.unsafe_separation_threshold,
            "unavailable_reason": self.unavailable_reason,
            "is_forward_looking": self.is_forward_looking,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PredictionEvidence":
        return cls(
            outcome=payload["outcome"],
            horizon_seconds=payload.get("horizon_seconds"),
            minimum_separation=payload.get("minimum_separation"),
            seconds_to_minimum_separation=payload.get(
                "seconds_to_minimum_separation"
            ),
            unsafe_separation_threshold=payload.get("unsafe_separation_threshold"),
            unavailable_reason=payload.get("unavailable_reason"),
            is_forward_looking=bool(payload.get("is_forward_looking", False)),
        )


@dataclass(frozen=True)
class TimeToRiskEvidence:
    """Structured time-to-risk, carried over verbatim."""

    status: str
    seconds: Optional[float] = None
    threshold: Optional[float] = None
    reason: Optional[str] = None

    @property
    def is_already_unsafe(self) -> bool:
        return self.status == "already_unsafe"

    @property
    def is_predicted(self) -> bool:
        return self.status == "predicted"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "seconds": self.seconds,
            "threshold": self.threshold,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TimeToRiskEvidence":
        return cls(
            status=payload["status"],
            seconds=payload.get("seconds"),
            threshold=payload.get("threshold"),
            reason=payload.get("reason"),
        )


@dataclass(frozen=True)
class IncidentEvidence:
    """Everything the AI layer is allowed to know.

    Constructed by :func:`evidence_from_assessment` from a completed
    deterministic assessment. It is immutable: an explanation cannot alter the
    evidence it was derived from.
    """

    incident_id: str
    timestamp: float
    coordinate_space: str
    distance_unit: str
    speed_unit: str
    entities: List[EntityEvidence]
    risk_score: float
    severity: str
    incident_type: str
    incident_state: str
    confidence: float
    factors: List[FactorEvidence] = field(default_factory=list)
    prediction: Optional[PredictionEvidence] = None
    time_to_risk: Optional[TimeToRiskEvidence] = None
    current_separation: Optional[float] = None
    closing_speed: Optional[float] = None
    triggered_event_ids: List[str] = field(default_factory=list)
    triggered_event_actions: List[str] = field(default_factory=list)
    recommended_intervention: str = ""
    calibration_active: bool = False
    space_fallback_reason: Optional[str] = None
    schema_version: str = EVIDENCE_SCHEMA_VERSION

    # ------------------------------------------------------------------
    @property
    def entity_ids(self) -> List[str]:
        return [e.entity_id for e in self.entities]

    @property
    def entity_classes(self) -> Dict[str, Optional[str]]:
        return {e.entity_id: e.class_name for e in self.entities}

    @property
    def zone_membership(self) -> Dict[str, List[str]]:
        return {e.entity_id: list(e.zones) for e in self.entities}

    @property
    def is_metric(self) -> bool:
        return self.coordinate_space == GROUND_PLANE_METERS

    @property
    def has_position_information(self) -> bool:
        return any(e.position is not None for e in self.entities)

    @property
    def has_usable_prediction(self) -> bool:
        return self.prediction is not None and self.prediction.is_forward_looking

    @property
    def insufficient_history(self) -> bool:
        """Did the deterministic layer decline to predict for lack of data?"""
        if self.prediction is None:
            return False
        return self.prediction.unavailable_reason == "insufficient_history"

    # ------------------------------------------------------------------
    def quantities(self) -> List[Quantity]:
        """Every spatial measurement, each carrying its unit."""
        items: List[Quantity] = []
        if self.current_separation is not None:
            items.append(
                Quantity(
                    "current separation",
                    self.current_separation,
                    self.distance_unit,
                    self.coordinate_space,
                )
            )
        if self.closing_speed is not None:
            items.append(
                Quantity(
                    "closing speed",
                    self.closing_speed,
                    self.speed_unit,
                    self.coordinate_space,
                )
            )
        if self.prediction and self.prediction.minimum_separation is not None:
            items.append(
                Quantity(
                    "predicted minimum separation",
                    self.prediction.minimum_separation,
                    self.distance_unit,
                    self.coordinate_space,
                )
            )
        return items

    def numeric_facts(self) -> List[float]:
        """Every number an explanation may legitimately state.

        The grounding check compares figures found in generated text against
        this list. A number that is not here was invented.
        """
        values: List[float] = [self.risk_score, self.timestamp]
        for quantity in self.quantities():
            values.append(quantity.value)
        if self.prediction:
            for value in (
                self.prediction.horizon_seconds,
                self.prediction.seconds_to_minimum_separation,
                self.prediction.unsafe_separation_threshold,
            ):
                if value is not None:
                    values.append(value)
        if self.time_to_risk:
            for value in (self.time_to_risk.seconds, self.time_to_risk.threshold):
                if value is not None:
                    values.append(value)
        for entity in self.entities:
            for coordinate in (entity.position or []) + (entity.velocity or []):
                values.append(float(coordinate))
            if entity.speed is not None:
                values.append(entity.speed)
        for factor in self.factors:
            values.extend((factor.score, factor.weight, factor.contribution))
        values.append(self.confidence)
        return values

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "incident_id": self.incident_id,
            "timestamp": round(self.timestamp, 4),
            "coordinate_space": self.coordinate_space,
            "distance_unit": self.distance_unit,
            "speed_unit": self.speed_unit,
            "calibration_active": self.calibration_active,
            "space_fallback_reason": self.space_fallback_reason,
            "entities": [e.to_dict() for e in self.entities],
            "risk_score": round(self.risk_score, 2),
            "severity": self.severity,
            "incident_type": self.incident_type,
            "incident_state": self.incident_state,
            "confidence": round(self.confidence, 4),
            "current_separation": self.current_separation,
            "closing_speed": self.closing_speed,
            "factors": [f.to_dict() for f in self.factors],
            "prediction": self.prediction.to_dict() if self.prediction else None,
            "time_to_risk": (
                self.time_to_risk.to_dict() if self.time_to_risk else None
            ),
            "triggered_event_ids": list(self.triggered_event_ids),
            "triggered_event_actions": list(self.triggered_event_actions),
            "recommended_intervention": self.recommended_intervention,
            "score_interpretation": (
                "ordinal 0-100 engineering signal, NOT a calibrated probability"
            ),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "IncidentEvidence":
        return cls(
            incident_id=payload["incident_id"],
            timestamp=float(payload["timestamp"]),
            coordinate_space=payload["coordinate_space"],
            distance_unit=payload["distance_unit"],
            speed_unit=payload["speed_unit"],
            entities=[EntityEvidence.from_dict(e) for e in payload["entities"]],
            risk_score=float(payload["risk_score"]),
            severity=payload["severity"],
            incident_type=payload["incident_type"],
            incident_state=payload["incident_state"],
            confidence=float(payload.get("confidence", 0.0)),
            factors=[FactorEvidence.from_dict(f) for f in payload.get("factors", [])],
            prediction=(
                PredictionEvidence.from_dict(payload["prediction"])
                if payload.get("prediction")
                else None
            ),
            time_to_risk=(
                TimeToRiskEvidence.from_dict(payload["time_to_risk"])
                if payload.get("time_to_risk")
                else None
            ),
            current_separation=payload.get("current_separation"),
            closing_speed=payload.get("closing_speed"),
            triggered_event_ids=list(payload.get("triggered_event_ids", [])),
            triggered_event_actions=list(payload.get("triggered_event_actions", [])),
            recommended_intervention=payload.get("recommended_intervention", ""),
            calibration_active=bool(payload.get("calibration_active", False)),
            space_fallback_reason=payload.get("space_fallback_reason"),
            schema_version=payload.get("schema_version", EVIDENCE_SCHEMA_VERSION),
        )

    @classmethod
    def from_json(cls, text: str) -> "IncidentEvidence":
        return cls.from_dict(json.loads(text))


# ---------------------------------------------------------------------------
def evidence_from_assessment(
    assessment: RiskAssessment,
    incident_id: Optional[str] = None,
    calibration_active: Optional[bool] = None,
    incident_state: Optional[str] = None,
    events: Sequence[Any] = (),
) -> IncidentEvidence:
    """Build evidence from a completed deterministic assessment.

    Copies only. If a value is absent from the assessment it is absent here —
    this function never fills a gap with a default that could be mistaken for a
    measurement.
    """
    from .lifecycle import derive_incident_state

    space = assessment.coordinate_space
    metric = space == GROUND_PLANE_METERS
    distance_unit = "m" if metric else "px"
    speed_unit = "m/s" if metric else "px/s"

    details = assessment.details or {}
    prediction_payload = details.get("prediction") or {}
    motions = details.get("motions") or {}

    entities = []
    for entity_id in assessment.involved_entity_ids:
        motion = motions.get(entity_id) or {}
        world = motion.get("world") or {}
        if metric and world:
            position = world.get("position_m")
            velocity = world.get("velocity_m_per_s")
            speed = world.get("speed_m_per_s")
        else:
            position = motion.get("position_px")
            velocity = motion.get("velocity_px_per_s")
            speed = motion.get("speed_px_per_s")
        entities.append(
            EntityEvidence(
                entity_id=entity_id,
                class_name=motion.get("class_name"),
                position=list(position) if position else None,
                velocity=list(velocity) if velocity else None,
                speed=speed,
                zones=_zones_for(assessment, entity_id),
                coordinate_space=space,
            )
        )

    prediction = None
    if prediction_payload:
        prediction = PredictionEvidence(
            outcome=prediction_payload.get("outcome", "PREDICTION_UNAVAILABLE"),
            horizon_seconds=prediction_payload.get("horizon_seconds"),
            minimum_separation=prediction_payload.get(
                f"minimum_separation_{distance_unit}"
            ),
            seconds_to_minimum_separation=prediction_payload.get(
                "seconds_to_minimum_separation"
            ),
            unsafe_separation_threshold=prediction_payload.get(
                f"unsafe_separation_threshold_{distance_unit}"
            ),
            unavailable_reason=prediction_payload.get("unavailable_reason"),
            is_forward_looking=prediction_payload.get("outcome")
            in ("PREDICTED_UNSAFE_PROXIMITY", "PREDICTED_TRAJECTORY_CONFLICT"),
        )

    time_to_risk = None
    if assessment.time_to_risk is not None:
        time_to_risk = TimeToRiskEvidence(
            status=assessment.time_to_risk.status,
            seconds=assessment.time_to_risk.seconds,
            threshold=assessment.time_to_risk.threshold,
            reason=assessment.time_to_risk.reason,
        )

    state = incident_state or derive_incident_state(assessment)

    return IncidentEvidence(
        incident_id=incident_id
        or f"INC-{assessment.timestamp:.2f}-{'+'.join(assessment.involved_entity_ids)}",
        timestamp=assessment.timestamp,
        coordinate_space=space,
        distance_unit=distance_unit,
        speed_unit=speed_unit,
        entities=entities,
        risk_score=assessment.risk_score,
        severity=assessment.severity,
        incident_type=assessment.incident_type,
        incident_state=state,
        confidence=assessment.confidence,
        factors=[
            FactorEvidence(
                name=f.name,
                score=f.score,
                weight=f.weight,
                contribution=f.contribution,
                rationale=f.rationale,
                confidence=f.confidence,
            )
            for f in assessment.contributing_factors
            if f.name != "escalation"
        ],
        prediction=prediction,
        time_to_risk=time_to_risk,
        current_separation=prediction_payload.get(f"current_separation_{distance_unit}"),
        closing_speed=prediction_payload.get(
            f"closing_speed_{'m_per_s' if metric else 'px_per_s'}"
        ),
        triggered_event_ids=list(assessment.evidence_event_ids),
        triggered_event_actions=_actions_for(assessment.evidence_event_ids, events),
        recommended_intervention=assessment.recommended_intervention,
        calibration_active=(
            metric if calibration_active is None else bool(calibration_active)
        ),
        space_fallback_reason=assessment.space_fallback_reason,
    )


def _actions_for(event_ids: Sequence[str], events: Sequence[Any]) -> List[str]:
    """Which kinds of event backed this assessment.

    ``events`` is optional: without it the evidence simply carries no action
    labels rather than guessing at them from the event IDs.
    """
    if not events:
        return []
    wanted = set(event_ids)
    return sorted(
        {e.action for e in events if getattr(e, "event_id", None) in wanted}
    )


def _zones_for(assessment: RiskAssessment, entity_id: str) -> List[str]:
    """Zones this entity occupied, as recorded by the zone factor.

    Read from the factor's ``zone_occupancy`` map, which the zone factor
    writes for every entity in the candidate. Nothing is inferred from the
    rationale text.
    """
    zone_factor = assessment.factor("zone")
    if zone_factor is None:
        return []
    occupancy = zone_factor.details.get("zone_occupancy") or {}
    return list(occupancy.get(entity_id, []))
