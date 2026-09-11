"""Risk assessment data model.

Everything the risk engine produces is one of these types. They are plain
dataclasses with no behaviour beyond serialisation, so an assessment can be
logged, diffed, replayed and asserted on without running the engine again.

Units
-----
Every spatial quantity in an assessment is in **image pixels** and every
assessment states ``coordinate_space`` explicitly. Time is in real seconds
(video timestamps are real), so ``predicted_time_to_incident_seconds`` is a
genuine duration — but a *distance* or a *speed* here is pixel-space and must
never be reported as metres or m/s. See :mod:`app.spatial`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ...spatial import GROUND_PLANE_METERS, IMAGE_PIXELS

# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------
SEVERITY_NORMAL = "normal"
SEVERITY_LOW = "low"
SEVERITY_MEDIUM = "medium"
SEVERITY_HIGH = "high"
SEVERITY_CRITICAL = "critical"

SEVERITIES = (
    SEVERITY_NORMAL,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    SEVERITY_HIGH,
    SEVERITY_CRITICAL,
)

#: Inclusive lower bound of each severity band, highest first.
#:
#: The bands are wide at the bottom and narrow at the top on purpose: the
#: difference between 5 and 15 points is noise, while the difference between
#: 85 and 95 is the difference between "watch this" and "act now".
SEVERITY_THRESHOLDS = (
    (90, SEVERITY_CRITICAL),
    (65, SEVERITY_HIGH),
    (40, SEVERITY_MEDIUM),
    (20, SEVERITY_LOW),
    (0, SEVERITY_NORMAL),
)

Severity = str


def severity_for(risk_score: float) -> Severity:
    """Map a 0-100 risk score onto a severity band."""
    for lower_bound, severity in SEVERITY_THRESHOLDS:
        if risk_score >= lower_bound:
            return severity
    return SEVERITY_NORMAL


# ---------------------------------------------------------------------------
# Incident types
# ---------------------------------------------------------------------------
INCIDENT_PERSON_VEHICLE_COLLISION = "PERSON_VEHICLE_COLLISION_RISK"
INCIDENT_PERSON_IN_OPERATING_ZONE = "PERSON_IN_VEHICLE_OPERATING_ZONE"
INCIDENT_CONVERGING_TRAJECTORIES = "CONVERGING_TRAJECTORIES"
INCIDENT_RAPID_SEPARATION_DECREASE = "RAPID_SEPARATION_DECREASE"
INCIDENT_PERSISTENT_ZONE_VIOLATION = "PERSISTENT_ZONE_VIOLATION"
INCIDENT_ESCALATING_SITUATION = "ESCALATING_SITUATION"
#: Two entities are near each other but nothing is developing — no motion
#: toward each other, no zone breach. Reported for situational awareness, not
#: as a predicted incident.
INCIDENT_CLOSE_PROXIMITY = "CLOSE_PROXIMITY"
INCIDENT_NONE = "NO_SIGNIFICANT_RISK"

INCIDENT_TYPES = (
    INCIDENT_PERSON_VEHICLE_COLLISION,
    INCIDENT_PERSON_IN_OPERATING_ZONE,
    INCIDENT_CONVERGING_TRAJECTORIES,
    INCIDENT_RAPID_SEPARATION_DECREASE,
    INCIDENT_PERSISTENT_ZONE_VIOLATION,
    INCIDENT_ESCALATING_SITUATION,
    INCIDENT_CLOSE_PROXIMITY,
    INCIDENT_NONE,
)

IncidentType = str


# ---------------------------------------------------------------------------
# Factor scores
# ---------------------------------------------------------------------------
@dataclass
class FactorScore:
    """One contributing factor's verdict, fully inspectable.

    ``score`` is the factor's normalised opinion in ``[0, 1]``; ``weight`` is
    how many risk points a score of 1.0 is worth. ``contribution`` is their
    product — the points this factor actually put on the board. Nothing is
    hidden: ``rationale`` says why in words and ``details`` carries the raw
    measurements the score was computed from.
    """

    name: str
    score: float
    weight: float
    rationale: str = ""
    confidence: float = 1.0
    evidence_event_ids: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    coordinate_space: str = IMAGE_PIXELS

    def __post_init__(self) -> None:
        self.score = _clamp(float(self.score), 0.0, 1.0)
        self.confidence = _clamp(float(self.confidence), 0.0, 1.0)
        self.weight = float(self.weight)

    @property
    def contribution(self) -> float:
        """Risk points contributed = ``score × weight``."""
        return round(self.score * self.weight, 4)

    @property
    def is_active(self) -> bool:
        return self.score > 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "weight": self.weight,
            "contribution": self.contribution,
            "confidence": round(self.confidence, 4),
            "rationale": self.rationale,
            "evidence_event_ids": list(self.evidence_event_ids),
            "details": self.details,
            "coordinate_space": self.coordinate_space,
        }


# ---------------------------------------------------------------------------
# Assessments
# ---------------------------------------------------------------------------
@dataclass
class TimeToRisk:
    """When a situation is expected to cross the unsafe-separation threshold.

    Three distinct states, never conflated:

    ``already_unsafe``
        The threshold is crossed *now*. ``seconds`` is 0.0.
    ``predicted``
        Not yet unsafe, but predicted to become so. ``seconds`` is the
        closed-form crossing time.
    ``not_predicted``
        The paths do not reach the threshold within the horizon, or no
        prediction is mathematically supported. ``seconds`` is ``None`` and
        ``reason`` says which.
    """

    status: str
    seconds: Optional[float] = None
    reason: Optional[str] = None
    threshold: Optional[float] = None
    coordinate_space: str = IMAGE_PIXELS
    units: str = "pixels"

    STATUS_ALREADY_UNSAFE = "already_unsafe"
    STATUS_PREDICTED = "predicted"
    STATUS_NOT_PREDICTED = "not_predicted"

    @property
    def is_predicted(self) -> bool:
        return self.status == self.STATUS_PREDICTED

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "status": self.status,
            "seconds": round(self.seconds, 3) if self.seconds is not None else None,
            "coordinate_space": self.coordinate_space,
            "units": self.units,
        }
        if self.reason:
            payload["reason"] = self.reason
        if self.threshold is not None:
            suffix = "m" if self.coordinate_space == GROUND_PLANE_METERS else "px"
            payload[f"unsafe_separation_threshold_{suffix}"] = round(self.threshold, 3)
        return payload


@dataclass
class RiskAssessment:
    """A single developing situation, scored and explained."""

    risk_score: float
    severity: Severity
    incident_type: IncidentType
    involved_entity_ids: List[str]
    timestamp: float
    confidence: float = 0.0
    predicted_time_to_incident_seconds: Optional[float] = None
    #: What the constant-velocity predictor says about this pair.
    prediction_outcome: Optional[str] = None
    #: Structured time-to-risk, distinguishing current from predicted.
    time_to_risk: Optional[TimeToRisk] = None
    #: Why this assessment is not in the configured space, if it is not.
    space_fallback_reason: Optional[str] = None
    contributing_factors: List[FactorScore] = field(default_factory=list)
    evidence_event_ids: List[str] = field(default_factory=list)
    recommended_intervention: str = ""
    escalation_multiplier: float = 1.0
    coordinate_space: str = IMAGE_PIXELS
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def factor_names(self) -> List[str]:
        return [f.name for f in self.contributing_factors]

    def factor(self, name: str) -> Optional[FactorScore]:
        """Look one contributing factor up by name."""
        for candidate in self.contributing_factors:
            if candidate.name == name:
                return candidate
        return None

    @property
    def active_factors(self) -> List[FactorScore]:
        return [f for f in self.contributing_factors if f.is_active]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_score": round(self.risk_score, 2),
            "severity": self.severity,
            "incident_type": self.incident_type,
            "involved_entity_ids": list(self.involved_entity_ids),
            "timestamp": round(self.timestamp, 4),
            "confidence": round(self.confidence, 4),
            "predicted_time_to_incident_seconds": (
                round(self.predicted_time_to_incident_seconds, 3)
                if self.predicted_time_to_incident_seconds is not None
                else None
            ),
            "prediction_outcome": self.prediction_outcome,
            "time_to_risk": self.time_to_risk.to_dict() if self.time_to_risk else None,
            "space_fallback_reason": self.space_fallback_reason,
            "contributing_factors": [f.to_dict() for f in self.contributing_factors],
            "evidence_event_ids": list(self.evidence_event_ids),
            "recommended_intervention": self.recommended_intervention,
            "escalation_multiplier": round(self.escalation_multiplier, 3),
            "coordinate_space": self.coordinate_space,
            "details": self.details,
        }


@dataclass
class RiskReport:
    """Every assessment produced for one moment in time."""

    timestamp: float
    assessments: List[RiskAssessment] = field(default_factory=list)
    coordinate_space: str = IMAGE_PIXELS
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.assessments)

    def __iter__(self):
        return iter(self.assessments)

    @property
    def top(self) -> Optional[RiskAssessment]:
        """The highest-scoring assessment, or ``None`` if there are none."""
        return self.assessments[0] if self.assessments else None

    @property
    def max_score(self) -> float:
        return self.top.risk_score if self.top else 0.0

    @property
    def severity(self) -> Severity:
        return self.top.severity if self.top else SEVERITY_NORMAL

    def at_or_above(self, severity: Severity) -> List[RiskAssessment]:
        """Assessments at least as severe as ``severity``."""
        floor = SEVERITIES.index(severity)
        return [a for a in self.assessments if SEVERITIES.index(a.severity) >= floor]

    def for_entity(self, entity_id: str) -> List[RiskAssessment]:
        return [a for a in self.assessments if entity_id in a.involved_entity_ids]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 4),
            "coordinate_space": self.coordinate_space,
            "score_interpretation": (
                "0-100 ordinal risk score; NOT a calibrated probability"
            ),
            "max_risk_score": round(self.max_score, 2),
            "severity": self.severity,
            "assessment_count": len(self.assessments),
            "assessments": [a.to_dict() for a in self.assessments],
            "metadata": self.metadata,
        }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
