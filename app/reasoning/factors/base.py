"""Factor interface and the shared evaluation context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ...events.schema import Event
from ...memory.temporal import EntityState, TemporalEventMemory
from ...spatial import IMAGE_PIXELS, ZoneSet
from ..config import RiskConfig
from ..kinematics import ClosestApproach, ImageMotion, closest_approach
from ..models.risk import FactorScore

#: Candidate kinds.
CANDIDATE_PAIR = "person_vehicle_pair"
CANDIDATE_ENTITY = "entity"


@dataclass
class RiskCandidate:
    """One situation the engine is considering.

    Either a person/vehicle pair (``primary`` is the person, ``secondary`` the
    vehicle) or a single entity being examined on its own.
    """

    entity_ids: List[str]
    primary: str
    secondary: Optional[str] = None
    kind: str = CANDIDATE_PAIR

    @property
    def is_pair(self) -> bool:
        return self.secondary is not None


@dataclass
class RiskContext:
    """Everything a factor may look at, and nothing more.

    A factor receives events, folded entity state, image-space motion and
    zone geometry. It has no access to frames, detections, tracks or any model
    — by construction, not by convention.
    """

    memory: TemporalEventMemory
    at: float
    config: RiskConfig
    candidate: RiskCandidate
    motions: Dict[str, ImageMotion] = field(default_factory=dict)
    coordinate_space: str = IMAGE_PIXELS
    _approach: Optional[ClosestApproach] = field(default=None, repr=False)
    _approach_computed: bool = field(default=False, repr=False)

    # ------------------------------------------------------------------
    @property
    def zones(self) -> ZoneSet:
        return self.config.zones or ZoneSet()

    def motion(self, entity_id: str) -> Optional[ImageMotion]:
        return self.motions.get(entity_id)

    def history(self, entity_id: str, **criteria: Any) -> List[Event]:
        """That entity's events up to and including ``at``."""
        return [
            e
            for e in self.memory.entity_history(entity_id, **criteria)
            if e.timestamp <= self.at
        ]

    def state(self, entity_id: str) -> Optional[EntityState]:
        return self.memory.state_at(entity_id, self.at)

    def class_of(self, entity_id: str) -> Optional[str]:
        motion = self.motion(entity_id)
        if motion is not None and motion.class_name:
            return motion.class_name
        state = self.state(entity_id)
        return state.class_name if state else None

    @property
    def primary_motion(self) -> Optional[ImageMotion]:
        return self.motion(self.candidate.primary)

    @property
    def secondary_motion(self) -> Optional[ImageMotion]:
        if self.candidate.secondary is None:
            return None
        return self.motion(self.candidate.secondary)

    def approach(self) -> Optional[ClosestApproach]:
        """Cached closest-point-of-approach solution for a pair candidate."""
        if self._approach_computed:
            return self._approach
        self._approach_computed = True
        primary, secondary = self.primary_motion, self.secondary_motion
        if primary is None or secondary is None:
            self._approach = None
        else:
            self._approach = closest_approach(primary, secondary)
        return self._approach


class RiskFactor(ABC):
    """One inspectable contribution to a risk score."""

    #: Stable identifier used in output and in tests.
    name: str = "factor"

    def __init__(self, weight: Optional[float] = None) -> None:
        self._weight_override = weight

    def weight(self, config: RiskConfig) -> float:
        if self._weight_override is not None:
            return self._weight_override
        return getattr(config.weights, self.name)

    @abstractmethod
    def evaluate(self, context: RiskContext) -> FactorScore:
        """Score this factor for the candidate in ``context``."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    def _score(
        self,
        context: RiskContext,
        score: float,
        rationale: str,
        confidence: float = 1.0,
        evidence_event_ids: Sequence[str] = (),
        **details: Any,
    ) -> FactorScore:
        return FactorScore(
            name=self.name,
            score=score,
            weight=self.weight(context.config),
            rationale=rationale,
            confidence=confidence,
            evidence_event_ids=list(evidence_event_ids),
            details=dict(details),
            coordinate_space=context.coordinate_space,
        )

    def _inactive(self, context: RiskContext, rationale: str, **details: Any) -> FactorScore:
        """A scored-zero result that still explains itself."""
        return self._score(context, 0.0, rationale, confidence=0.0, **details)


class MetaRiskFactor(RiskFactor):
    """A factor that reads the base factors' results rather than raw data.

    Escalation is the only one at M0.3: "several weak signals about the same
    pair" is a property of the *set* of factor scores, not of the geometry.
    """

    @abstractmethod
    def evaluate_with(
        self, context: RiskContext, scores: Sequence[FactorScore]
    ) -> FactorScore:
        raise NotImplementedError

    def evaluate(self, context: RiskContext) -> FactorScore:
        return self.evaluate_with(context, ())


def linear_falloff(value: float, full_at: float, zero_at: float) -> float:
    """Map ``value`` onto ``[0, 1]``, 1.0 at ``full_at`` and 0.0 at ``zero_at``.

    Used by several factors so the shape of "closer is worse" is defined once
    and is the same everywhere. Works in either direction: ``full_at`` may be
    smaller than ``zero_at`` (distance-like) or larger (speed-like).
    """
    if full_at == zero_at:
        return 1.0 if value == full_at else 0.0
    ratio = (zero_at - value) / (zero_at - full_at)
    return max(0.0, min(1.0, ratio))
