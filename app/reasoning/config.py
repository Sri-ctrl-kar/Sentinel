"""Risk engine configuration.

Every tunable lives here, with the reasoning for its default written next to
it. There are no magic numbers buried in the factor implementations, and no
opaque weights: a reviewer can read this file and know exactly what the engine
believes.

**All distances are image pixels and all speeds are pixels per second.** They
are suffixed ``_px`` and ``_px_per_s`` without exception. Defaults are quoted
for a roughly 640-960px-wide frame with subjects at mid-depth; they are not
physical quantities and must be re-tuned per camera. See :mod:`app.spatial`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..spatial import IMAGE_PIXELS, ZoneSet

#: COCO-ish class names treated as people.
DEFAULT_PERSON_CLASSES: Sequence[str] = ("person",)

#: Class names treated as vehicles or powered machinery. Forklift is not a
#: COCO class; it is listed so a custom-trained detector works unchanged.
DEFAULT_VEHICLE_CLASSES: Sequence[str] = (
    "car",
    "truck",
    "bus",
    "motorcycle",
    "train",
    "forklift",
)


@dataclass
class FactorWeights:
    """How many risk points each factor is worth at full score.

    The four base factors sum to 100, so a situation that maxes out every base
    factor scores 100 before persistence or escalation are considered. This is
    deliberate: **no single factor can reach `critical` on its own**. Proximity
    alone tops out at 30 points ("low"), which is the right answer — two things
    being near each other is not an incident.

    Persistence adds up to a further 15 points, and the escalation multiplier
    can raise the subtotal further, with the final score clamped to 100.
    """

    proximity: float = 30.0
    closing_speed: float = 25.0
    trajectory: float = 25.0
    zone: float = 20.0
    persistence: float = 15.0

    @property
    def base_total(self) -> float:
        return self.proximity + self.closing_speed + self.trajectory + self.zone

    def to_dict(self) -> Dict[str, float]:
        return {
            "proximity": self.proximity,
            "closing_speed": self.closing_speed,
            "trajectory": self.trajectory,
            "zone": self.zone,
            "persistence": self.persistence,
        }


@dataclass
class RiskConfig:
    """Thresholds and weights for the risk engine."""

    # --- what counts as what ---------------------------------------------
    person_classes: Sequence[str] = field(default_factory=lambda: list(DEFAULT_PERSON_CLASSES))
    vehicle_classes: Sequence[str] = field(default_factory=lambda: list(DEFAULT_VEHICLE_CLASSES))

    #: Zones in which a vehicle or machine operates. A person inside one of
    #: these is the situation the zone factor exists to catch.
    operating_zones: Sequence[str] = field(default_factory=list)

    #: Zones a person is not permitted to enter at all. Repeated entries drive
    #: the persistence factor.
    restricted_zones: Sequence[str] = field(default_factory=list)

    # --- proximity --------------------------------------------------------
    #: Beyond this image-plane separation the pair is not considered to be
    #: interacting at all; proximity scores 0.
    interaction_radius_px: float = 260.0

    #: At or below this separation proximity scores 1.0. Roughly "close enough
    #: that a person could be struck before reacting", in pixels, for a
    #: mid-depth subject.
    critical_radius_px: float = 70.0

    # --- closing speed ----------------------------------------------------
    #: Closing speed (radial component of relative image velocity) at which
    #: the closing-speed factor saturates.
    closing_speed_reference_px_per_s: float = 160.0

    #: Below this closing speed the gap is treated as effectively steady.
    closing_speed_floor_px_per_s: float = 10.0

    # --- trajectory -------------------------------------------------------
    #: Closest-approach distance at or below which trajectories are treated as
    #: genuinely intersecting rather than merely converging.
    conflict_radius_px: float = 90.0

    #: Closest approach beyond this distance contributes no trajectory risk.
    trajectory_miss_radius_px: float = 320.0

    #: How far ahead the engine is willing to predict. A closest approach
    #: further out than this scores 0 — constant-velocity extrapolation over
    #: long horizons is not credible.
    prediction_horizon_seconds: float = 6.0

    # --- persistence / escalation ----------------------------------------
    #: Repeat violations at which the persistence factor saturates.
    persistence_violation_threshold: int = 3

    #: How far back persistence looks for prior violations.
    persistence_window_seconds: float = 60.0

    #: A factor scoring at least this much counts as one corroborating signal
    #: for escalation.
    escalation_signal_threshold: float = 0.20

    #: Corroboration multiplier by number of independent signals at or above
    #: `escalation_signal_threshold`. Index = signal count.
    #:
    #: Rationale: one factor firing is a measurement; several factors firing
    #: about the same pair is a *situation*. The multiplier is capped at 1.35
    #: so corroboration can promote a borderline case across a severity band
    #: but can never invent risk that no factor observed (0 × 1.35 is still 0).
    escalation_multipliers: Sequence[float] = (1.0, 1.0, 1.10, 1.20, 1.30, 1.35)

    # --- reporting --------------------------------------------------------
    #: Assessments scoring below this are dropped from the report, so a quiet
    #: scene produces an empty report rather than pages of zeroes.
    report_threshold: float = 1.0

    #: Zone geometry, needed to answer "is this person in the operating zone".
    zones: Optional[ZoneSet] = None

    weights: FactorWeights = field(default_factory=FactorWeights)

    coordinate_space: str = IMAGE_PIXELS

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.coordinate_space != IMAGE_PIXELS:
            raise ValueError(
                f"RiskConfig: only '{IMAGE_PIXELS}' is supported at M0.3; "
                f"got '{self.coordinate_space}'"
            )
        if self.critical_radius_px >= self.interaction_radius_px:
            raise ValueError("critical_radius_px must be < interaction_radius_px")
        if self.conflict_radius_px >= self.trajectory_miss_radius_px:
            raise ValueError("conflict_radius_px must be < trajectory_miss_radius_px")

    def is_person(self, class_name: Optional[str]) -> bool:
        return class_name is not None and class_name.lower() in {
            c.lower() for c in self.person_classes
        }

    def is_vehicle(self, class_name: Optional[str]) -> bool:
        return class_name is not None and class_name.lower() in {
            c.lower() for c in self.vehicle_classes
        }

    def escalation_multiplier(self, signal_count: int) -> float:
        """Corroboration multiplier for ``signal_count`` independent signals."""
        if signal_count < 0:
            raise ValueError("signal_count must be >= 0")
        table = self.escalation_multipliers
        return float(table[min(signal_count, len(table) - 1)])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coordinate_space": self.coordinate_space,
            "person_classes": list(self.person_classes),
            "vehicle_classes": list(self.vehicle_classes),
            "operating_zones": list(self.operating_zones),
            "restricted_zones": list(self.restricted_zones),
            "interaction_radius_px": self.interaction_radius_px,
            "critical_radius_px": self.critical_radius_px,
            "closing_speed_reference_px_per_s": self.closing_speed_reference_px_per_s,
            "conflict_radius_px": self.conflict_radius_px,
            "trajectory_miss_radius_px": self.trajectory_miss_radius_px,
            "prediction_horizon_seconds": self.prediction_horizon_seconds,
            "persistence_violation_threshold": self.persistence_violation_threshold,
            "escalation_signal_threshold": self.escalation_signal_threshold,
            "weights": self.weights.to_dict(),
        }
