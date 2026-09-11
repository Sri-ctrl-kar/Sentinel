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

from ..spatial import GROUND_PLANE_METERS, IMAGE_PIXELS, ZoneSet

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


@dataclass(frozen=True)
class SpatialThresholds:
    """Distance and speed thresholds for exactly ONE coordinate space.

    The risk factors are written against this view rather than against named
    config fields, so the same formula serves both spaces without the values
    ever being converted between them. Pixels and metres are configured
    independently and chosen independently; nothing here divides one by the
    other.
    """

    coordinate_space: str
    #: Suffix used when writing measurements into factor details, e.g. "px".
    distance_unit: str
    #: Suffix for speeds, e.g. "px_per_s".
    speed_unit: str
    #: Human-readable unit for rationales, e.g. "px" or "m".
    distance_label: str
    speed_label: str

    interaction_radius: float
    critical_radius: float
    closing_speed_reference: float
    closing_speed_floor: float
    conflict_radius: float
    miss_radius: float
    #: Separation at or below which a pair is considered unsafely close.
    #: Used by trajectory prediction for threshold-crossing times.
    unsafe_separation: float

    # ------------------------------------------------------------------
    def distance_key(self, name: str) -> str:
        """Detail key for a distance, e.g. ``separation`` -> ``separation_px``."""
        return f"{name}_{self.distance_unit}"

    def speed_key(self, name: str) -> str:
        return f"{name}_{self.speed_unit}"

    def format_distance(self, value: float) -> str:
        if self.distance_unit == "px":
            return f"{value:.0f}px"
        return f"{value:.2f} m"

    def format_speed(self, value: float) -> str:
        if self.speed_unit == "px_per_s":
            return f"{value:.0f}px/s"
        return f"{value:.2f} m/s"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coordinate_space": self.coordinate_space,
            self.distance_key("interaction_radius"): self.interaction_radius,
            self.distance_key("critical_radius"): self.critical_radius,
            self.distance_key("conflict_radius"): self.conflict_radius,
            self.distance_key("miss_radius"): self.miss_radius,
            self.distance_key("unsafe_separation"): self.unsafe_separation,
            self.speed_key("closing_speed_reference"): self.closing_speed_reference,
            self.speed_key("closing_speed_floor"): self.closing_speed_floor,
        }


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

    # --- ground plane (M0.5) ---------------------------------------------
    # Metric thresholds, chosen independently of the pixel ones above. They are
    # NOT conversions: no pixels-per-metre ratio exists outside a specific
    # calibration, so deriving one set from the other would be meaningless.
    #
    # These are engineering judgement for a warehouse-scale scene and have NOT
    # been validated against incident data. They are defensible starting points,
    # not measurements.

    #: Beyond ~6.5m a person and a vehicle are not meaningfully interacting.
    interaction_radius_m: float = 6.5

    #: Within ~1.5m a person cannot reliably step clear of a moving vehicle.
    critical_radius_m: float = 1.5

    #: ~4 m/s closing (roughly 14 km/h) saturates the closing-speed factor.
    closing_speed_reference_m_per_s: float = 4.0

    #: Below 0.25 m/s the gap is effectively steady.
    closing_speed_floor_m_per_s: float = 0.25

    #: A predicted closest approach within 2m is a genuine trajectory conflict —
    #: roughly the combined body width of a person and a forklift plus margin.
    conflict_radius_m: float = 2.0

    #: Predicted closest approach beyond 8m contributes no trajectory risk.
    trajectory_miss_radius_m: float = 8.0

    #: Separation at or below which the pair is unsafely close. Time-to-risk is
    #: the predicted moment of crossing this.
    unsafe_separation_m: float = 2.0

    #: Image-space equivalent of the above, for uncalibrated cameras.
    unsafe_separation_px: float = 90.0

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
        if self.critical_radius_m >= self.interaction_radius_m:
            raise ValueError("critical_radius_m must be < interaction_radius_m")
        if self.conflict_radius_m >= self.trajectory_miss_radius_m:
            raise ValueError("conflict_radius_m must be < trajectory_miss_radius_m")

    def is_person(self, class_name: Optional[str]) -> bool:
        return class_name is not None and class_name.lower() in {
            c.lower() for c in self.person_classes
        }

    def is_vehicle(self, class_name: Optional[str]) -> bool:
        return class_name is not None and class_name.lower() in {
            c.lower() for c in self.vehicle_classes
        }

    # ------------------------------------------------------------------
    def thresholds(self, coordinate_space: str = IMAGE_PIXELS) -> SpatialThresholds:
        """The threshold set for one coordinate space.

        Raises rather than guessing for an unknown space: a silently wrong unit
        is the single most dangerous failure this system can have.
        """
        if coordinate_space == IMAGE_PIXELS:
            return SpatialThresholds(
                coordinate_space=IMAGE_PIXELS,
                distance_unit="px",
                speed_unit="px_per_s",
                distance_label="px",
                speed_label="px/s",
                interaction_radius=self.interaction_radius_px,
                critical_radius=self.critical_radius_px,
                closing_speed_reference=self.closing_speed_reference_px_per_s,
                closing_speed_floor=self.closing_speed_floor_px_per_s,
                conflict_radius=self.conflict_radius_px,
                miss_radius=self.trajectory_miss_radius_px,
                unsafe_separation=self.unsafe_separation_px,
            )
        if coordinate_space == GROUND_PLANE_METERS:
            return SpatialThresholds(
                coordinate_space=GROUND_PLANE_METERS,
                distance_unit="m",
                speed_unit="m_per_s",
                distance_label="m",
                speed_label="m/s",
                interaction_radius=self.interaction_radius_m,
                critical_radius=self.critical_radius_m,
                closing_speed_reference=self.closing_speed_reference_m_per_s,
                closing_speed_floor=self.closing_speed_floor_m_per_s,
                conflict_radius=self.conflict_radius_m,
                miss_radius=self.trajectory_miss_radius_m,
                unsafe_separation=self.unsafe_separation_m,
            )
        raise ValueError(
            f"No thresholds defined for coordinate space '{coordinate_space}'."
        )

    # ------------------------------------------------------------------
    def threshold_audit(self) -> List[Dict[str, Any]]:
        """Every threshold, with its units, purpose and provenance.

        ``provenance`` is the honest column. As of M0.6 **every row is
        engineering-selected**: the values are defensible reasoning about a
        warehouse-scale scene, not measurements. Nothing here has been fitted
        to incident data, and nothing has been tuned against the benchmark —
        tuning against the evaluation set would make the benchmark a
        measurement of itself.

        A row may only become ``empirically-validated`` when it has been
        calibrated against annotated real footage held out from tuning.
        """
        engineering = "engineering-selected"
        rows: List[Dict[str, Any]] = [
            ("interaction_radius_px", self.interaction_radius_px, "px",
             "beyond this an image-space pair is not interacting", engineering),
            ("critical_radius_px", self.critical_radius_px, "px",
             "image-space separation at which proximity saturates", engineering),
            ("conflict_radius_px", self.conflict_radius_px, "px",
             "predicted miss distance counting as an image-space conflict", engineering),
            ("trajectory_miss_radius_px", self.trajectory_miss_radius_px, "px",
             "predicted miss beyond this contributes no trajectory risk", engineering),
            ("unsafe_separation_px", self.unsafe_separation_px, "px",
             "image-space separation defining an unsafe state", engineering),
            ("closing_speed_reference_px_per_s", self.closing_speed_reference_px_per_s,
             "px/s", "image-space closing speed at which the factor saturates",
             engineering),
            ("closing_speed_floor_px_per_s", self.closing_speed_floor_px_per_s,
             "px/s", "below this the image-space gap counts as steady", engineering),
            ("interaction_radius_m", self.interaction_radius_m, "m",
             "beyond this a person and vehicle are not interacting", engineering),
            ("critical_radius_m", self.critical_radius_m, "m",
             "separation at which a person cannot step clear", engineering),
            ("conflict_radius_m", self.conflict_radius_m, "m",
             "predicted miss distance counting as a trajectory conflict", engineering),
            ("trajectory_miss_radius_m", self.trajectory_miss_radius_m, "m",
             "predicted miss beyond this contributes no trajectory risk", engineering),
            ("unsafe_separation_m", self.unsafe_separation_m, "m",
             "ground separation defining an unsafe state; time-to-risk targets it",
             engineering),
            ("closing_speed_reference_m_per_s", self.closing_speed_reference_m_per_s,
             "m/s", "closing speed at which the factor saturates", engineering),
            ("closing_speed_floor_m_per_s", self.closing_speed_floor_m_per_s,
             "m/s", "below this the gap counts as steady", engineering),
            ("prediction_horizon_seconds", self.prediction_horizon_seconds, "s",
             "how far constant-velocity extrapolation is trusted", engineering),
            ("persistence_violation_threshold", self.persistence_violation_threshold,
             "count", "repeat zone entries at which persistence saturates", engineering),
            ("persistence_window_seconds", self.persistence_window_seconds, "s",
             "how far back persistence looks for violations", engineering),
            ("escalation_signal_threshold", self.escalation_signal_threshold, "score",
             "factor score counting as one corroborating signal", engineering),
            ("report_threshold", self.report_threshold, "points",
             "assessments below this are not reported", engineering),
            ("weight.proximity", self.weights.proximity, "points",
             "risk points for full proximity score", engineering),
            ("weight.closing_speed", self.weights.closing_speed, "points",
             "risk points for full closing-speed score", engineering),
            ("weight.trajectory", self.weights.trajectory, "points",
             "risk points for full trajectory score", engineering),
            ("weight.zone", self.weights.zone, "points",
             "risk points for full zone score", engineering),
            ("weight.persistence", self.weights.persistence, "points",
             "risk points for full persistence score", engineering),
        ]
        return [
            {
                "parameter": name,
                "value": value,
                "units": units,
                "purpose": purpose,
                "provenance": provenance,
            }
            for name, value, units, purpose, provenance in rows
        ]

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
            "thresholds": {
                IMAGE_PIXELS: self.thresholds(IMAGE_PIXELS).to_dict(),
                GROUND_PLANE_METERS: self.thresholds(GROUND_PLANE_METERS).to_dict(),
            },
            "thresholds_note": (
                "pixel and metre thresholds are configured independently; "
                "neither is derived from the other"
            ),
        }
