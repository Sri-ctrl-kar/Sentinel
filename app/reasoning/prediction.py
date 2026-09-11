"""Explainable baseline trajectory prediction.

The model
--------
Constant velocity, in whichever coordinate space the motion was measured in::

    position(t) = position_0 + velocity * t

That is the entire model. It is chosen because it is the weakest assumption
that still predicts anything, and because every consequence of it can be
written down in closed form and checked by hand. A learned motion model would
predict better and explain worse; at this stage explaining worse is the greater
cost.

What is computed in closed form
-------------------------------
For two entities with relative position ``r = p_b - p_a`` and relative velocity
``v = v_b - v_a``:

* **Minimum separation** is at ``t* = -(r·v)/(v·v)``, clamped to ``t >= 0``.
* **Time to a separation threshold** ``d`` solves ``|r + v t| = d``, i.e. the
  quadratic ``(v·v)t² + 2(r·v)t + (r·r - d²) = 0``; the smallest non-negative
  root is when the pair first becomes that close.

Both are exact, so the reported crossing time does not depend on how finely the
trajectory happens to be sampled. A sampled series is produced as well, purely
so a reader can see the separation curve the closed form describes.

Terminology
-----------
This module never says "collision". It has no object extents, no 3D shape and
no knowledge of what the entities would do on seeing each other. It reports
**predicted trajectory conflict** and **predicted unsafe proximity**: statements
about where the lines go, not about what will hit what.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..spatial import GROUND_PLANE_METERS, IMAGE_PIXELS, Point
from .kinematics import ImageMotion, motion_in_space

# --- prediction outcomes ---------------------------------------------------
#: Paths stay clear of the unsafe separation threshold over the horizon.
NO_PREDICTED_CONFLICT = "NO_PREDICTED_CONFLICT"
#: The pair is predicted to come within the unsafe separation threshold.
PREDICTED_UNSAFE_PROXIMITY = "PREDICTED_UNSAFE_PROXIMITY"
#: Predicted closest approach falls inside the tighter conflict radius — the
#: paths do not merely come close, they effectively intersect.
PREDICTED_TRAJECTORY_CONFLICT = "PREDICTED_TRAJECTORY_CONFLICT"
#: The pair is unsafely close *right now*. A statement about the present, not a
#: prediction — kept distinct so a current state is never reported as foresight.
CURRENTLY_UNSAFE_PROXIMITY = "CURRENTLY_UNSAFE_PROXIMITY"
#: Not enough history, or the geometry does not support a prediction.
PREDICTION_UNAVAILABLE = "PREDICTION_UNAVAILABLE"

PREDICTION_OUTCOMES = (
    NO_PREDICTED_CONFLICT,
    PREDICTED_UNSAFE_PROXIMITY,
    PREDICTED_TRAJECTORY_CONFLICT,
    CURRENTLY_UNSAFE_PROXIMITY,
    PREDICTION_UNAVAILABLE,
)

#: Outcomes that mean the pair is at unsafe separation now or is predicted to be.
UNSAFE_OUTCOMES = (
    PREDICTED_UNSAFE_PROXIMITY,
    PREDICTED_TRAJECTORY_CONFLICT,
    CURRENTLY_UNSAFE_PROXIMITY,
)

#: Why a prediction could not be made. Reported instead of a fabricated number.
REASON_INSUFFICIENT_HISTORY = "insufficient_history"
REASON_BOTH_STATIONARY = "both_stationary"
REASON_NO_RELATIVE_MOTION = "no_relative_motion"
REASON_DIVERGING = "diverging"
REASON_NO_MOTION_DATA = "no_motion_data"

DEFAULT_HORIZON_SECONDS = 6.0
DEFAULT_TIMESTEP_SECONDS = 0.25


@dataclass
class TrajectorySample:
    """One point on a predicted separation curve."""

    seconds_ahead: float
    separation: float
    position_a: Point
    position_b: Point

    def to_dict(self, distance_key: str = "separation") -> Dict[str, Any]:
        return {
            "seconds_ahead": round(self.seconds_ahead, 3),
            distance_key: round(self.separation, 3),
        }


@dataclass
class PredictedTrajectory:
    """One entity's predicted path under constant velocity."""

    entity_id: str
    coordinate_space: str
    units: str
    origin: Point
    velocity: Point
    horizon_seconds: float
    points: List[Tuple[float, Point]] = field(default_factory=list)
    is_estimable: bool = True

    def position_at(self, seconds_ahead: float) -> Point:
        return (
            self.origin[0] + self.velocity[0] * seconds_ahead,
            self.origin[1] + self.velocity[1] * seconds_ahead,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "coordinate_space": self.coordinate_space,
            "units": self.units,
            "origin": [round(v, 3) for v in self.origin],
            "velocity": [round(v, 3) for v in self.velocity],
            "horizon_seconds": self.horizon_seconds,
            "is_estimable": self.is_estimable,
            "sample_count": len(self.points),
        }


@dataclass
class PairPrediction:
    """What two entities' paths are predicted to do relative to each other.

    ``outcome`` is always one of :data:`PREDICTION_OUTCOMES`. When it is
    ``PREDICTION_UNAVAILABLE``, ``unavailable_reason`` says why and every
    predictive field is ``None`` — the engine reports that it cannot predict
    rather than returning a confident-looking zero.
    """

    entity_ids: Tuple[str, str]
    coordinate_space: str
    units: str
    outcome: str
    horizon_seconds: float

    current_separation: Optional[float] = None
    closing_speed: Optional[float] = None
    minimum_separation: Optional[float] = None
    seconds_to_minimum_separation: Optional[float] = None
    seconds_to_unsafe_separation: Optional[float] = None
    unsafe_separation_threshold: Optional[float] = None
    conflict_radius: Optional[float] = None
    enters_unsafe_separation: bool = False
    is_converging: bool = False
    unavailable_reason: Optional[str] = None
    in_calibrated_region: bool = True
    samples: List[TrajectorySample] = field(default_factory=list)

    @property
    def is_available(self) -> bool:
        return self.outcome != PREDICTION_UNAVAILABLE

    @property
    def predicts_conflict(self) -> bool:
        """Is a *future* unsafe approach predicted?

        Excludes :data:`CURRENTLY_UNSAFE_PROXIMITY`, which describes the
        present. Conflating the two would let the system take credit for
        predicting something that had already happened.
        """
        return self.outcome in (
            PREDICTED_UNSAFE_PROXIMITY,
            PREDICTED_TRAJECTORY_CONFLICT,
        )

    @property
    def is_currently_unsafe(self) -> bool:
        return self.outcome == CURRENTLY_UNSAFE_PROXIMITY

    def separation_at(self, seconds_ahead: float) -> Optional[float]:
        """Predicted separation at a future moment, by interpolation of samples."""
        if not self.samples:
            return None
        nearest = min(self.samples, key=lambda s: abs(s.seconds_ahead - seconds_ahead))
        return nearest.separation

    def to_dict(self) -> Dict[str, Any]:
        distance = "m" if self.coordinate_space == GROUND_PLANE_METERS else "px"
        speed = "m_per_s" if distance == "m" else "px_per_s"
        payload: Dict[str, Any] = {
            "entity_ids": list(self.entity_ids),
            "coordinate_space": self.coordinate_space,
            "units": self.units,
            "outcome": self.outcome,
            "horizon_seconds": self.horizon_seconds,
            "enters_unsafe_separation": self.enters_unsafe_separation,
            "is_converging": self.is_converging,
            "in_calibrated_region": self.in_calibrated_region,
        }
        if self.unavailable_reason:
            payload["unavailable_reason"] = self.unavailable_reason
        for label, value in (
            (f"current_separation_{distance}", self.current_separation),
            (f"minimum_separation_{distance}", self.minimum_separation),
            (f"unsafe_separation_threshold_{distance}", self.unsafe_separation_threshold),
            (f"conflict_radius_{distance}", self.conflict_radius),
            (f"closing_speed_{speed}", self.closing_speed),
            ("seconds_to_minimum_separation", self.seconds_to_minimum_separation),
            ("seconds_to_unsafe_separation", self.seconds_to_unsafe_separation),
        ):
            if value is not None:
                payload[label] = round(value, 3)
        if self.samples:
            payload["separation_curve"] = [
                s.to_dict(f"separation_{distance}") for s in self.samples
            ]
        return payload


@dataclass
class TrajectoryPredictor:
    """Constant-velocity prediction over a configurable horizon."""

    horizon_seconds: float = DEFAULT_HORIZON_SECONDS
    timestep_seconds: float = DEFAULT_TIMESTEP_SECONDS
    #: Below this speed an entity is treated as not going anywhere, so two
    #: stationary entities never generate a predicted conflict from jitter.
    min_speed: float = 1e-6

    def __post_init__(self) -> None:
        if self.horizon_seconds <= 0:
            raise ValueError("horizon_seconds must be > 0")
        if self.timestep_seconds <= 0:
            raise ValueError("timestep_seconds must be > 0")

    # ------------------------------------------------------------------
    def predict_entity(
        self, motion: ImageMotion, coordinate_space: str
    ) -> Optional[PredictedTrajectory]:
        """One entity's path, or ``None`` when its motion is not estimable."""
        resolved = motion_in_space(motion, coordinate_space)
        if resolved is None:
            return None
        position, velocity, is_estimable, _stationary = resolved

        trajectory = PredictedTrajectory(
            entity_id=motion.entity_id,
            coordinate_space=coordinate_space,
            units=_units_for(coordinate_space),
            origin=position,
            velocity=velocity if is_estimable else (0.0, 0.0),
            horizon_seconds=self.horizon_seconds,
            is_estimable=is_estimable,
        )
        if is_estimable:
            trajectory.points = [
                (t, trajectory.position_at(t)) for t in self._timesteps()
            ]
        return trajectory

    # ------------------------------------------------------------------
    def predict_pair(
        self,
        a: ImageMotion,
        b: ImageMotion,
        coordinate_space: str,
        unsafe_separation: float,
        conflict_radius: Optional[float] = None,
    ) -> PairPrediction:
        """Predict how two entities' separation evolves.

        ``unsafe_separation`` and ``conflict_radius`` must be expressed in
        ``coordinate_space``'s units — the caller takes them from the matching
        :class:`~app.reasoning.config.SpatialThresholds`, so there is no path
        by which a pixel threshold reaches a metric prediction.
        """
        ids = (a.entity_id, b.entity_id)
        units = _units_for(coordinate_space)
        conflict_radius = (
            unsafe_separation if conflict_radius is None else conflict_radius
        )

        resolved_a = motion_in_space(a, coordinate_space)
        resolved_b = motion_in_space(b, coordinate_space)
        if resolved_a is None or resolved_b is None:
            return self._unavailable(ids, coordinate_space, units, REASON_NO_MOTION_DATA)

        pos_a, vel_a, estimable_a, stationary_a = resolved_a
        pos_b, vel_b, estimable_b, stationary_b = resolved_b
        separation_now = _distance(pos_a, pos_b)
        in_region = _in_region(a, b, coordinate_space)

        if not (estimable_a and estimable_b):
            prediction = self._unavailable(
                ids, coordinate_space, units, REASON_INSUFFICIENT_HISTORY
            )
            # The current separation is an observation, not a prediction, so it
            # is still reported.
            prediction.current_separation = separation_now
            prediction.unsafe_separation_threshold = unsafe_separation
            prediction.in_calibrated_region = in_region
            return prediction

        if stationary_a and stationary_b:
            prediction = self._unavailable(
                ids, coordinate_space, units, REASON_BOTH_STATIONARY
            )
            prediction.current_separation = separation_now
            prediction.unsafe_separation_threshold = unsafe_separation
            prediction.in_calibrated_region = in_region
            # Nothing is moving, so the outcome is knowable: no conflict.
            prediction.outcome = NO_PREDICTED_CONFLICT
            prediction.minimum_separation = separation_now
            prediction.seconds_to_minimum_separation = 0.0
            prediction.closing_speed = 0.0
            return prediction

        rx, ry = pos_b[0] - pos_a[0], pos_b[1] - pos_a[1]
        vx, vy = vel_b[0] - vel_a[0], vel_b[1] - vel_a[1]
        relative_speed_squared = vx * vx + vy * vy

        closing_speed = 0.0
        if separation_now > 1e-12:
            closing_speed = -((rx * vx + ry * vy) / separation_now)

        prediction = PairPrediction(
            entity_ids=ids,
            coordinate_space=coordinate_space,
            units=units,
            outcome=NO_PREDICTED_CONFLICT,
            horizon_seconds=self.horizon_seconds,
            current_separation=separation_now,
            closing_speed=closing_speed,
            unsafe_separation_threshold=unsafe_separation,
            conflict_radius=conflict_radius,
            is_converging=closing_speed > 0.0,
            in_calibrated_region=in_region,
        )

        if relative_speed_squared <= 1e-12:
            # Identical velocities: the gap never changes.
            prediction.minimum_separation = separation_now
            prediction.seconds_to_minimum_separation = 0.0
            prediction.unavailable_reason = REASON_NO_RELATIVE_MOTION
            prediction.samples = self._sample(pos_a, vel_a, pos_b, vel_b)
            if separation_now <= unsafe_separation:
                prediction.enters_unsafe_separation = True
                prediction.seconds_to_unsafe_separation = 0.0
                prediction.outcome = CURRENTLY_UNSAFE_PROXIMITY
            return prediction

        t_star = -(rx * vx + ry * vy) / relative_speed_squared
        t_star = max(0.0, min(t_star, self.horizon_seconds))
        minimum = _distance(
            (pos_a[0] + vel_a[0] * t_star, pos_a[1] + vel_a[1] * t_star),
            (pos_b[0] + vel_b[0] * t_star, pos_b[1] + vel_b[1] * t_star),
        )
        prediction.minimum_separation = minimum
        prediction.seconds_to_minimum_separation = t_star
        prediction.samples = self._sample(pos_a, vel_a, pos_b, vel_b)

        crossing = _time_to_separation(
            rx, ry, vx, vy, unsafe_separation, self.horizon_seconds
        )
        if crossing is not None:
            prediction.enters_unsafe_separation = True
            prediction.seconds_to_unsafe_separation = crossing
            if separation_now <= unsafe_separation:
                # Already inside the threshold: that is an observation about
                # now, not a prediction about later.
                prediction.outcome = CURRENTLY_UNSAFE_PROXIMITY
            else:
                prediction.outcome = (
                    PREDICTED_TRAJECTORY_CONFLICT
                    if minimum <= conflict_radius
                    else PREDICTED_UNSAFE_PROXIMITY
                )
        else:
            prediction.outcome = NO_PREDICTED_CONFLICT
        return prediction

    # ------------------------------------------------------------------
    def _timesteps(self) -> List[float]:
        steps: List[float] = []
        moment = 0.0
        while moment <= self.horizon_seconds + 1e-9:
            steps.append(round(moment, 6))
            moment += self.timestep_seconds
        return steps

    def _sample(
        self, pos_a: Point, vel_a: Point, pos_b: Point, vel_b: Point
    ) -> List[TrajectorySample]:
        samples = []
        for t in self._timesteps():
            a = (pos_a[0] + vel_a[0] * t, pos_a[1] + vel_a[1] * t)
            b = (pos_b[0] + vel_b[0] * t, pos_b[1] + vel_b[1] * t)
            samples.append(TrajectorySample(t, _distance(a, b), a, b))
        return samples

    def _unavailable(
        self, ids, coordinate_space: str, units: str, reason: str
    ) -> PairPrediction:
        return PairPrediction(
            entity_ids=ids,
            coordinate_space=coordinate_space,
            units=units,
            outcome=PREDICTION_UNAVAILABLE,
            horizon_seconds=self.horizon_seconds,
            unavailable_reason=reason,
        )


# ---------------------------------------------------------------------------
def _time_to_separation(
    rx: float, ry: float, vx: float, vy: float, threshold: float, horizon: float
) -> Optional[float]:
    """First time within ``horizon`` at which separation reaches ``threshold``.

    Solves ``|r + v t| = threshold`` exactly. Returns ``None`` when the pair
    never gets that close within the horizon, and ``0.0`` when it already is.
    """
    a = vx * vx + vy * vy
    b = 2.0 * (rx * vx + ry * vy)
    c = rx * rx + ry * ry - threshold * threshold

    if c <= 0.0:
        return 0.0  # already inside the threshold
    if a <= 1e-12:
        return None

    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None  # never reaches the threshold

    root = discriminant**0.5
    candidates = sorted(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)))
    for candidate in candidates:
        if 0.0 <= candidate <= horizon:
            return candidate
    return None


def _distance(a: Point, b: Point) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _units_for(coordinate_space: str) -> str:
    if coordinate_space == GROUND_PLANE_METERS:
        return "meters"
    if coordinate_space == IMAGE_PIXELS:
        return "pixels"
    raise ValueError(f"Unknown coordinate space '{coordinate_space}'")


def _in_region(a: ImageMotion, b: ImageMotion, coordinate_space: str) -> bool:
    if coordinate_space != GROUND_PLANE_METERS:
        return True
    if a.world is None or b.world is None:
        return True
    return a.world.in_calibrated_region and b.world.in_calibrated_region
