"""Image-space motion estimation from the event stream.

This module answers one question: *given an entity's recent events, where is it
and which way is it going?* It reads :class:`~app.events.schema.Event` objects
only — it never sees a Track, a frame or a model output.

Units — read this before using anything here
--------------------------------------------
Velocity is **pixels per second in the image plane**. It is named
``velocity_px_per_s`` everywhere, without exception, because it is not and
cannot be converted to metres per second:

* An object moving at constant physical speed produces a *changing* pixel
  velocity as its distance from the camera changes.
* Two objects at different depths moving at the same pixel velocity have very
  different physical speeds.
* A camera pan produces pixel velocity for objects that are not moving at all.

Times derived from these velocities (time to closest approach, for example)
are in real seconds and *are* meaningful as durations — but only under a
constant-image-velocity assumption, which is itself an approximation. See
:mod:`app.spatial` for the full coordinate-space caveat.

Ground-plane motion (M0.4)
--------------------------
When a :class:`~app.calibration.planar.GroundPlaneCalibration` is supplied, the
estimator *additionally* produces :class:`WorldMotion` in metres and metres per
second, attached to the image motion as ``.world``. Two rules make this safe:

* World velocity is fitted from **world positions over time**, never by scaling
  image velocity. Under perspective a fixed pixel velocity corresponds to a
  changing ground velocity, so scaling would be wrong by a factor that varies
  across the frame.
* With no calibration, ``.world`` is ``None`` and every field stays in pixels.
  There is no path by which pixel numbers acquire metric names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..calibration.planar import GroundPlaneCalibration, GroundPoint
from ..events.schema import Event
from ..spatial import (
    GROUND_PLANE_METERS,
    IMAGE_PIXELS,
    UNIT_METERS,
    Point,
    euclidean_distance,
    pixel_distance,
)

#: Minimum number of positioned observations before a velocity is estimated.
#: Two points define a line but cannot distinguish motion from detector jitter.
DEFAULT_MIN_SAMPLES = 3

#: Minimum time the observations must span. Three samples 10ms apart say
#: nothing useful about where something will be in two seconds.
DEFAULT_MIN_TIME_SPAN = 0.2

#: Sample count and time span at which trajectory confidence reaches 1.0.
DEFAULT_FULL_CONFIDENCE_SAMPLES = 6
DEFAULT_FULL_CONFIDENCE_SPAN = 1.0

#: How far back to look when estimating current motion. Older observations
#: describe where the entity *was* going, not where it is going now.
DEFAULT_LOOKBACK_SECONDS = 2.0

#: An entity not seen for this long is treated as unobservable rather than
#: assumed to have continued on its last heading.
DEFAULT_MAX_STALENESS_SECONDS = 1.0

#: Below this image-plane speed an entity is treated as stationary and given
#: no trajectory. Prevents detector jitter from manufacturing a heading.
DEFAULT_STATIONARY_SPEED_PX_PER_S = 8.0


@dataclass
class WorldMotion:
    """An entity's position and heading on the calibrated ground plane.

    Every field is in metres or metres per second, and says so in its name.
    This type only ever exists when a calibration was supplied, so the presence
    of a ``WorldMotion`` is itself the evidence that metric units are justified.
    """

    entity_id: str
    timestamp: float
    position_m: Point
    velocity_m_per_s: Point = (0.0, 0.0)
    samples: int = 0
    time_span_seconds: float = 0.0
    is_estimable: bool = False
    is_stationary: bool = True
    confidence: float = 0.0
    in_calibrated_region: bool = True
    coordinate_space: str = GROUND_PLANE_METERS
    units: str = UNIT_METERS

    @property
    def speed_m_per_s(self) -> float:
        vx, vy = self.velocity_m_per_s
        return (vx * vx + vy * vy) ** 0.5

    def predict_position_m(self, seconds_ahead: float) -> Point:
        """Constant-ground-velocity extrapolation, in metres."""
        if not self.is_estimable:
            return self.position_m
        vx, vy = self.velocity_m_per_s
        return (
            self.position_m[0] + vx * seconds_ahead,
            self.position_m[1] + vy * seconds_ahead,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "position_m": [round(v, 3) for v in self.position_m],
            "velocity_m_per_s": [round(v, 3) for v in self.velocity_m_per_s],
            "speed_m_per_s": round(self.speed_m_per_s, 3),
            "samples": self.samples,
            "time_span_seconds": round(self.time_span_seconds, 3),
            "is_estimable": self.is_estimable,
            "is_stationary": self.is_stationary,
            "confidence": round(self.confidence, 4),
            "in_calibrated_region": self.in_calibrated_region,
            "coordinate_space": self.coordinate_space,
            "units": self.units,
        }


@dataclass
class ImageMotion:
    """An entity's position and heading in image space at a moment in time.

    ``is_estimable`` is the honest-uncertainty flag: when it is ``False`` the
    velocity is zero and no consumer may predict where this entity is going.
    """

    entity_id: str
    timestamp: float
    position_px: Point
    velocity_px_per_s: Point = (0.0, 0.0)
    samples: int = 0
    time_span_seconds: float = 0.0
    last_seen: float = 0.0
    class_name: Optional[str] = None
    is_estimable: bool = False
    is_stationary: bool = True
    confidence: float = 0.0
    coordinate_space: str = IMAGE_PIXELS
    evidence_event_ids: List[str] = field(default_factory=list)
    #: Ground-plane motion, present only when a calibration was supplied.
    #: ``None`` means the system is in pixel-space mode, explicitly.
    world: Optional[WorldMotion] = None

    @property
    def has_world_motion(self) -> bool:
        return self.world is not None

    @property
    def speed_px_per_s(self) -> float:
        vx, vy = self.velocity_px_per_s
        return (vx * vx + vy * vy) ** 0.5

    def predict_position_px(self, seconds_ahead: float) -> Point:
        """Constant-image-velocity extrapolation.

        Returns the current position unchanged when the motion is not
        estimable — refusing to guess rather than extrapolating from noise.
        """
        if not self.is_estimable:
            return self.position_px
        vx, vy = self.velocity_px_per_s
        return (
            self.position_px[0] + vx * seconds_ahead,
            self.position_px[1] + vy * seconds_ahead,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "entity_id": self.entity_id,
            "class_name": self.class_name,
            "timestamp": round(self.timestamp, 4),
            "position_px": [round(v, 2) for v in self.position_px],
            "velocity_px_per_s": [round(v, 2) for v in self.velocity_px_per_s],
            "speed_px_per_s": round(self.speed_px_per_s, 2),
            "samples": self.samples,
            "time_span_seconds": round(self.time_span_seconds, 3),
            "is_estimable": self.is_estimable,
            "is_stationary": self.is_stationary,
            "confidence": round(self.confidence, 4),
            "coordinate_space": self.coordinate_space,
        }
        if self.world is not None:
            payload["world"] = self.world.to_dict()
        return payload


@dataclass
class MotionEstimator:
    """Estimates :class:`ImageMotion` from an entity's events.

    Velocity is a least-squares linear fit of position against time over the
    lookback window, which is deterministic and rejects single-frame outliers
    better than differencing the endpoints.

    Supplying ``calibration`` additionally produces ground-plane motion; with
    ``calibration=None`` (the default) behaviour is identical to M0.3 and
    everything stays in pixels.
    """

    min_samples: int = DEFAULT_MIN_SAMPLES
    min_time_span: float = DEFAULT_MIN_TIME_SPAN
    lookback_seconds: float = DEFAULT_LOOKBACK_SECONDS
    max_staleness_seconds: float = DEFAULT_MAX_STALENESS_SECONDS
    stationary_speed_px_per_s: float = DEFAULT_STATIONARY_SPEED_PX_PER_S
    full_confidence_samples: int = DEFAULT_FULL_CONFIDENCE_SAMPLES
    full_confidence_span: float = DEFAULT_FULL_CONFIDENCE_SPAN
    calibration: Optional[GroundPlaneCalibration] = None
    #: Ground speed below which an entity counts as stationary, in m/s.
    #: ~0.2 m/s is slower than a slow walk.
    stationary_speed_m_per_s: float = 0.2

    @property
    def coordinate_space(self) -> str:
        """The richest space this estimator can produce."""
        return (
            self.calibration.coordinate_space
            if self.calibration is not None
            else IMAGE_PIXELS
        )

    def estimate(
        self, events: Sequence[Event], at: float, entity_id: Optional[str] = None
    ) -> Optional[ImageMotion]:
        """Estimate motion at time ``at`` from an entity's event history.

        Returns ``None`` when the entity has no positioned observation at or
        before ``at``, or when its last sighting is stale.
        """
        samples = [
            e
            for e in events
            if e.position is not None and e.timestamp <= at
        ]
        if not samples:
            return None

        entity_id = entity_id or samples[-1].entity_id
        last = samples[-1]
        if at - last.timestamp > self.max_staleness_seconds:
            return None

        window = [e for e in samples if last.timestamp - e.timestamp <= self.lookback_seconds]
        times = [e.timestamp for e in window]
        span = max(times) - min(times) if times else 0.0

        motion = ImageMotion(
            entity_id=entity_id,
            timestamp=at,
            position_px=(float(last.position[0]), float(last.position[1])),
            samples=len(window),
            time_span_seconds=round(span, 4),
            last_seen=last.timestamp,
            class_name=last.attributes.get("class_name"),
            evidence_event_ids=[e.event_id for e in window if e.event_id],
        )

        if len(window) < self.min_samples or span < self.min_time_span:
            # Not enough history. Leave velocity at zero, is_estimable False
            # and confidence 0 — the engine must not predict from this.
            return motion

        motion.velocity_px_per_s = _least_squares_velocity(window)
        motion.is_estimable = True
        motion.is_stationary = motion.speed_px_per_s < self.stationary_speed_px_per_s
        motion.confidence = self._confidence(len(window), span)

        if self.calibration is not None:
            motion.world = self._world_motion(window, motion, at, entity_id)
        return motion

    # ------------------------------------------------------------------
    def _world_motion(
        self,
        window: Sequence[Event],
        image_motion: ImageMotion,
        at: float,
        entity_id: str,
    ) -> Optional[WorldMotion]:
        """Fit ground-plane motion from the same observation window.

        The fit is done on **ground positions**, not by rescaling the image
        velocity: under perspective the pixels-per-metre ratio changes as an
        object moves, so a scaled image velocity would be wrong by a factor
        that varies across the frame.
        """
        calibration = self.calibration
        if calibration is None:
            return None

        ground: List[GroundPoint] = []
        times: List[float] = []
        for event in window:
            projected = _project_event(calibration, event)
            if projected is None:
                continue
            ground.append(projected)
            times.append(event.timestamp)

        if not ground:
            return None

        last = ground[-1]
        world = WorldMotion(
            entity_id=entity_id,
            timestamp=at,
            position_m=last.as_tuple,
            samples=len(ground),
            time_span_seconds=round(max(times) - min(times), 4),
            in_calibrated_region=last.in_calibrated_region,
            coordinate_space=calibration.coordinate_space,
            units=calibration.world_units,
        )

        span = max(times) - min(times)
        if len(ground) < self.min_samples or span < self.min_time_span:
            return world

        world.velocity_m_per_s = _least_squares_slope(
            times, [p.as_tuple for p in ground]
        )
        world.is_estimable = True
        world.is_stationary = world.speed_m_per_s < self.stationary_speed_m_per_s
        world.confidence = image_motion.confidence
        return world

    def _confidence(self, samples: int, span: float) -> float:
        """Confidence in the velocity estimate, in ``[0, 1]``.

        The product of two independently sensible ratios: how many samples we
        have relative to what we'd like, and how long they span relative to
        what we'd like. Both saturate at 1.0.
        """
        sample_term = min(1.0, samples / float(self.full_confidence_samples))
        span_term = min(1.0, span / float(self.full_confidence_span))
        return round(sample_term * span_term, 4)


def _project_event(
    calibration: GroundPlaneCalibration, event: Event
) -> Optional[GroundPoint]:
    """Ground position of one observation.

    Prefers the bounding box's ground-contact anchor over the box centre: the
    centre of an upright object floats above the floor, and projecting it onto
    the calibrated plane would report a position the object is not standing at.
    """
    try:
        if event.bbox is not None:
            return calibration.image_box_to_world(event.bbox)
        if event.position is not None:
            return calibration.image_to_world(event.position)
    except ValueError:
        # Maps to the transform's horizon: no finite ground position exists.
        return None
    return None


def _least_squares_velocity(events: Sequence[Event]) -> Point:
    """Slope of image position against time, per axis, in px/s."""
    return _least_squares_slope(
        [e.timestamp for e in events],
        [(float(e.position[0]), float(e.position[1])) for e in events],
    )


def _least_squares_slope(times: Sequence[float], points: Sequence[Point]) -> Point:
    """Per-axis slope of position against time.

    Unit-agnostic: the caller supplies pixels or metres and gets px/s or m/s
    back. Shared so the image and ground fits can never drift apart.
    """
    if not times or len(times) != len(points):
        return (0.0, 0.0)
    mean_t = sum(times) / len(times)
    denominator = sum((t - mean_t) ** 2 for t in times)
    if denominator <= 0:
        return (0.0, 0.0)

    slope = []
    for axis in (0, 1):
        values = [float(p[axis]) for p in points]
        mean_v = sum(values) / len(values)
        numerator = sum((t - mean_t) * (v - mean_v) for t, v in zip(times, values))
        slope.append(numerator / denominator)
    return (slope[0], slope[1])


# ---------------------------------------------------------------------------
# Pairwise geometry (all image-space)
# ---------------------------------------------------------------------------
@dataclass
class ClosestApproach:
    """Constant-image-velocity closest-point-of-approach solution.

    ``seconds_to_closest_approach`` is a real duration; ``distance_px`` is a
    pixel distance. Valid only while both entities keep their current image
    velocity, which is an approximation and not a physical prediction.
    """

    distance_px: float
    seconds_to_closest_approach: float
    current_separation_px: float
    closing_speed_px_per_s: float
    is_converging: bool
    is_estimable: bool
    coordinate_space: str = IMAGE_PIXELS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "closest_approach_distance_px": round(self.distance_px, 2),
            "seconds_to_closest_approach": round(self.seconds_to_closest_approach, 3),
            "current_separation_px": round(self.current_separation_px, 2),
            "closing_speed_px_per_s": round(self.closing_speed_px_per_s, 2),
            "is_converging": self.is_converging,
            "is_estimable": self.is_estimable,
            "coordinate_space": self.coordinate_space,
        }


@dataclass
class _ApproachSolution:
    """Units-agnostic closest-point-of-approach result."""

    distance: float
    seconds: float
    separation: float
    closing_speed: float
    is_converging: bool
    is_estimable: bool


def _solve_closest_approach(
    position_a: Point,
    velocity_a: Point,
    position_b: Point,
    velocity_b: Point,
    estimable: bool,
) -> _ApproachSolution:
    """Closest approach of two constant-velocity points, in their own units.

    With relative position ``r = p_b - p_a`` and relative velocity
    ``v = v_b - v_a``, separation is minimised at ``t* = -(r·v)/(v·v)``.
    A negative ``t*`` means the closest approach is in the past — the pair is
    already separating — and is clamped to zero.

    Shared by the image-space and ground-space wrappers so the two can never
    disagree about which way a pair is moving.
    """
    separation_now = euclidean_distance(position_a, position_b)

    if not estimable:
        return _ApproachSolution(
            separation_now, 0.0, separation_now, 0.0, False, False
        )

    rx = position_b[0] - position_a[0]
    ry = position_b[1] - position_a[1]
    vx = velocity_b[0] - velocity_a[0]
    vy = velocity_b[1] - velocity_a[1]

    relative_speed_squared = vx * vx + vy * vy
    if relative_speed_squared <= 1e-9:
        # Same heading and speed: separation never changes.
        return _ApproachSolution(
            separation_now, 0.0, separation_now, 0.0, False, True
        )

    t_star = -(rx * vx + ry * vy) / relative_speed_squared
    # Closing speed is the radial component of relative velocity: positive
    # when the gap is shrinking.
    closing_speed = 0.0
    if separation_now > 1e-9:
        closing_speed = -((rx * vx + ry * vy) / separation_now)

    if t_star <= 0.0:
        return _ApproachSolution(
            separation_now, 0.0, separation_now, closing_speed, False, True
        )

    closest = ((rx + vx * t_star) ** 2 + (ry + vy * t_star) ** 2) ** 0.5
    return _ApproachSolution(
        closest, t_star, separation_now, closing_speed, closing_speed > 0.0, True
    )


def closest_approach(a: ImageMotion, b: ImageMotion) -> ClosestApproach:
    """Solve for when two constant-velocity **image** tracks are nearest."""
    solution = _solve_closest_approach(
        a.position_px,
        a.velocity_px_per_s,
        b.position_px,
        b.velocity_px_per_s,
        estimable=a.is_estimable and b.is_estimable,
    )
    return ClosestApproach(
        distance_px=solution.distance,
        seconds_to_closest_approach=solution.seconds,
        current_separation_px=solution.separation,
        closing_speed_px_per_s=solution.closing_speed,
        is_converging=solution.is_converging,
        is_estimable=solution.is_estimable,
    )


@dataclass
class GroundClosestApproach:
    """Closest-approach solution on the calibrated ground plane.

    Distances are metres and speeds are metres per second — genuinely, because
    this type exists only when a calibration produced the positions it was
    solved from.
    """

    distance_m: float
    seconds_to_closest_approach: float
    current_separation_m: float
    closing_speed_m_per_s: float
    is_converging: bool
    is_estimable: bool
    in_calibrated_region: bool = True
    coordinate_space: str = GROUND_PLANE_METERS
    units: str = UNIT_METERS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "closest_approach_distance_m": round(self.distance_m, 3),
            "seconds_to_closest_approach": round(self.seconds_to_closest_approach, 3),
            "current_separation_m": round(self.current_separation_m, 3),
            "closing_speed_m_per_s": round(self.closing_speed_m_per_s, 3),
            "is_converging": self.is_converging,
            "is_estimable": self.is_estimable,
            "in_calibrated_region": self.in_calibrated_region,
            "coordinate_space": self.coordinate_space,
            "units": self.units,
        }


def ground_closest_approach(
    a: ImageMotion, b: ImageMotion
) -> Optional[GroundClosestApproach]:
    """Closest approach in metres, or ``None`` without a calibration.

    Returning ``None`` rather than falling back to pixels is deliberate: a
    caller asking for metres must never silently receive pixels.
    """
    if a.world is None or b.world is None:
        return None

    solution = _solve_closest_approach(
        a.world.position_m,
        a.world.velocity_m_per_s,
        b.world.position_m,
        b.world.velocity_m_per_s,
        estimable=a.world.is_estimable and b.world.is_estimable,
    )
    return GroundClosestApproach(
        distance_m=solution.distance,
        seconds_to_closest_approach=solution.seconds,
        current_separation_m=solution.separation,
        closing_speed_m_per_s=solution.closing_speed,
        is_converging=solution.is_converging,
        is_estimable=solution.is_estimable,
        in_calibrated_region=(
            a.world.in_calibrated_region and b.world.in_calibrated_region
        ),
        coordinate_space=a.world.coordinate_space,
        units=a.world.units,
    )


def ground_separation(a: ImageMotion, b: ImageMotion) -> Optional[float]:
    """Separation between two entities in metres, or ``None`` uncalibrated."""
    if a.world is None or b.world is None:
        return None
    return euclidean_distance(a.world.position_m, b.world.position_m)


def ground_displacement(motion: WorldMotion, seconds: float) -> float:
    """How far this entity travels in ``seconds`` at its current ground speed."""
    return motion.speed_m_per_s * float(seconds)
