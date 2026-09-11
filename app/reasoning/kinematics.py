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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..events.schema import Event
from ..spatial import IMAGE_PIXELS, Point, pixel_distance

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
        return {
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


@dataclass
class MotionEstimator:
    """Estimates :class:`ImageMotion` from an entity's events.

    Velocity is a least-squares linear fit of position against time over the
    lookback window, which is deterministic and rejects single-frame outliers
    better than differencing the endpoints.
    """

    min_samples: int = DEFAULT_MIN_SAMPLES
    min_time_span: float = DEFAULT_MIN_TIME_SPAN
    lookback_seconds: float = DEFAULT_LOOKBACK_SECONDS
    max_staleness_seconds: float = DEFAULT_MAX_STALENESS_SECONDS
    stationary_speed_px_per_s: float = DEFAULT_STATIONARY_SPEED_PX_PER_S
    full_confidence_samples: int = DEFAULT_FULL_CONFIDENCE_SAMPLES
    full_confidence_span: float = DEFAULT_FULL_CONFIDENCE_SPAN

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
        return motion

    def _confidence(self, samples: int, span: float) -> float:
        """Confidence in the velocity estimate, in ``[0, 1]``.

        The product of two independently sensible ratios: how many samples we
        have relative to what we'd like, and how long they span relative to
        what we'd like. Both saturate at 1.0.
        """
        sample_term = min(1.0, samples / float(self.full_confidence_samples))
        span_term = min(1.0, span / float(self.full_confidence_span))
        return round(sample_term * span_term, 4)


def _least_squares_velocity(events: Sequence[Event]) -> Point:
    """Slope of position against time, per axis, in px/s."""
    times = [e.timestamp for e in events]
    mean_t = sum(times) / len(times)
    denominator = sum((t - mean_t) ** 2 for t in times)
    if denominator <= 0:
        return (0.0, 0.0)

    velocity = []
    for axis in (0, 1):
        values = [float(e.position[axis]) for e in events]
        mean_v = sum(values) / len(values)
        numerator = sum(
            (t - mean_t) * (v - mean_v) for t, v in zip(times, values)
        )
        velocity.append(numerator / denominator)
    return (velocity[0], velocity[1])


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


def closest_approach(a: ImageMotion, b: ImageMotion) -> ClosestApproach:
    """Solve for when two constant-velocity image tracks are nearest.

    With relative position ``r = p_b - p_a`` and relative velocity
    ``v = v_b - v_a``, separation is minimised at ``t* = -(r·v)/(v·v)``.
    A negative ``t*`` means the closest approach is in the past — the pair is
    already separating — and is clamped to zero.
    """
    separation_now = pixel_distance(a.position_px, b.position_px)

    if not (a.is_estimable and b.is_estimable):
        return ClosestApproach(
            distance_px=separation_now,
            seconds_to_closest_approach=0.0,
            current_separation_px=separation_now,
            closing_speed_px_per_s=0.0,
            is_converging=False,
            is_estimable=False,
        )

    rx = b.position_px[0] - a.position_px[0]
    ry = b.position_px[1] - a.position_px[1]
    vx = b.velocity_px_per_s[0] - a.velocity_px_per_s[0]
    vy = b.velocity_px_per_s[1] - a.velocity_px_per_s[1]

    relative_speed_squared = vx * vx + vy * vy
    if relative_speed_squared <= 1e-9:
        # Same heading and speed: separation never changes.
        return ClosestApproach(
            distance_px=separation_now,
            seconds_to_closest_approach=0.0,
            current_separation_px=separation_now,
            closing_speed_px_per_s=0.0,
            is_converging=False,
            is_estimable=True,
        )

    t_star = -(rx * vx + ry * vy) / relative_speed_squared
    # Closing speed is the radial component of relative velocity: positive
    # when the gap is shrinking.
    closing_speed = 0.0
    if separation_now > 1e-9:
        closing_speed = -((rx * vx + ry * vy) / separation_now)

    if t_star <= 0.0:
        # Already past the closest point: they are moving apart.
        return ClosestApproach(
            distance_px=separation_now,
            seconds_to_closest_approach=0.0,
            current_separation_px=separation_now,
            closing_speed_px_per_s=closing_speed,
            is_converging=False,
            is_estimable=True,
        )

    closest = ((rx + vx * t_star) ** 2 + (ry + vy * t_star) ** 2) ** 0.5
    return ClosestApproach(
        distance_px=closest,
        seconds_to_closest_approach=t_star,
        current_separation_px=separation_now,
        closing_speed_px_per_s=closing_speed,
        is_converging=closing_speed > 0.0,
        is_estimable=True,
    )
