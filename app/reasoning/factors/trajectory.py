"""Trajectory convergence factor.

Answers: *if both entities keep their current image-plane heading, will they
end up in the same place, and how soon?*

This is the only factor that predicts. It is therefore the only one that can
be confidently wrong, and it is deliberately the most conservative:

* It refuses to score at all when either entity has insufficient history
  (Scenario G) — no velocity, no prediction.
* It refuses to score when both entities are effectively stationary
  (Scenario F) — detector jitter must never be read as a heading.
* It refuses to score beyond ``prediction_horizon_seconds`` — constant-velocity
  extrapolation over long horizons is not credible.

Works in whichever space the candidate is scored in. A "closest approach of
40px" is 40 pixels in the frame; a "closest approach of 1.2 m" is 1.2 metres on
the calibrated floor. The two are never mixed, and the thresholds come from the
active :class:`~app.reasoning.config.SpatialThresholds`.

The factor deliberately does not say "collision". It has no object extents and
no model of what the entities would do on seeing each other; it reports that
predicted paths converge inside a conflict radius. See
:mod:`app.reasoning.prediction` for the vocabulary.
"""

from __future__ import annotations

from ..models.risk import FactorScore
from .base import RiskContext, RiskFactor, linear_falloff


class TrajectoryFactor(RiskFactor):
    """Whether two image-space trajectories are converging on a conflict.

    Scoring
    -------
    The closest-point-of-approach solution gives a miss distance ``d`` and a
    time ``t`` until it happens. Two independent terms are multiplied:

    ``spatial``
        1.0 when the predicted miss distance is at or inside the conflict
        radius, falling to 0.0 at the miss radius.

    ``temporal``
        1.0 for an immediate conflict, falling linearly to 0.0 at
        ``prediction_horizon_seconds``.

    ::

        score = spatial × temporal

    Multiplied, not averaged, because both must hold: a collision course that
    resolves in 40 seconds is not urgent, and an imminent closest approach that
    misses by 300px is not a conflict. Averaging would let either one alone
    carry the factor, which is exactly the false-positive behaviour to avoid.
    """

    name = "trajectory"

    def evaluate(self, context: RiskContext) -> FactorScore:
        geometry = context.geometry()
        if geometry is None:
            return self._inactive(
                context, "no pair to measure: trajectory needs two observed entities"
            )

        primary = context.primary_motion
        secondary = context.secondary_motion
        config = context.config
        thresholds = context.thresholds

        # --- refusal 1: not enough history to have a heading at all --------
        if not geometry.is_estimable:
            missing = [
                m.entity_id
                for m in (primary, secondary)
                if m is not None and not m.is_estimable
            ]
            return self._inactive(
                context,
                "insufficient history to estimate a trajectory for "
                + ", ".join(missing)
                + "; no prediction attempted",
                insufficient_history=True,
                entities_without_trajectory=missing,
                samples={
                    m.entity_id: m.samples for m in (primary, secondary) if m is not None
                },
            )

        # --- refusal 2: nothing is actually moving -------------------------
        if _both_stationary(context, primary, secondary):
            return self._inactive(
                context,
                f"both entities are stationary in {context.coordinate_space}; "
                "no trajectory risk",
                both_stationary=True,
                **{
                    thresholds.speed_key("primary_speed"): round(
                        _speed(context, primary), 3
                    ),
                    thresholds.speed_key("secondary_speed"): round(
                        _speed(context, secondary), 3
                    ),
                },
            )

        # --- refusal 3: already separating ---------------------------------
        if not geometry.is_converging:
            return self._score(
                context,
                0.0,
                f"closest approach is in the past; {primary.entity_id} and "
                f"{secondary.entity_id} are moving apart",
                confidence=min(primary.confidence, secondary.confidence),
                is_converging=False,
                **{
                    thresholds.speed_key("closing_speed"): round(
                        geometry.closing_speed, 3
                    )
                },
            )

        miss_distance = geometry.closest_approach_distance
        seconds_ahead = geometry.seconds_to_closest_approach

        spatial = linear_falloff(
            miss_distance,
            full_at=thresholds.conflict_radius,
            zero_at=thresholds.miss_radius,
        )
        temporal = linear_falloff(
            seconds_ahead,
            full_at=0.0,
            zero_at=config.prediction_horizon_seconds,
        )
        score = spatial * temporal

        space_label = "ground-plane" if context.is_world_space else "image-space"
        if spatial <= 0.0:
            rationale = (
                f"trajectories converge but miss by "
                f"{thresholds.format_distance(miss_distance)} (beyond the "
                f"{thresholds.format_distance(thresholds.miss_radius)} conflict window)"
            )
        elif temporal <= 0.0:
            rationale = (
                f"predicted conflict is {seconds_ahead:.1f}s away, beyond the "
                f"{config.prediction_horizon_seconds:.0f}s prediction horizon"
            )
        else:
            rationale = (
                f"{space_label} trajectories converge to "
                f"{thresholds.format_distance(miss_distance)} in {seconds_ahead:.1f}s"
            )

        confidence = min(primary.confidence, secondary.confidence)
        evidence = [e for e in primary.evidence_event_ids[-3:]]
        evidence += [e for e in secondary.evidence_event_ids[-3:]]

        return self._score(
            context,
            score,
            rationale,
            confidence=confidence,
            evidence_event_ids=evidence,
            seconds_to_closest_approach=round(seconds_ahead, 3),
            spatial_term=round(spatial, 4),
            temporal_term=round(temporal, 4),
            prediction_horizon_seconds=config.prediction_horizon_seconds,
            **{
                thresholds.distance_key("closest_approach_distance"): round(
                    miss_distance, 3
                ),
                thresholds.distance_key("conflict_radius"): thresholds.conflict_radius,
                thresholds.speed_key("primary_velocity"): [
                    round(v, 3) for v in _velocity(context, primary)
                ],
                thresholds.speed_key("secondary_velocity"): [
                    round(v, 3) for v in _velocity(context, secondary)
                ],
            },
        )

    # ------------------------------------------------------------------
    @staticmethod
    def predicted_time_to_incident(context: RiskContext) -> "float | None":
        """Seconds until the predicted conflict, or ``None`` if not predictable.

        Only returned when the closest approach is actually inside the conflict
        radius: "they will pass 300px apart in 2 seconds" is not a time to
        incident, it is a time to a near miss.
        """
        geometry = context.geometry()
        if geometry is None or not geometry.is_estimable or not geometry.is_converging:
            return None
        thresholds = context.thresholds
        if geometry.closest_approach_distance > thresholds.conflict_radius:
            return None
        if (
            geometry.seconds_to_closest_approach
            > context.config.prediction_horizon_seconds
        ):
            return None
        return geometry.seconds_to_closest_approach


def _velocity(context: RiskContext, motion):
    """Velocity in the active coordinate space."""
    if context.is_world_space and motion.world is not None:
        return motion.world.velocity_m_per_s
    return motion.velocity_px_per_s


def _speed(context: RiskContext, motion) -> float:
    if context.is_world_space and motion.world is not None:
        return motion.world.speed_m_per_s
    return motion.speed_px_per_s


def _both_stationary(context: RiskContext, primary, secondary) -> bool:
    if context.is_world_space and primary.world and secondary.world:
        return primary.world.is_stationary and secondary.world.is_stationary
    return primary.is_stationary and secondary.is_stationary
