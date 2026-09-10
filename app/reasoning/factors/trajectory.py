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

Everything here is image-space. A "closest approach of 40px" is 40 pixels in
the frame, not 40 centimetres on the floor.
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
        1.0 when the predicted miss distance is at or inside
        ``conflict_radius_px``, falling to 0.0 at ``trajectory_miss_radius_px``.

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
        approach = context.approach()
        if approach is None:
            return self._inactive(
                context, "no pair to measure: trajectory needs two observed entities"
            )

        primary = context.primary_motion
        secondary = context.secondary_motion
        config = context.config

        # --- refusal 1: not enough history to have a heading at all --------
        if not approach.is_estimable:
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
        if primary.is_stationary and secondary.is_stationary:
            return self._inactive(
                context,
                "both entities are stationary in image space; "
                "no trajectory risk",
                both_stationary=True,
                primary_speed_px_per_s=round(primary.speed_px_per_s, 2),
                secondary_speed_px_per_s=round(secondary.speed_px_per_s, 2),
            )

        # --- refusal 3: already separating ---------------------------------
        if not approach.is_converging:
            return self._score(
                context,
                0.0,
                f"closest approach is in the past; {primary.entity_id} and "
                f"{secondary.entity_id} are moving apart",
                confidence=min(primary.confidence, secondary.confidence),
                closing_speed_px_per_s=round(approach.closing_speed_px_per_s, 2),
                is_converging=False,
            )

        miss_distance = approach.distance_px
        seconds_ahead = approach.seconds_to_closest_approach

        spatial = linear_falloff(
            miss_distance,
            full_at=config.conflict_radius_px,
            zero_at=config.trajectory_miss_radius_px,
        )
        temporal = linear_falloff(
            seconds_ahead,
            full_at=0.0,
            zero_at=config.prediction_horizon_seconds,
        )
        score = spatial * temporal

        if spatial <= 0.0:
            rationale = (
                f"trajectories converge but miss by {miss_distance:.0f}px "
                f"(beyond the {config.trajectory_miss_radius_px:.0f}px conflict window)"
            )
        elif temporal <= 0.0:
            rationale = (
                f"predicted conflict is {seconds_ahead:.1f}s away, beyond the "
                f"{config.prediction_horizon_seconds:.0f}s prediction horizon"
            )
        else:
            rationale = (
                f"image-space trajectories converge to {miss_distance:.0f}px "
                f"in {seconds_ahead:.1f}s"
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
            closest_approach_distance_px=round(miss_distance, 2),
            seconds_to_closest_approach=round(seconds_ahead, 3),
            spatial_term=round(spatial, 4),
            temporal_term=round(temporal, 4),
            conflict_radius_px=config.conflict_radius_px,
            prediction_horizon_seconds=config.prediction_horizon_seconds,
            primary_velocity_px_per_s=[round(v, 2) for v in primary.velocity_px_per_s],
            secondary_velocity_px_per_s=[
                round(v, 2) for v in secondary.velocity_px_per_s
            ],
        )

    # ------------------------------------------------------------------
    @staticmethod
    def predicted_time_to_incident(context: RiskContext) -> "float | None":
        """Seconds until the predicted conflict, or ``None`` if not predictable.

        Only returned when the closest approach is actually inside the conflict
        radius: "they will pass 300px apart in 2 seconds" is not a time to
        incident, it is a time to a near miss.
        """
        approach = context.approach()
        if approach is None or not approach.is_estimable or not approach.is_converging:
            return None
        if approach.distance_px > context.config.conflict_radius_px:
            return None
        if approach.seconds_to_closest_approach > context.config.prediction_horizon_seconds:
            return None
        return approach.seconds_to_closest_approach
