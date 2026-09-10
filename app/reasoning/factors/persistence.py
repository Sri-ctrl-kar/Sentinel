"""Persistence and escalation factors.

Two different kinds of "it's not one thing, it's a pattern":

``PersistenceFactor``
    The same entity breaking the same rule repeatedly *over time*. One person
    stepping into a restricted zone is a mistake; the same person doing it
    four times in a minute is a behaviour, and it deserves a higher score than
    any single entry does.

``EscalationFactor``
    Several *different* weak signals about the same situation *at the same
    time*. Nothing individually alarming, everything pointing the same way.
    This is a meta-factor: it reads the other factors' scores rather than the
    event stream, so it runs after them.
"""

from __future__ import annotations

from typing import List, Sequence

from ...events.schema import ACTION_ENTERED_ZONE
from ..models.risk import FactorScore
from .base import MetaRiskFactor, RiskContext, RiskFactor


class PersistenceFactor(RiskFactor):
    """Repeated violations of a configured spatial rule by one entity.

    Scoring
    -------
    Counts this entity's entries into operating or restricted zones within
    ``persistence_window_seconds``::

        score = clamp01(violations / persistence_violation_threshold)

    The first entry contributes nothing beyond what the zone factor already
    scored — persistence is explicitly about *repetition*, so a single entry
    scores ``1/3`` at the default threshold and only sustained repetition
    saturates it.

    Counted from ``entered_zone`` events, which the M0.2 event generator
    already emits, so this factor is a query rather than a computation.
    """

    name = "persistence"

    def evaluate(self, context: RiskContext) -> FactorScore:
        config = context.config
        watched = {z.lower() for z in config.operating_zones}
        watched |= {z.lower() for z in config.restricted_zones}
        if not watched:
            return self._inactive(context, "no operating or restricted zones configured")

        window_start = context.at - config.persistence_window_seconds
        violations: List = []
        for entity_id in context.candidate.entity_ids:
            if not config.is_person(context.class_of(entity_id)):
                continue
            for event in context.history(entity_id, actions=[ACTION_ENTERED_ZONE]):
                if event.timestamp < window_start:
                    continue
                zone = (event.attributes.get("zone") or "").lower()
                if zone in watched:
                    violations.append(event)

        if not violations:
            return self._score(
                context,
                0.0,
                "no repeated zone violations in the persistence window",
                confidence=1.0,
                violation_count=0,
            )

        threshold = max(1, config.persistence_violation_threshold)
        score = min(1.0, len(violations) / float(threshold))
        entities = sorted({e.entity_id for e in violations})
        zones = sorted({e.attributes.get("zone") for e in violations if e.attributes.get("zone")})

        if len(violations) == 1:
            rationale = (
                f"{entities[0]} entered {zones[0]} once in the last "
                f"{config.persistence_window_seconds:.0f}s"
            )
        else:
            rationale = (
                f"{', '.join(entities)} entered {', '.join(zones)} "
                f"{len(violations)} times in the last "
                f"{config.persistence_window_seconds:.0f}s "
                f"(saturates at {threshold})"
            )

        return self._score(
            context,
            score,
            rationale,
            confidence=1.0,
            evidence_event_ids=[e.event_id for e in violations if e.event_id],
            violation_count=len(violations),
            violation_threshold=threshold,
            violated_zones=zones,
            window_seconds=config.persistence_window_seconds,
        )


class EscalationFactor(MetaRiskFactor):
    """Corroboration across independent factors.

    This is the factor that turns Scenario E — several individually
    unremarkable readings — into a situation worth acting on.

    Scoring
    -------
    Counts how many base factors scored at or above
    ``escalation_signal_threshold`` (default 0.20), then looks the count up in
    ``escalation_multipliers``::

        signals  0     1     2     3     4     5+
        mult     1.00  1.00  1.10  1.20  1.30  1.35

    Unlike the other factors this one contributes no points of its own. It
    returns a *multiplier* applied to the subtotal, reported in
    ``details['multiplier']``. That distinction matters: corroboration can
    promote a borderline situation across a severity band, but it cannot
    manufacture risk out of nothing — a subtotal of 0 stays 0 no matter how
    many factors agree that nothing is happening.
    """

    name = "escalation"

    def weight(self, config) -> float:  # noqa: ANN001 - matches base signature
        # Contributes a multiplier, not points.
        return 0.0

    def evaluate_with(
        self, context: RiskContext, scores: Sequence[FactorScore]
    ) -> FactorScore:
        config = context.config
        threshold = config.escalation_signal_threshold
        signals = [s for s in scores if s.name != self.name and s.score >= threshold]
        multiplier = config.escalation_multiplier(len(signals))

        names = [s.name for s in signals]
        if len(signals) < 2:
            rationale = (
                f"{len(signals)} corroborating signal(s); no escalation applied"
            )
        else:
            rationale = (
                f"{len(signals)} independent signals agree ({', '.join(names)}); "
                f"escalating by ×{multiplier:.2f}"
            )

        evidence: List[str] = []
        for signal in signals:
            evidence.extend(signal.evidence_event_ids)

        return self._score(
            context,
            # `score` here reports how corroborated the situation is, on the
            # same 0-1 scale as everything else, purely for inspection.
            score=min(1.0, len(signals) / 4.0),
            rationale=rationale,
            confidence=1.0,
            evidence_event_ids=_dedupe(evidence),
            multiplier=multiplier,
            signal_count=len(signals),
            signal_names=names,
            signal_threshold=threshold,
        )


def _dedupe(values: Sequence[str]) -> List[str]:
    seen = {}
    for value in values:
        seen.setdefault(value, None)
    return list(seen)
