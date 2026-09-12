"""Vehicle operating zone factor.

Answers: *is a person standing where a machine works?*

Unlike the geometric factors this one is not a prediction — it is a reading of
recorded fact. The temporal memory already knows which zones each entity
occupies, because the M0.2 event generator emitted ``entered_zone`` /
``exited_zone`` events for it. This factor consumes that state; it does not
re-derive geometry, and it does not need to.
"""

from __future__ import annotations

from typing import List, Optional

from ...events.schema import ACTION_ENTERED_ZONE
from ...spatial import euclidean_distance
from ..models.risk import FactorScore
from .base import RiskContext, RiskFactor, linear_falloff


class ZoneFactor(RiskFactor):
    """A person inside a configured vehicle/machine operating zone.

    Scoring
    -------
    ============================================================ =====
    Situation                                                    Score
    ============================================================ =====
    Person not in any operating zone                              0.00
    Person in an operating zone, no vehicle involved               0.60
    Person in an operating zone, vehicle elsewhere                 0.75
    Person in an operating zone, vehicle approaching or inside     1.00
    ============================================================ =====

    A person in a machine's operating zone is dangerous on its own — that is
    what the zone was drawn for — so the floor is high. It reaches 1.0 only
    when the machine is actually there or heading in, which is the difference
    between "should not be standing there" and "is about to be struck".

    The step between "vehicle elsewhere" and "vehicle approaching" is
    interpolated by the vehicle's distance to the zone occupant, so the score
    moves smoothly as the vehicle closes rather than jumping.
    """

    name = "zone"

    #: Score for a person in an operating zone with no vehicle in the candidate.
    SCORE_UNACCOMPANIED = 0.60
    #: Score with a vehicle present but far away.
    SCORE_VEHICLE_DISTANT = 0.75
    #: Score with the vehicle in or right on top of the zone.
    SCORE_VEHICLE_PRESENT = 1.00

    def evaluate(self, context: RiskContext) -> FactorScore:
        """Score the candidate and record who was standing in which zone.

        The occupancy map is attached to every result, including the inactive
        ones, so a downstream consumer never has to parse the rationale string
        to find out where an entity was.
        """
        score = self._evaluate(context)
        score.details.setdefault("zone_occupancy", self._occupancy(context))
        return score

    def _evaluate(self, context: RiskContext) -> FactorScore:
        config = context.config
        operating = {z.lower() for z in config.operating_zones}
        if not operating:
            return self._inactive(context, "no operating zones configured")

        person_id = self._person_id(context)
        if person_id is None:
            return self._inactive(
                context, "no person in this candidate; zone factor does not apply"
            )

        occupied = self._zones_of(context, person_id)
        breached = sorted(z for z in occupied if z.lower() in operating)
        if not breached:
            return self._score(
                context,
                0.0,
                f"{person_id} is not inside any operating zone",
                confidence=1.0,
                occupied_zones=sorted(occupied),
                operating_zones=sorted(operating),
            )

        evidence = self._entry_evidence(context, person_id, breached)
        vehicle_id = self._vehicle_id(context, person_id)

        if vehicle_id is None:
            return self._score(
                context,
                self.SCORE_UNACCOMPANIED,
                f"{person_id} is inside operating zone "
                f"{', '.join(breached)} (no vehicle in this candidate)",
                confidence=1.0,
                evidence_event_ids=evidence,
                breached_zones=breached,
                vehicle_present=False,
            )

        # Vehicle involved: interpolate between "distant" and "present" by how
        # close the vehicle is to the person standing in its zone.
        vehicle_zones = self._zones_of(context, vehicle_id)
        vehicle_in_zone = any(z in breached for z in vehicle_zones)

        thresholds = context.thresholds
        person_position = context.position_of(person_id)
        vehicle_position = context.position_of(vehicle_id)
        separation: Optional[float] = None
        if person_position is not None and vehicle_position is not None:
            separation = euclidean_distance(person_position, vehicle_position)

        if vehicle_in_zone:
            score = self.SCORE_VEHICLE_PRESENT
            detail = f"{vehicle_id} is in the same zone"
        elif separation is None:
            score = self.SCORE_VEHICLE_DISTANT
            detail = f"{vehicle_id} position unknown"
        else:
            nearness = linear_falloff(
                separation,
                full_at=thresholds.critical_radius,
                zero_at=thresholds.interaction_radius,
            )
            score = self.SCORE_VEHICLE_DISTANT + nearness * (
                self.SCORE_VEHICLE_PRESENT - self.SCORE_VEHICLE_DISTANT
            )
            detail = f"{vehicle_id} is {thresholds.format_distance(separation)} away"

        extra = {
            thresholds.distance_key("vehicle_separation"): (
                round(separation, 3) if separation is not None else None
            )
        }
        return self._score(
            context,
            score,
            f"{person_id} is inside operating zone {', '.join(breached)}; {detail}",
            confidence=1.0,
            evidence_event_ids=evidence,
            breached_zones=breached,
            vehicle_present=True,
            vehicle_in_zone=vehicle_in_zone,
            **extra,
        )

    # ------------------------------------------------------------------
    def _occupancy(self, context: RiskContext) -> dict:
        """Which zones each entity in this candidate occupies, from memory."""
        return {
            entity_id: self._zones_of(context, entity_id)
            for entity_id in context.candidate.entity_ids
        }

    def _person_id(self, context: RiskContext) -> Optional[str]:
        for entity_id in context.candidate.entity_ids:
            if context.config.is_person(context.class_of(entity_id)):
                return entity_id
        return None

    def _vehicle_id(self, context: RiskContext, person_id: str) -> Optional[str]:
        for entity_id in context.candidate.entity_ids:
            if entity_id == person_id:
                continue
            if context.config.is_vehicle(context.class_of(entity_id)):
                return entity_id
        return None

    def _zones_of(self, context: RiskContext, entity_id: str) -> List[str]:
        state = context.state(entity_id)
        return list(state.zones) if state else []

    def _entry_evidence(
        self, context: RiskContext, person_id: str, breached: List[str]
    ) -> List[str]:
        """The ``entered_zone`` events that put this person in the zone."""
        events = context.history(person_id, actions=[ACTION_ENTERED_ZONE])
        return [
            e.event_id
            for e in events
            if e.event_id and e.attributes.get("zone") in breached
        ]
