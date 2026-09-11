"""The predictive risk engine.

Consumes temporal event memory and produces scored, explained assessments of
developing situations. This is the first Sentinel component that *reasons*
rather than records.

Layering
--------
The engine reads :class:`~app.events.schema.Event` objects and
:class:`~app.memory.temporal.TemporalEventMemory`. It has no knowledge of
YOLO, torch, Ultralytics, OpenCV, video, frames, detections or tracks — the
boundary is enforced by ``tests/test_memory_layering.py``, not by convention.
Swapping the detector cannot change a single line here.

Scoring model
-------------
Deterministic and fully inspectable. For each candidate situation::

    base_points   = Σ over {proximity, closing_speed, trajectory, zone}
                        factor.score × factor.weight          →  0 .. 100
    persistence   = persistence.score × 15                     →  0 ..  15
    subtotal      = base_points + persistence                  →  0 .. 115
    multiplier    = escalation multiplier by signal count      →  1.00 .. 1.35
    risk_score    = clamp(0, 100, subtotal × multiplier)

Every term is recoverable from the output: each ``FactorScore`` carries its
own score, weight, contribution, confidence, rationale, the raw measurements
it used, and the event IDs that evidence it. Re-running the engine on the same
memory at the same timestamp always produces byte-identical output — there is
no randomness, no learned model and no wall-clock dependence.

Units
-----
**Each assessment states the space it was scored in.** With a
:class:`~app.calibration.planar.GroundPlaneCalibration` supplying world
positions for both entities, scoring uses ground-plane metres and the metric
threshold set. Without one — or when an individual pair cannot be projected —
scoring uses image pixels and the pixel threshold set, exactly as in M0.3.

The two threshold sets are configured independently and neither is derived
from the other: no pixels-per-metre ratio exists outside a specific
calibration, so converting between them would be meaningless. A pair is scored
wholly in one space; values from the two are never mixed. Where a candidate
falls back, ``space_fallback_reason`` says why, so the change of space is
visible rather than silent.

**A risk score is not a probability.** It is an ordinal 0-100 severity ranking
produced by a documented formula. Nothing here is calibrated against incident
frequencies, and no field should be read as a likelihood.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..calibration.planar import GroundPlaneCalibration
from ..events.schema import Event
from ..memory.temporal import TemporalEventMemory
from ..spatial import GROUND_PLANE_METERS, IMAGE_PIXELS
from .config import RiskConfig
from .factors import (
    ClosingSpeedFactor,
    EscalationFactor,
    PersistenceFactor,
    ProximityFactor,
    RiskCandidate,
    RiskContext,
    TrajectoryFactor,
    ZoneFactor,
)
from .factors.base import CANDIDATE_ENTITY, CANDIDATE_PAIR
from .kinematics import (
    ImageMotion,
    MotionEstimator,
    ground_closest_approach,
    ground_separation,
)
from .prediction import (
    PREDICTED_TRAJECTORY_CONFLICT,
    PREDICTED_UNSAFE_PROXIMITY,
    PREDICTION_UNAVAILABLE,
    REASON_INSUFFICIENT_HISTORY,
    PairPrediction,
    TrajectoryPredictor,
)
from .models.risk import (
    INCIDENT_CLOSE_PROXIMITY,
    INCIDENT_CONVERGING_TRAJECTORIES,
    INCIDENT_ESCALATING_SITUATION,
    INCIDENT_NONE,
    INCIDENT_PERSISTENT_ZONE_VIOLATION,
    INCIDENT_PERSON_IN_OPERATING_ZONE,
    INCIDENT_PERSON_VEHICLE_COLLISION,
    INCIDENT_RAPID_SEPARATION_DECREASE,
    FactorScore,
    RiskAssessment,
    RiskReport,
    TimeToRisk,
    severity_for,
)

#: Why a candidate was scored in image space despite a calibration existing.
FALLBACK_NO_WORLD_POSITION = "no_world_position_for_one_or_both_entities"
FALLBACK_NO_CALIBRATION = "no_calibration_supplied"

#: Recommended interventions per incident type. Deliberately a static lookup:
#: at M0.3 the engine must not improvise advice, and a reviewer must be able to
#: see exactly what will be recommended for a given incident type.
INTERVENTIONS: Dict[str, str] = {
    INCIDENT_PERSON_VEHICLE_COLLISION: (
        "slow/stop the vehicle and redirect the person out of its path"
    ),
    INCIDENT_PERSON_IN_OPERATING_ZONE: (
        "halt machine operation in this zone until the person has cleared it"
    ),
    INCIDENT_CONVERGING_TRAJECTORIES: (
        "alert both parties; hold the vehicle until paths diverge"
    ),
    INCIDENT_RAPID_SEPARATION_DECREASE: (
        "reduce vehicle speed and confirm the operator has seen the person"
    ),
    INCIDENT_PERSISTENT_ZONE_VIOLATION: (
        "review access control and brief the individual on the zone boundary"
    ),
    INCIDENT_ESCALATING_SITUATION: (
        "review the area: several independent indicators are trending together"
    ),
    INCIDENT_CLOSE_PROXIMITY: (
        "monitor only: entities are near each other but nothing is developing"
    ),
    INCIDENT_NONE: "no action required",
}


class RiskEngine:
    """Scores developing situations from temporal event memory.

    Usage::

        engine = RiskEngine(RiskConfig(operating_zones=["forklift_bay"], zones=zones))
        report = engine.assess(memory, at=12.5)
        print(report.top.risk_score, report.top.incident_type)
    """

    def __init__(
        self,
        config: Optional[RiskConfig] = None,
        motion_estimator: Optional[MotionEstimator] = None,
        factors: Optional[Sequence[Any]] = None,
        meta_factors: Optional[Sequence[Any]] = None,
        calibration: Optional[GroundPlaneCalibration] = None,
        predictor: Optional[TrajectoryPredictor] = None,
    ) -> None:
        self.config = config or RiskConfig()
        self.calibration = calibration
        self.motion_estimator = motion_estimator or MotionEstimator(
            calibration=calibration
        )
        self.predictor = predictor or TrajectoryPredictor(
            horizon_seconds=self.config.prediction_horizon_seconds
        )
        # Base factors contribute points; meta factors read their results.
        self.factors = list(
            factors
            if factors is not None
            else [
                ProximityFactor(),
                ClosingSpeedFactor(),
                TrajectoryFactor(),
                ZoneFactor(),
                PersistenceFactor(),
            ]
        )
        self.meta_factors = list(
            meta_factors if meta_factors is not None else [EscalationFactor()]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def assess(
        self, memory: TemporalEventMemory, at: Optional[float] = None
    ) -> RiskReport:
        """Assess every candidate situation as of time ``at``.

        ``at`` defaults to the last timestamp in memory. Assessments are
        sorted by risk score, highest first.
        """
        if at is None:
            _first, last = memory.span
            at = last if last is not None else 0.0

        motions = self._motions(memory, at)
        assessments: List[RiskAssessment] = []

        for candidate in self._candidates(memory, at, motions):
            assessment = self._assess_candidate(memory, at, candidate, motions)
            if assessment.risk_score >= self.config.report_threshold:
                assessments.append(assessment)

        assessments.sort(key=lambda a: (-a.risk_score, a.involved_entity_ids))
        return RiskReport(
            timestamp=at,
            assessments=assessments,
            coordinate_space=self.configured_space,
            metadata={
                "config": self.config.to_dict(),
                "entities_considered": sorted(motions),
                "candidates_evaluated": len(
                    self._candidates(memory, at, motions)
                ),
                "configured_coordinate_space": self.configured_space,
                "scoring_coordinate_space": self.configured_space,
                "spaces_used": _count_spaces(assessments),
                "score_interpretation": (
                    "0-100 ordinal risk score; NOT a calibrated probability"
                ),
                "calibration": (
                    self.calibration.metadata() if self.calibration else None
                ),
            },
        )

    def assess_timeline(
        self,
        memory: TemporalEventMemory,
        start: Optional[float] = None,
        end: Optional[float] = None,
        step: float = 0.5,
    ) -> List[RiskReport]:
        """Assess repeatedly across a time range, for reviewing a whole clip."""
        first, last = memory.span
        if first is None or last is None:
            return []
        start = first if start is None else start
        end = last if end is None else end
        if step <= 0:
            raise ValueError("step must be > 0")

        reports = []
        moment = start
        while moment <= end + 1e-9:
            reports.append(self.assess(memory, at=round(moment, 4)))
            moment += step
        return reports

    def score(self, events: Iterable[Event]) -> RiskReport:
        """Assess a plain sequence of events (M0 compatibility entry point)."""
        return self.assess(TemporalEventMemory.from_events(events))

    @property
    def configured_space(self) -> str:
        """The space this engine will score in where the data allows."""
        return GROUND_PLANE_METERS if self.calibration else IMAGE_PIXELS

    def _resolve_space(
        self, candidate: RiskCandidate, motions: Dict[str, ImageMotion]
    ) -> Tuple[str, Optional[str]]:
        """Which space to score this candidate in, and why if it is not the default.

        World space needs a world *position* for every entity involved — not a
        velocity, which mirrors how image space already lets proximity work
        while trajectory refuses. A pair that cannot be projected is scored in
        pixels with the reason recorded, never in a mixture of the two.
        """
        if self.calibration is None:
            return IMAGE_PIXELS, None
        for entity_id in candidate.entity_ids:
            motion = motions.get(entity_id)
            if motion is None or motion.world is None:
                return IMAGE_PIXELS, FALLBACK_NO_WORLD_POSITION
        return GROUND_PLANE_METERS, None

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------
    def _motions(
        self, memory: TemporalEventMemory, at: float
    ) -> Dict[str, ImageMotion]:
        """Image-space motion for every entity observable at ``at``."""
        motions: Dict[str, ImageMotion] = {}
        for entity_id in memory.entities():
            history = [
                e for e in memory.entity_history(entity_id) if e.timestamp <= at
            ]
            if not history:
                continue
            state = memory.state_at(entity_id, at)
            if state is None or not state.present:
                continue
            motion = self.motion_estimator.estimate(history, at, entity_id)
            if motion is not None:
                motions[entity_id] = motion
        return motions

    def _candidates(
        self,
        memory: TemporalEventMemory,
        at: float,
        motions: Dict[str, ImageMotion],
    ) -> List[RiskCandidate]:
        """Which situations are worth scoring.

        Person/vehicle pairs are the primary case. A person on their own is
        also considered, so that repeated zone violations are caught even when
        no vehicle is present.
        """
        people = [
            eid for eid, m in motions.items() if self.config.is_person(m.class_name)
        ]
        vehicles = [
            eid for eid, m in motions.items() if self.config.is_vehicle(m.class_name)
        ]

        candidates: List[RiskCandidate] = []
        for person in sorted(people):
            for vehicle in sorted(vehicles):
                candidates.append(
                    RiskCandidate(
                        entity_ids=[person, vehicle],
                        primary=person,
                        secondary=vehicle,
                        kind=CANDIDATE_PAIR,
                    )
                )
            candidates.append(
                RiskCandidate(
                    entity_ids=[person],
                    primary=person,
                    secondary=None,
                    kind=CANDIDATE_ENTITY,
                )
            )
        return candidates

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _assess_candidate(
        self,
        memory: TemporalEventMemory,
        at: float,
        candidate: RiskCandidate,
        motions: Dict[str, ImageMotion],
    ) -> RiskAssessment:
        space, fallback_reason = self._resolve_space(candidate, motions)
        thresholds = self.config.thresholds(space)
        prediction = self._predict(candidate, motions, space, thresholds)

        context = RiskContext(
            memory=memory,
            at=at,
            config=self.config,
            candidate=candidate,
            motions=motions,
            coordinate_space=space,
            space_fallback_reason=fallback_reason,
            prediction=prediction,
        )

        scores = [factor.evaluate(context) for factor in self.factors]
        meta_scores = [
            factor.evaluate_with(context, scores) for factor in self.meta_factors
        ]

        subtotal = sum(score.contribution for score in scores)
        multiplier = 1.0
        for meta in meta_scores:
            multiplier *= float(meta.details.get("multiplier", 1.0))

        risk_score = max(0.0, min(100.0, subtotal * multiplier))
        all_scores = scores + meta_scores

        incident_type = self._incident_type(risk_score, scores, meta_scores)
        ground = self._ground_plane_details(candidate, motions)
        return RiskAssessment(
            risk_score=round(risk_score, 2),
            severity=severity_for(risk_score),
            incident_type=incident_type,
            involved_entity_ids=list(candidate.entity_ids),
            timestamp=at,
            confidence=self._confidence(scores),
            predicted_time_to_incident_seconds=TrajectoryFactor.predicted_time_to_incident(
                context
            ),
            prediction_outcome=prediction.outcome if prediction else None,
            time_to_risk=self._time_to_risk(prediction, thresholds, space),
            space_fallback_reason=fallback_reason,
            contributing_factors=all_scores,
            evidence_event_ids=_dedupe(
                eid for score in all_scores for eid in score.evidence_event_ids
            ),
            recommended_intervention=INTERVENTIONS.get(incident_type, ""),
            escalation_multiplier=multiplier,
            coordinate_space=space,
            details={
                "candidate_kind": candidate.kind,
                "coordinate_space": space,
                "units": thresholds.distance_label,
                "thresholds": thresholds.to_dict(),
                **(
                    {"prediction": prediction.to_dict()}
                    if prediction is not None
                    else {}
                ),
                "subtotal_points": round(subtotal, 4),
                "base_points": round(
                    sum(s.contribution for s in scores if s.name != "persistence"), 4
                ),
                "persistence_points": round(
                    sum(s.contribution for s in scores if s.name == "persistence"), 4
                ),
                "motions": {
                    eid: motions[eid].to_dict()
                    for eid in candidate.entity_ids
                    if eid in motions
                },
                **({"ground_plane": ground} if ground else {}),
            },
        )

    # ------------------------------------------------------------------
    # Prediction and time-to-risk
    # ------------------------------------------------------------------
    def _predict(
        self,
        candidate: RiskCandidate,
        motions: Dict[str, ImageMotion],
        space: str,
        thresholds: Any,
    ) -> Optional[PairPrediction]:
        """Constant-velocity prediction for a pair, in the active space."""
        if not candidate.is_pair:
            return None
        primary = motions.get(candidate.primary)
        secondary = motions.get(candidate.secondary or "")
        if primary is None or secondary is None:
            return None
        return self.predictor.predict_pair(
            primary,
            secondary,
            coordinate_space=space,
            unsafe_separation=thresholds.unsafe_separation,
            conflict_radius=thresholds.conflict_radius,
        )

    def _time_to_risk(
        self, prediction: Optional[PairPrediction], thresholds: Any, space: str
    ) -> Optional[TimeToRisk]:
        """When this pair is expected to become unsafely close.

        Distinguishes "already unsafe" from "predicted to become unsafe" from
        "no prediction supported", and never invents a number for the last.
        """
        units = "meters" if space == GROUND_PLANE_METERS else "pixels"
        if prediction is None:
            return None

        if prediction.outcome == PREDICTION_UNAVAILABLE:
            return TimeToRisk(
                status=TimeToRisk.STATUS_NOT_PREDICTED,
                seconds=None,
                reason=prediction.unavailable_reason or REASON_INSUFFICIENT_HISTORY,
                threshold=thresholds.unsafe_separation,
                coordinate_space=space,
                units=units,
            )

        if (
            prediction.current_separation is not None
            and prediction.current_separation <= thresholds.unsafe_separation
        ):
            return TimeToRisk(
                status=TimeToRisk.STATUS_ALREADY_UNSAFE,
                seconds=0.0,
                threshold=thresholds.unsafe_separation,
                coordinate_space=space,
                units=units,
            )

        if prediction.seconds_to_unsafe_separation is not None:
            return TimeToRisk(
                status=TimeToRisk.STATUS_PREDICTED,
                seconds=prediction.seconds_to_unsafe_separation,
                threshold=thresholds.unsafe_separation,
                coordinate_space=space,
                units=units,
            )

        return TimeToRisk(
            status=TimeToRisk.STATUS_NOT_PREDICTED,
            seconds=None,
            reason="threshold not reached within the prediction horizon",
            threshold=thresholds.unsafe_separation,
            coordinate_space=space,
            units=units,
        )

    def _ground_plane_details(
        self, candidate: RiskCandidate, motions: Dict[str, ImageMotion]
    ) -> Optional[Dict[str, Any]]:
        """Metric measurements for this candidate, when a calibration exists.

        Reported alongside the score, never fed into it. A reader gets "these
        two are 2.4 m apart" as well as the pixel-based verdict, and the two
        cannot be confused because every key names its unit.
        """
        if self.calibration is None or not candidate.is_pair:
            return None
        primary = motions.get(candidate.primary)
        secondary = motions.get(candidate.secondary or "")
        if primary is None or secondary is None:
            return None
        if primary.world is None or secondary.world is None:
            return None

        details: Dict[str, Any] = {
            "coordinate_space": primary.world.coordinate_space,
            "units": primary.world.units,
            "note": "reported only; risk scoring remains image-space at M0.4",
            "positions_m": {
                candidate.primary: [round(v, 3) for v in primary.world.position_m],
                candidate.secondary: [
                    round(v, 3) for v in secondary.world.position_m
                ],
            },
            "speeds_m_per_s": {
                candidate.primary: round(primary.world.speed_m_per_s, 3),
                candidate.secondary: round(secondary.world.speed_m_per_s, 3),
            },
            "in_calibrated_region": (
                primary.world.in_calibrated_region
                and secondary.world.in_calibrated_region
            ),
        }
        separation = ground_separation(primary, secondary)
        if separation is not None:
            details["separation_m"] = round(separation, 3)
        approach = ground_closest_approach(primary, secondary)
        if approach is not None:
            details["closest_approach"] = approach.to_dict()
        return details

    def _confidence(self, scores: Sequence[FactorScore]) -> float:
        """Confidence in the assessment, in ``[0, 1]``.

        A contribution-weighted mean of the confidences of the factors that
        actually scored. Factors that scored zero are excluded — their
        confidence says nothing about a risk they did not report. With no
        contributions at all, confidence is 0: the engine is not confident
        that nothing is happening, it simply has nothing to say.
        """
        active = [s for s in scores if s.contribution > 0]
        if not active:
            return 0.0
        total = sum(s.contribution for s in active)
        if total <= 0:
            return 0.0
        return round(
            sum(s.confidence * s.contribution for s in active) / total, 4
        )

    def _incident_type(
        self,
        risk_score: float,
        scores: Sequence[FactorScore],
        meta_scores: Sequence[FactorScore],
    ) -> str:
        """Classify the situation from which factors drove the score.

        Deterministic priority order, most specific first. The rule is: name
        the situation by its most *actionable* driver, not simply its largest
        numeric contributor — a person standing in a machine's path is a zone
        incident even when proximity happens to contribute more points.
        """
        if risk_score < self.config.report_threshold:
            return INCIDENT_NONE

        by_name = {s.name: s for s in scores}
        zone = by_name.get("zone")
        trajectory = by_name.get("trajectory")
        closing = by_name.get("closing_speed")
        persistence = by_name.get("persistence")
        escalation = meta_scores[0] if meta_scores else None

        zone_active = zone is not None and zone.score > 0
        trajectory_active = trajectory is not None and trajectory.score > 0
        closing_active = closing is not None and closing.score > 0

        # A person in a machine's operating zone with that machine converging
        # is the canonical collision case.
        if zone_active and (trajectory_active or closing_active):
            return INCIDENT_PERSON_VEHICLE_COLLISION
        if zone_active:
            if persistence is not None and persistence.score >= 1.0:
                return INCIDENT_PERSISTENT_ZONE_VIOLATION
            return INCIDENT_PERSON_IN_OPERATING_ZONE
        if trajectory_active:
            # "Collision risk" is reserved for paths that actually conflict.
            # Converging on lines that miss by more than the conflict radius
            # is a near miss, and calling it a collision would cry wolf.
            miss_px = trajectory.details.get("closest_approach_distance_px")
            if miss_px is not None and miss_px <= self.config.conflict_radius_px:
                return INCIDENT_PERSON_VEHICLE_COLLISION
            return INCIDENT_CONVERGING_TRAJECTORIES
        if closing_active:
            return INCIDENT_RAPID_SEPARATION_DECREASE
        if persistence is not None and persistence.score > 0:
            return INCIDENT_PERSISTENT_ZONE_VIOLATION
        if escalation is not None and escalation.details.get("signal_count", 0) >= 2:
            return INCIDENT_ESCALATING_SITUATION

        proximity = by_name.get("proximity")
        if proximity is not None and proximity.score > 0:
            # Near each other, but nothing moving toward anything. Reported so
            # the situation is visible, named so nobody reads it as a
            # prediction.
            return INCIDENT_CLOSE_PROXIMITY
        return INCIDENT_NONE


def _count_spaces(assessments: Sequence[RiskAssessment]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for assessment in assessments:
        counts[assessment.coordinate_space] = (
            counts.get(assessment.coordinate_space, 0) + 1
        )
    return counts


def _dedupe(values: Iterable[str]) -> List[str]:
    seen: Dict[str, None] = {}
    for value in values:
        if value:
            seen.setdefault(value, None)
    return list(seen)
