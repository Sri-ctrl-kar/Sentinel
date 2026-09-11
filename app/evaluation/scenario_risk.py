"""Evaluation harness for the predictive scenarios.

Runs a scenario end to end and reports what Sentinel decided, how early it
decided it, and whether that decision was borne out by the scenario's own data.
This is the basis for the benchmark, so its definitions matter more than its
convenience.

What ``prediction_lead_time_seconds`` is
----------------------------------------
The gap between the first moment Sentinel makes a **valid forward-looking
prediction** of an unsafe approach and the moment the entities *actually* reach
unsafe separation in the recorded data::

    prediction_lead_time_seconds = first_unsafe_time - first_valid_prediction_time

A prediction is *valid* only when all of these hold (M0.6):

* the outcome is forward-looking — ``PREDICTED_UNSAFE_PROXIMITY`` or
  ``PREDICTED_TRAJECTORY_CONFLICT``. ``CURRENTLY_UNSAFE_PROXIMITY`` describes
  the present and is explicitly excluded;
* the predictor had sufficient history (the prediction is available at all);
* it was made **strictly before** the actual unsafe transition.

Earlier revisions measured from the first *severity alert* instead, which would
have credited the system for "predicting" an unsafe state after it had already
begun. That was an evaluation bug and is fixed here.

What it is **not**:

* It is **not accuracy**. It says nothing about whether the alert was correct,
  only when it fired relative to an event in the same synthetic data.
* It is **not validated against real incidents.** These scenarios are scripted
  constant-velocity motion. Real people change direction.
* It is **not a probability or a confidence.** Nothing here is calibrated
  against incident frequencies.

``first_unsafe_time`` is measured from observed separation in the scoring
space, not from the predictor — otherwise the metric would be grading the
prediction against itself.

It records an **onset**: a transition from safe to unsafe separation. A pair
that is already unsafely close in its first frame and never transitions has
nothing to predict, and is reported as ``no_unsafe_transition`` rather than as
a missed detection. Counting it as a miss would be measuring the wrong thing —
a predictive metric grades predicted transitions, not static closeness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..calibration.planar import GroundPlaneCalibration
from ..memory.temporal import TemporalEventMemory
from ..reasoning.config import RiskConfig
from ..reasoning.kinematics import MotionEstimator, separation_geometry
from ..reasoning.models.risk import SEVERITIES, RiskAssessment
from ..reasoning.prediction import (
    CURRENTLY_UNSAFE_PROXIMITY,
    PREDICTED_TRAJECTORY_CONFLICT,
    PREDICTED_UNSAFE_PROXIMITY,
)
from ..reasoning.risk_engine import RiskEngine
from ..scenarios import ALL_WORLD_SCENARIOS, WorldScenario

#: Severity at or above which the harness counts a situation as "alerted".
DEFAULT_ALERT_SEVERITY = "high"

# --- outcome classification ------------------------------------------------
OUTCOME_TRUE_POSITIVE = "true_positive"
OUTCOME_POSSIBLE_FALSE_POSITIVE = "possible_false_positive"
OUTCOME_MISSED = "missed_detection"
OUTCOME_TRUE_NEGATIVE = "true_negative"
#: The pair was already unsafely close at the first observation and never
#: transitioned, so no prediction was possible either way.
OUTCOME_NO_TRANSITION = "no_unsafe_transition"

#: Forward-looking prediction outcomes. A present-tense observation of unsafe
#: proximity is NOT one of these: predicting something that has already begun
#: is not prediction.
PREDICTIVE_OUTCOMES = (PREDICTED_UNSAFE_PROXIMITY, PREDICTED_TRAJECTORY_CONFLICT)


@dataclass
class ScenarioResult:
    """One scenario's evaluation."""

    scenario: str
    description: str
    risk_score: float
    severity: str
    coordinate_space: str
    units: str
    entities: List[str]
    predicted_conflict: bool
    prediction_outcome: Optional[str]
    time_to_risk_status: Optional[str]
    time_to_risk_seconds: Optional[float]
    current_separation: Optional[float] = None
    minimum_separation: Optional[float] = None
    closing_speed: Optional[float] = None
    first_alert_time: Optional[float] = None
    #: First moment a forward-looking unsafe prediction was made.
    first_valid_prediction_time: Optional[float] = None
    first_unsafe_time: Optional[float] = None
    prediction_lead_time_seconds: Optional[float] = None
    #: Set when a prediction existed but only after the transition had begun.
    late_prediction: bool = False
    outcome_status: str = OUTCOME_TRUE_NEGATIVE
    unsafe_at_start: bool = False
    space_fallback_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        distance_suffix = "m" if self.units == "meters" else "px"
        speed_suffix = "m_per_s" if self.units == "meters" else "px_per_s"
        return {
            "scenario": self.scenario,
            "description": self.description,
            "risk_score": round(self.risk_score, 2),
            "severity": self.severity,
            "coordinate_space": self.coordinate_space,
            "units": self.units,
            "entities": list(self.entities),
            "predicted_conflict": self.predicted_conflict,
            "prediction_outcome": self.prediction_outcome,
            "time_to_risk_status": self.time_to_risk_status,
            "time_to_risk_seconds": _round(self.time_to_risk_seconds),
            f"current_separation_{distance_suffix}": _round(self.current_separation),
            f"minimum_separation_{distance_suffix}": _round(self.minimum_separation),
            f"closing_speed_{speed_suffix}": _round(self.closing_speed),
            "first_alert_time": _round(self.first_alert_time),
            "first_valid_prediction_time": _round(self.first_valid_prediction_time),
            "first_unsafe_time": _round(self.first_unsafe_time),
            "late_prediction": self.late_prediction,
            "unsafe_at_start": self.unsafe_at_start,
            "prediction_lead_time_seconds": _round(self.prediction_lead_time_seconds),
            "outcome_status": self.outcome_status,
            "space_fallback_reason": self.space_fallback_reason,
        }


@dataclass
class LeadTimeStats:
    """Distribution of lead times over the valid predictions only.

    Scenarios with no unsafe transition, no prediction, or a late prediction
    contribute nothing — a lead-time number is never manufactured to fill a
    gap, so ``count`` is as important as ``mean``.
    """

    values: List[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> Optional[float]:
        return sum(self.values) / len(self.values) if self.values else None

    @property
    def median(self) -> Optional[float]:
        if not self.values:
            return None
        ordered = sorted(self.values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2.0

    @property
    def minimum(self) -> Optional[float]:
        return min(self.values) if self.values else None

    @property
    def maximum(self) -> Optional[float]:
        return max(self.values) if self.values else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid_predictions": self.count,
            "mean_seconds": _round(self.mean),
            "median_seconds": _round(self.median),
            "min_seconds": _round(self.minimum),
            "max_seconds": _round(self.maximum),
            "definition": (
                "actual_unsafe_transition_time - first_valid_prediction_time, "
                "counted only for forward-looking predictions made before the "
                "transition"
            ),
        }


@dataclass
class EvaluationReport:
    """All scenario results, with a rendered table."""

    results: List[ScenarioResult] = field(default_factory=list)
    alert_severity: str = DEFAULT_ALERT_SEVERITY

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def by_name(self, scenario: str) -> Optional[ScenarioResult]:
        for result in self.results:
            if result.scenario.upper() == scenario.upper():
                return result
        return None

    @property
    def possible_false_positives(self) -> List[ScenarioResult]:
        return [
            r for r in self.results if r.outcome_status == OUTCOME_POSSIBLE_FALSE_POSITIVE
        ]

    @property
    def missed(self) -> List[ScenarioResult]:
        return [r for r in self.results if r.outcome_status == OUTCOME_MISSED]

    @property
    def lead_time(self) -> LeadTimeStats:
        return LeadTimeStats(
            values=[
                r.prediction_lead_time_seconds
                for r in self.results
                if r.prediction_lead_time_seconds is not None
            ]
        )

    @property
    def mean_lead_time_seconds(self) -> Optional[float]:
        return self.lead_time.mean

    @property
    def late_predictions(self) -> List[ScenarioResult]:
        """Predictions that arrived only after the unsafe state had begun."""
        return [r for r in self.results if r.late_prediction]

    @property
    def unsafe_transitions(self) -> List[ScenarioResult]:
        return [r for r in self.results if r.first_unsafe_time is not None]

    def to_table(self) -> str:
        header = (
            f"  {'':2} {'risk':>6} {'severity':<9} {'space':<20} {'conflict':<9} "
            f"{'t-to-risk':>10} {'min sep':>9} {'closing':>10} {'lead':>7}  outcome"
        )
        lines = [
            "SENTINEL PREDICTIVE EVALUATION",
            "==============================",
            f"  alert threshold: severity >= {self.alert_severity}",
            "",
            header,
            "  " + "-" * (len(header) - 2),
        ]
        for r in self.results:
            unit = "m" if r.units == "meters" else "px"
            lines.append(
                f"  {r.scenario:<2} {r.risk_score:6.1f} {r.severity:<9} "
                f"{r.coordinate_space:<20} {str(r.predicted_conflict):<9} "
                f"{_fmt(r.time_to_risk_seconds, 's'):>10} "
                f"{_fmt(r.minimum_separation, unit):>9} "
                f"{_fmt(r.closing_speed, unit + '/s'):>10} "
                f"{_fmt(r.prediction_lead_time_seconds, 's'):>7}  {r.outcome_status}"
            )
        lines.append("")
        stats = self.lead_time
        if stats.count:
            lines.append(
                f"  lead time over {stats.count} valid prediction(s): "
                f"mean {stats.mean:.2f}s  median {stats.median:.2f}s  "
                f"min {stats.minimum:.2f}s  max {stats.maximum:.2f}s"
            )
        else:
            lines.append("  lead time: no valid predictions to measure")
        if self.late_predictions:
            lines.append(
                f"  late predictions (after the transition began): "
                f"{[r.scenario for r in self.late_predictions]}"
            )
        no_transition = sum(
            1 for r in self.results if r.outcome_status == OUTCOME_NO_TRANSITION
        )
        lines.append(
            f"  possible false positives: {len(self.possible_false_positives)}"
            f"   missed: {len(self.missed)}"
            f"   no unsafe transition: {no_transition}"
        )
        lines.append("")
        lines.append(
            "  Lead time is measured against each scenario's own observed"
        )
        lines.append(
            "  separation. It is NOT accuracy, and NOT validated against real"
        )
        lines.append(
            "  incidents. Risk scores are ordinal 0-100, NOT probabilities."
        )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_severity": self.alert_severity,
            "scenario_count": len(self.results),
            "lead_time": self.lead_time.to_dict(),
            "unsafe_transitions": len(self.unsafe_transitions),
            "late_predictions": [r.scenario for r in self.late_predictions],
            "possible_false_positives": [
                r.scenario for r in self.possible_false_positives
            ],
            "missed_detections": [r.scenario for r in self.missed],
            "metric_caveat": (
                "prediction_lead_time_seconds measures warning time against the "
                "scenario's own data; it is not accuracy and not validated "
                "against real incidents"
            ),
            "results": [r.to_dict() for r in self.results],
        }


class ScenarioEvaluator:
    """Runs scenarios through the risk engine and scores the outcome."""

    def __init__(
        self,
        alert_severity: str = DEFAULT_ALERT_SEVERITY,
        step_seconds: float = 0.1,
    ) -> None:
        if alert_severity not in SEVERITIES:
            raise ValueError(f"Unknown severity '{alert_severity}'")
        self.alert_severity = alert_severity
        self.step_seconds = step_seconds

    # ------------------------------------------------------------------
    def evaluate(self, scenario: WorldScenario) -> ScenarioResult:
        engine = RiskEngine(scenario.config, calibration=scenario.calibration)
        report = engine.assess(scenario.memory, at=scenario.at)
        pairs = [a for a in report.assessments if len(a.involved_entity_ids) == 2]
        assessment = (
            max(pairs, key=lambda a: a.risk_score) if pairs else report.top
        )

        first_alert = self._first_alert_time(engine, scenario)
        first_prediction = self._first_valid_prediction_time(engine, scenario)
        first_unsafe, unsafe_at_start = self._unsafe_onset(scenario)

        # Lead time exists only for a genuine forward-looking prediction made
        # before the transition. No transition, no prediction, or a prediction
        # that arrived late all yield None rather than a manufactured number.
        lead = None
        late = False
        if first_prediction is not None and first_unsafe is not None:
            if first_prediction < first_unsafe:
                lead = round(first_unsafe - first_prediction, 4)
            else:
                late = True

        return self._build_result(
            scenario,
            assessment,
            first_alert,
            first_unsafe,
            lead,
            unsafe_at_start,
            first_prediction,
            late,
        )

    def evaluate_all(
        self, scenarios: Optional[Iterable[WorldScenario]] = None
    ) -> EvaluationReport:
        if scenarios is None:
            scenarios = [build() for build in ALL_WORLD_SCENARIOS.values()]
        return EvaluationReport(
            results=[self.evaluate(s) for s in scenarios],
            alert_severity=self.alert_severity,
        )

    # ------------------------------------------------------------------
    def _first_alert_time(
        self, engine: RiskEngine, scenario: WorldScenario
    ) -> Optional[float]:
        """First moment the engine reaches the alert severity."""
        for report in engine.assess_timeline(scenario.memory, step=self.step_seconds):
            if report.at_or_above(self.alert_severity):
                return report.timestamp
        return None

    def _first_valid_prediction_time(
        self, engine: RiskEngine, scenario: WorldScenario
    ) -> Optional[float]:
        """First moment a forward-looking unsafe prediction is made.

        Present-tense ``CURRENTLY_UNSAFE_PROXIMITY`` and unavailable
        predictions are both excluded, so nothing here can credit the system
        for observing what has already happened.
        """
        for report in engine.assess_timeline(scenario.memory, step=self.step_seconds):
            for assessment in report.assessments:
                if assessment.prediction_outcome in PREDICTIVE_OUTCOMES:
                    return report.timestamp
        return None

    def _unsafe_onset(self, scenario: WorldScenario):
        """``(onset_time, unsafe_at_start)`` from the observed separations.

        Measured from the recorded positions, never from the predictor — a
        metric that graded the prediction against itself would be worthless.
        The onset is a *transition* into unsafe separation; a pair that starts
        unsafe reports ``unsafe_at_start`` and no onset.
        """
        space = (
            "ground_plane_meters" if scenario.calibration else "image_pixels"
        )
        thresholds = scenario.config.thresholds(space)
        estimator = MotionEstimator(calibration=scenario.calibration)

        first, last = scenario.memory.span
        if first is None or last is None:
            return None, False

        unsafe_at_start = False
        seen_safe = False
        moment = first
        while moment <= last + 1e-9:
            motions = {}
            for entity_id in scenario.memory.entities():
                history = [
                    e
                    for e in scenario.memory.entity_history(entity_id)
                    if e.timestamp <= moment
                ]
                if history:
                    motion = estimator.estimate(history, moment, entity_id)
                    if motion is not None:
                        motions[entity_id] = motion

            ids = sorted(motions)
            any_unsafe = False
            measured = False
            for i, a_id in enumerate(ids):
                for b_id in ids[i + 1 :]:
                    geometry = separation_geometry(motions[a_id], motions[b_id], space)
                    if geometry is None:
                        continue
                    measured = True
                    if geometry.separation <= thresholds.unsafe_separation:
                        any_unsafe = True

            if measured:
                if any_unsafe:
                    if not seen_safe:
                        # Unsafe from the outset: no transition to predict.
                        unsafe_at_start = True
                    else:
                        return round(moment, 4), unsafe_at_start
                else:
                    seen_safe = True
            moment = round(moment + self.step_seconds, 6)
        return None, unsafe_at_start

    def _build_result(
        self,
        scenario: WorldScenario,
        assessment: Optional[RiskAssessment],
        first_alert: Optional[float],
        first_unsafe: Optional[float],
        lead: Optional[float],
        unsafe_at_start: bool = False,
        first_prediction: Optional[float] = None,
        late_prediction: bool = False,
    ) -> ScenarioResult:
        alerted = first_alert is not None
        became_unsafe = first_unsafe is not None
        if alerted and became_unsafe:
            outcome = OUTCOME_TRUE_POSITIVE
        elif became_unsafe:
            outcome = OUTCOME_MISSED
        elif unsafe_at_start:
            # Nothing transitioned, so neither an alert nor its absence is a
            # prediction about anything.
            outcome = OUTCOME_NO_TRANSITION
        elif alerted:
            outcome = OUTCOME_POSSIBLE_FALSE_POSITIVE
        else:
            outcome = OUTCOME_TRUE_NEGATIVE

        if assessment is None:
            return ScenarioResult(
                scenario=scenario.name,
                description=scenario.description,
                risk_score=0.0,
                severity="normal",
                coordinate_space="image_pixels",
                units="pixels",
                entities=[],
                predicted_conflict=False,
                prediction_outcome=None,
                time_to_risk_status=None,
                time_to_risk_seconds=None,
                first_alert_time=first_alert,
                first_unsafe_time=first_unsafe,
                prediction_lead_time_seconds=lead,
                outcome_status=outcome,
                unsafe_at_start=unsafe_at_start,
                first_valid_prediction_time=first_prediction,
                late_prediction=late_prediction,
            )

        units = (
            "meters"
            if assessment.coordinate_space == "ground_plane_meters"
            else "pixels"
        )
        suffix = "m" if units == "meters" else "px"
        speed_suffix = "m_per_s" if units == "meters" else "px_per_s"
        prediction = assessment.details.get("prediction", {}) or {}
        time_to_risk = assessment.time_to_risk

        return ScenarioResult(
            scenario=scenario.name,
            description=scenario.description,
            risk_score=assessment.risk_score,
            severity=assessment.severity,
            coordinate_space=assessment.coordinate_space,
            units=units,
            entities=list(assessment.involved_entity_ids),
            predicted_conflict=bool(prediction.get("enters_unsafe_separation")),
            prediction_outcome=assessment.prediction_outcome,
            time_to_risk_status=time_to_risk.status if time_to_risk else None,
            time_to_risk_seconds=time_to_risk.seconds if time_to_risk else None,
            current_separation=prediction.get(f"current_separation_{suffix}"),
            minimum_separation=prediction.get(f"minimum_separation_{suffix}"),
            closing_speed=prediction.get(f"closing_speed_{speed_suffix}"),
            first_alert_time=first_alert,
            first_unsafe_time=first_unsafe,
            prediction_lead_time_seconds=lead,
            outcome_status=outcome,
            unsafe_at_start=unsafe_at_start,
            first_valid_prediction_time=first_prediction,
            late_prediction=late_prediction,
            space_fallback_reason=assessment.space_fallback_reason,
        )


# ---------------------------------------------------------------------------
def _round(value: Optional[float], digits: int = 3) -> Optional[float]:
    return round(value, digits) if value is not None else None


def _fmt(value: Optional[float], unit: str) -> str:
    return "-" if value is None else f"{value:.2f}{unit}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m app.evaluation`` runs every scenario and prints the table."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="sentinel-evaluate",
        description="Run the deterministic predictive scenarios and report results.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead")
    parser.add_argument(
        "--alert-severity",
        default=DEFAULT_ALERT_SEVERITY,
        help=f"Severity counted as an alert (default: {DEFAULT_ALERT_SEVERITY})",
    )
    args = parser.parse_args(argv)

    report = ScenarioEvaluator(alert_severity=args.alert_severity).evaluate_all()
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.to_table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
