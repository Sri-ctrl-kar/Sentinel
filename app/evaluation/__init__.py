"""Evaluation and benchmarking.

Consumes the outputs of every layer below it and scores them. It does not
reimplement detection, tracking, event generation, calibration, prediction or
risk — it runs them and grades the result, which is why a metric moving here
always points at a layer rather than at the metric itself.

Five metric families, deliberately never collapsed into one "accuracy" number:

1. :mod:`~app.evaluation.tracking` — detection and tracking quality
2. :mod:`~app.evaluation.events` — event recognition quality
3. :mod:`~app.evaluation.scenario_risk` — unsafe-state and prediction quality
4. :mod:`~app.evaluation.spatial` — calibration conditioning and its effect
5. :mod:`~app.evaluation.report` — the assembled benchmark

Nothing here imports a model runtime, and nothing here is imported by the risk
engine.
"""

from .assignment import solve_max_score, solve_min_cost
from .events import (
    EVALUATED_ACTIONS,
    ActionScore,
    AnnotatedEvent,
    EventMetrics,
    evaluate_events,
)
from .protocol import ClipAnnotation, EntityTrack, load_annotation, run_clip
from .report import BenchmarkReport, run_benchmark
from .scenario_risk import (
    DEFAULT_ALERT_SEVERITY,
    OUTCOME_MISSED,
    OUTCOME_NO_TRANSITION,
    OUTCOME_POSSIBLE_FALSE_POSITIVE,
    OUTCOME_TRUE_NEGATIVE,
    OUTCOME_TRUE_POSITIVE,
    EvaluationReport,
    LeadTimeStats,
    ScenarioEvaluator,
    ScenarioResult,
)
from .spatial import (
    CalibrationConditionReport,
    PerspectiveCheck,
    evaluate_calibration_conditioning,
    perspective_distinction_check,
)
from .tracking import MOTMetrics, TrackingMetrics, evaluate, mot_metrics

__all__ = [
    "solve_min_cost",
    "solve_max_score",
    "TrackingMetrics",
    "MOTMetrics",
    "evaluate",
    "mot_metrics",
    "AnnotatedEvent",
    "ActionScore",
    "EventMetrics",
    "evaluate_events",
    "EVALUATED_ACTIONS",
    "ScenarioEvaluator",
    "ScenarioResult",
    "EvaluationReport",
    "LeadTimeStats",
    "DEFAULT_ALERT_SEVERITY",
    "OUTCOME_TRUE_POSITIVE",
    "OUTCOME_POSSIBLE_FALSE_POSITIVE",
    "OUTCOME_MISSED",
    "OUTCOME_TRUE_NEGATIVE",
    "OUTCOME_NO_TRANSITION",
    "CalibrationConditionReport",
    "PerspectiveCheck",
    "evaluate_calibration_conditioning",
    "perspective_distinction_check",
    "ClipAnnotation",
    "EntityTrack",
    "load_annotation",
    "run_clip",
    "BenchmarkReport",
    "run_benchmark",
]


def main(argv=None) -> int:
    """``python -m app.evaluation`` runs the full benchmark."""
    from .report import main as report_main

    return report_main(argv)
