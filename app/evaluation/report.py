"""The assembled benchmark.

``python -m app.evaluation`` runs everything and prints one report. It is
written to be *credible*, not impressive: every section states what it measured
and on what, five metric families stay separate, and anything unvalidated says
so in the output rather than only in the README.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..calibration.examples import perspective_calibration, warehouse_calibration
from ..reasoning.config import RiskConfig
from ..scenarios import SCENARIO_G_FAR_DEPTH_M, SCENARIO_G_NEAR_DEPTH_M, SCENARIO_G_PIXEL_GAP
from .events import EventMetrics, evaluate_events
from .protocol import ClipAnnotation, ClipRun, load_annotation, run_clip
from .scenario_risk import (
    DEFAULT_ALERT_SEVERITY,
    OUTCOME_MISSED,
    OUTCOME_NO_TRANSITION,
    OUTCOME_POSSIBLE_FALSE_POSITIVE,
    OUTCOME_TRUE_POSITIVE,
    EvaluationReport,
    ScenarioEvaluator,
)
from .spatial import (
    CalibrationConditionReport,
    PerspectiveCheck,
    evaluate_calibration_conditioning,
    perspective_distinction_check,
)
from .tracking import MOTMetrics, mot_metrics

#: Calibration configurations exercised by the conditioning section.
CALIBRATION_CASES = (
    (
        "well_conditioned_rectangle",
        [(100, 100), (900, 100), (900, 500), (100, 500)],
        [(0, 0), (20, 0), (20, 10), (0, 10)],
    ),
    (
        "perspective_trapezoid",
        [(350, 150), (650, 150), (900, 500), (100, 500)],
        [(0, 20), (10, 20), (10, 0), (0, 0)],
    ),
    (
        "ill_conditioned_flat_view",
        [(100, 300), (400, 338), (700, 360), (900, 320)],
        [(0, 0), (7, 1.5), (14, 0), (18, 4)],
    ),
    (
        "collinear_rejected",
        [(100, 100), (300, 100), (500, 100), (700, 100)],
        [(0, 0), (5, 0), (10, 0), (15, 0)],
    ),
)


@dataclass
class ClipResult:
    """Tracking and event scores for one annotated clip."""

    name: str
    detector_backend: str
    coordinate_space: str
    frames_evaluated: int
    ground_truth_entities: int
    predicted_tracks: int
    tracking: MOTMetrics
    events: Optional[EventMetrics] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "clip": self.name,
            "detector_backend": self.detector_backend,
            "coordinate_space": self.coordinate_space,
            "frames_evaluated": self.frames_evaluated,
            "ground_truth_entities": self.ground_truth_entities,
            "predicted_tracks": self.predicted_tracks,
            "notes": self.notes,
            "tracking": self.tracking.to_dict(),
        }
        if self.events is not None:
            payload["events"] = self.events.to_dict()
        return payload


@dataclass
class BenchmarkReport:
    """Everything the benchmark measured, in one object."""

    risk: EvaluationReport
    calibration: List[CalibrationConditionReport] = field(default_factory=list)
    perspective: Optional[PerspectiveCheck] = None
    clips: List[ClipResult] = field(default_factory=list)
    thresholds: List[Dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def calibrated_scenarios(self) -> int:
        return sum(
            1 for r in self.risk if r.coordinate_space == "ground_plane_meters"
        )

    @property
    def uncalibrated_scenarios(self) -> int:
        return sum(1 for r in self.risk if r.coordinate_space == "image_pixels")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tracking": [c.to_dict() for c in self.clips],
            "risk": self.risk.to_dict(),
            "spatial": {
                "calibrated_scenarios": self.calibrated_scenarios,
                "uncalibrated_scenarios": self.uncalibrated_scenarios,
                "calibration_conditioning": [c.to_dict() for c in self.calibration],
                "perspective_distinction": (
                    self.perspective.to_dict() if self.perspective else None
                ),
            },
            "thresholds": self.thresholds,
            "validity": VALIDITY_STATEMENT,
        }

    # ------------------------------------------------------------------
    def to_text(self) -> str:
        lines: List[str] = [
            "SENTINEL BENCHMARK",
            "==================",
            "",
        ]
        lines.extend(self._tracking_section())
        lines.append("")
        lines.extend(self._events_section())
        lines.append("")
        lines.extend(self._risk_section())
        lines.append("")
        lines.extend(self._spatial_section())
        lines.append("")
        lines.extend(self._threshold_section())
        lines.append("")
        lines.extend(self._validity_section())
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _tracking_section(self) -> List[str]:
        lines = ["TRACKING", "--------"]
        if not self.clips:
            lines.append("  no annotated clips evaluated")
            return lines
        for clip in self.clips:
            metrics = clip.tracking
            lines.append(f"  clip: {clip.name}  ({clip.notes})")
            lines.append(f"    detector           : {clip.detector_backend}")
            lines.append(f"    frames evaluated   : {clip.frames_evaluated}")
            lines.append(f"    GT entities        : {clip.ground_truth_entities}")
            lines.append(f"    predicted tracks   : {clip.predicted_tracks}")
            lines.append(f"    ID switches        : {metrics.id_switches}")
            lines.append(
                f"    false negatives    : {metrics.false_negatives}"
                f"   false positives: {metrics.false_positives}"
            )
            lines.append(
                f"    MOTA               : {_fmt(metrics.mota)}"
                "     (1-(FN+FP+IDSW)/GT, unbounded below)"
            )
            lines.append(
                f"    IDF1               : {_fmt(metrics.idf1)}"
                "     (identity F1, optimal assignment)"
            )
            lines.append(
                f"    det precision/recall: {_fmt(metrics.precision)} / "
                f"{_fmt(metrics.recall)}   @IoU>={metrics.iou_threshold}"
            )
        return lines

    def _events_section(self) -> List[str]:
        lines = ["EVENTS", "------"]
        scored = [c for c in self.clips if c.events is not None]
        if not scored:
            lines.append("  no annotated events evaluated")
            return lines
        for clip in scored:
            lines.append(f"  clip: {clip.name}")
            lines.append(
                f"    {'action':<16} {'TP':>4} {'FP':>4} {'FN':>4} "
                f"{'P':>7} {'R':>7} {'F1':>7}  mean |dt|"
            )
            for score in clip.events.scored_actions:  # type: ignore[union-attr]
                lines.append(
                    f"    {score.action:<16} {score.true_positives:>4} "
                    f"{score.false_positives:>4} {score.false_negatives:>4} "
                    f"{_fmt(score.precision):>7} {_fmt(score.recall):>7} "
                    f"{_fmt(score.f1):>7}  {_fmt(score.mean_timing_error, 's')}"
                )
            total = clip.events.totals  # type: ignore[union-attr]
            lines.append(
                f"    {'MICRO-AVERAGE':<16} {total.true_positives:>4} "
                f"{total.false_positives:>4} {total.false_negatives:>4} "
                f"{_fmt(total.precision):>7} {_fmt(total.recall):>7} "
                f"{_fmt(total.f1):>7}"
            )
            unscored = clip.events.unscored_actions  # type: ignore[union-attr]
            if unscored:
                summary = ", ".join(
                    f"{action} x{count}" for action, count in sorted(unscored.items())
                )
                lines.append(f"    not annotated, so not scored: {summary}")
        return lines

    def _risk_section(self) -> List[str]:
        stats = self.risk.lead_time
        counts = {
            "unsafe transitions": len(self.risk.unsafe_transitions),
            "correct predictions": sum(
                1 for r in self.risk if r.outcome_status == OUTCOME_TRUE_POSITIVE
            ),
            "false positives": sum(
                1
                for r in self.risk
                if r.outcome_status == OUTCOME_POSSIBLE_FALSE_POSITIVE
            ),
            "missed transitions": sum(
                1 for r in self.risk if r.outcome_status == OUTCOME_MISSED
            ),
            "no unsafe transition": sum(
                1 for r in self.risk if r.outcome_status == OUTCOME_NO_TRANSITION
            ),
            "late predictions": len(self.risk.late_predictions),
        }
        lines = ["RISK", "----", f"  scenarios evaluated : {len(self.risk)}"]
        for label, value in counts.items():
            lines.append(f"  {label:<20}: {value}")
        lines.append("")
        lines.append(
            f"  lead time (valid predictions only, n={stats.count}):"
        )
        if stats.count:
            lines.append(
                f"    mean {stats.mean:.2f}s   median {stats.median:.2f}s   "
                f"min {stats.minimum:.2f}s   max {stats.maximum:.2f}s"
            )
        else:
            lines.append("    none — no scenario produced a valid forward prediction")
        lines.append("")
        lines.append(
            f"  {'':2} {'risk':>6} {'severity':<9} {'space':<20} "
            f"{'prediction':<31} {'lead':>7}  outcome"
        )
        for result in self.risk:
            lines.append(
                f"  {result.scenario:<2} {result.risk_score:6.1f} "
                f"{result.severity:<9} {result.coordinate_space:<20} "
                f"{str(result.prediction_outcome or '-'):<31} "
                f"{_fmt(result.prediction_lead_time_seconds, 's'):>7}  "
                f"{result.outcome_status}"
            )
        return lines

    def _spatial_section(self) -> List[str]:
        lines = [
            "SPATIAL",
            "-------",
            f"  calibrated cases    : {self.calibrated_scenarios}",
            f"  uncalibrated cases  : {self.uncalibrated_scenarios}",
        ]
        if self.perspective:
            verdict = "PASS" if self.perspective.passed else "FAIL"
            lines.append(f"  perspective distinction: {verdict}")
            lines.append(
                f"    {self.perspective.near_pixel_gap:.1f}px near = "
                f"{self.perspective.near_world_gap_m:.2f} m,  "
                f"{self.perspective.far_pixel_gap:.1f}px far = "
                f"{self.perspective.far_world_gap_m:.2f} m  "
                f"(ratio {self.perspective.world_ratio:.2f}x)"
            )
        lines.append("")
        lines.append("  calibration conditioning (world error from a 2px survey error):")
        lines.append(
            f"    {'case':<28} {'accepted':>8} {'cond':>7} "
            f"{'mean err':>9} {'worst err':>10}"
        )
        for report in self.calibration:
            lines.append(
                f"    {report.name:<28} {str(report.accepted):>8} "
                f"{report.minimum_triangle_sine:>7.4f} "
                f"{_fmt(report.mean_error_m, 'm'):>9} "
                f"{_fmt(report.worst_case_error_m, 'm'):>10}"
            )
        return lines

    def _threshold_section(self) -> List[str]:
        lines = [
            "THRESHOLDS",
            "----------",
            f"  {'parameter':<34} {'value':>8} {'units':<8} provenance",
        ]
        for row in self.thresholds:
            lines.append(
                f"  {row['parameter']:<34} {row['value']:>8} "
                f"{row['units']:<8} {row['provenance']}"
            )
        validated = sum(
            1 for r in self.thresholds if r["provenance"] == "empirically-validated"
        )
        lines.append("")
        lines.append(
            f"  {validated} of {len(self.thresholds)} thresholds are empirically "
            "validated. The rest are engineering assumptions."
        )
        return lines

    def _validity_section(self) -> List[str]:
        return ["VALIDITY", "--------"] + [f"  {line}" for line in VALIDITY_STATEMENT]


#: Printed with every benchmark run, so the caveats travel with the numbers.
VALIDITY_STATEMENT = [
    "Every clip in this benchmark is RENDERED, not real footage. Ground truth",
    "comes from the renderer's own drawing commands, so it is independent of",
    "the pipeline — but a rendered scene has perfect contrast, no motion blur,",
    "no occlusion by scene geometry and no detector domain gap.",
    "",
    "These results establish that the pipeline is internally correct and",
    "self-consistent. They establish NOTHING about accuracy on real cameras.",
    "",
    "Risk scores are ordinal 0-100 rankings, NOT calibrated probabilities.",
    "Lead time is warning time measured against each scenario's own data; it is",
    "not accuracy. World-space accuracy is bounded by calibration quality, and",
    "no calibration here was measured against a real surveyed scene.",
]


# ---------------------------------------------------------------------------
def run_benchmark(
    clips: Sequence[str] = (),
    alert_severity: str = DEFAULT_ALERT_SEVERITY,
    include_default_clip: bool = True,
) -> BenchmarkReport:
    """Run every benchmark section and assemble the report."""
    risk = ScenarioEvaluator(alert_severity=alert_severity).evaluate_all()

    calibration = [
        evaluate_calibration_conditioning(image, world, name)
        for name, image, world in CALIBRATION_CASES
    ]
    perspective = perspective_distinction_check(
        perspective_calibration(),
        near_depth_m=SCENARIO_G_NEAR_DEPTH_M,
        far_depth_m=SCENARIO_G_FAR_DEPTH_M,
        pixel_gap=SCENARIO_G_PIXEL_GAP,
    )

    clip_paths = list(clips)
    if include_default_clip and not clip_paths:
        default = _default_clip_path()
        if default:
            clip_paths.append(default)

    clip_results = []
    for path in clip_paths:
        try:
            clip_results.append(evaluate_clip(load_annotation(path)))
        except (OSError, ValueError, ImportError) as exc:
            clip_results.append(
                ClipResult(
                    name=path,
                    detector_backend="unavailable",
                    coordinate_space="-",
                    frames_evaluated=0,
                    ground_truth_entities=0,
                    predicted_tracks=0,
                    tracking=mot_metrics([], []),
                    notes=f"skipped: {exc}",
                )
            )

    return BenchmarkReport(
        risk=risk,
        calibration=calibration,
        perspective=perspective,
        clips=clip_results,
        thresholds=RiskConfig().threshold_audit(),
    )


def evaluate_clip(annotation: ClipAnnotation, **run_options: Any) -> ClipResult:
    """Run the pipeline over one annotated clip and score tracking + events."""
    run = run_clip(annotation, **run_options)
    truth_frames = annotation.ground_truth_frames()
    predicted = run.predicted_frames[: len(truth_frames)]
    # Pad if the pipeline produced fewer frames than were annotated, so missing
    # frames are counted as misses rather than silently dropped.
    while len(predicted) < len(truth_frames):
        predicted.append({})

    tracking = mot_metrics(truth_frames, predicted)
    events = None
    if annotation.events:
        events = evaluate_events(
            annotation.events,
            list(run.memory) if run.memory is not None else [],
            id_mapping=tracking.id_mapping,
            actions=tuple(annotation.annotated_actions),
        )

    return ClipResult(
        name=annotation.name,
        detector_backend=run.detector_backend,
        coordinate_space=run.coordinate_space,
        frames_evaluated=len(truth_frames),
        ground_truth_entities=len(annotation.entities),
        predicted_tracks=len({t for frame in predicted for t in frame}),
        tracking=tracking,
        events=events,
        notes=annotation.notes,
    )


def _default_clip_path() -> Optional[str]:
    import os

    candidate = os.path.join("data", "clips", "rendered_bay.json")
    return candidate if os.path.exists(candidate) else None


def _fmt(value: Optional[float], unit: str = "") -> str:
    return "-" if value is None else f"{value:.3f}{unit}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sentinel-benchmark",
        description=(
            "Run the Sentinel benchmark: tracking, events, risk, spatial and "
            "threshold transparency. All clips are rendered, not real footage."
        ),
    )
    parser.add_argument(
        "--clip",
        action="append",
        default=[],
        metavar="ANNOTATION.json",
        help="Evaluate an annotated clip. Repeatable.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    parser.add_argument(
        "--alert-severity",
        default=DEFAULT_ALERT_SEVERITY,
        help=f"Severity counted as an alert (default: {DEFAULT_ALERT_SEVERITY})",
    )
    parser.add_argument(
        "--no-default-clip",
        action="store_true",
        help="Skip the bundled rendered clip even if it exists",
    )
    args = parser.parse_args(argv)

    report = run_benchmark(
        clips=args.clip,
        alert_severity=args.alert_severity,
        include_default_clip=not args.no_default_clip,
    )
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.to_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
