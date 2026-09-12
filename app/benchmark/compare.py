"""Comparing two benchmark runs: speed, and — more importantly — sameness.

Speed is the easy half. The half that matters is parity: acceleration is only
worth having if Sentinel *means the same thing* on the new device. A GPU that
produces 4x the throughput and a different set of incidents has not
accelerated Sentinel, it has replaced it.

Tolerances, and why they are not zero
-------------------------------------
Identical weights on different hardware do not produce bit-identical floats.
Different BLAS kernels, different reduction orders, different fused
operations — all legitimate, all giving detection confidences that differ in
the fifth decimal. A box at 0.2500001 versus 0.2499999 confidence flips across
the 0.25 threshold, which changes a detection count by one without anything
being wrong.

So the numeric tolerances below are loose, and the semantic ones are strict:

* detection and event counts may differ by a small relative margin;
* mean confidence may differ by a small absolute margin;
* the **set of detected classes** must match;
* the **risk score** must match to two decimal places, and severity,
  incident type, lifecycle state and coordinate space must match exactly.

That last group is the actual claim of this milestone: the device changed, the
verdict did not. Any difference is reported, never smoothed away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .harness import BenchmarkResult

#: Relative tolerance on counts that a threshold crossing can legitimately move.
COUNT_TOLERANCE = 0.02
#: Absolute tolerance on mean detection confidence.
CONFIDENCE_TOLERANCE = 0.01
#: Risk scores are reported to two decimals; they must agree there.
RISK_TOLERANCE = 0.01


@dataclass(frozen=True)
class Speedup:
    """How much faster one device was than another, per stage."""

    baseline_label: str
    candidate_label: str
    inference: float
    pipeline: float
    baseline_inference_fps: float
    candidate_inference_fps: float
    baseline_pipeline_fps: float
    candidate_pipeline_fps: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline_label,
            "candidate": self.candidate_label,
            "inference_speedup": round(self.inference, 3),
            "pipeline_speedup": round(self.pipeline, 3),
            "baseline_inference_fps": round(self.baseline_inference_fps, 2),
            "candidate_inference_fps": round(self.candidate_inference_fps, 2),
            "baseline_pipeline_fps": round(self.baseline_pipeline_fps, 2),
            "candidate_pipeline_fps": round(self.candidate_pipeline_fps, 2),
        }

    def lines(self) -> List[str]:
        return [
            f"inference : {self.baseline_inference_fps:8.2f} fps -> "
            f"{self.candidate_inference_fps:8.2f} fps   {self.inference:.2f}x",
            f"pipeline  : {self.baseline_pipeline_fps:8.2f} fps -> "
            f"{self.candidate_pipeline_fps:8.2f} fps   {self.pipeline:.2f}x",
        ]


def speedup(baseline: BenchmarkResult, candidate: BenchmarkResult) -> Speedup:
    """Throughput ratio between two runs of the *same* workload."""
    if baseline.config.workload_key() != candidate.config.workload_key():
        raise ValueError(
            "refusing to compare two different workloads; the frame count, "
            "resolution, weights, thresholds and clip must all match"
        )
    return Speedup(
        baseline_label=baseline.device.label,
        candidate_label=candidate.device.label,
        inference=_ratio(candidate.inference_fps, baseline.inference_fps),
        pipeline=_ratio(candidate.pipeline_fps, baseline.pipeline_fps),
        baseline_inference_fps=baseline.inference_fps,
        candidate_inference_fps=candidate.inference_fps,
        baseline_pipeline_fps=baseline.pipeline_fps,
        candidate_pipeline_fps=candidate.pipeline_fps,
    )


def _ratio(candidate: float, baseline: float) -> float:
    return candidate / baseline if baseline > 0 else 0.0


# ---------------------------------------------------------------------------
@dataclass
class ParityDifference:
    """One way two devices disagreed."""

    field: str
    baseline: Any
    candidate: Any
    tolerated: bool
    note: str = ""

    def __str__(self) -> str:
        verdict = "within tolerance" if self.tolerated else "DIFFERS"
        detail = f" — {self.note}" if self.note else ""
        return f"{self.field}: {self.baseline} vs {self.candidate} [{verdict}]{detail}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "tolerated": self.tolerated,
            "note": self.note,
        }


@dataclass
class ParityReport:
    """Whether two devices produced the same Sentinel, not just the same speed."""

    baseline_label: str
    candidate_label: str
    differences: List[ParityDifference] = field(default_factory=list)
    #: Things the reader must know that are not differences — above all, a
    #: comparison that could not be made. A check that never ran must never
    #: read as a check that passed.
    notes: List[str] = field(default_factory=list)
    #: Whether the risk verdict was actually compared on both devices.
    risk_compared: bool = False

    @property
    def ok(self) -> bool:
        """True when nothing differs beyond the documented tolerances."""
        return all(d.tolerated for d in self.differences)

    @property
    def semantics_preserved(self) -> Optional[bool]:
        """True when no risk field differs, ``None`` when none was compared.

        ``None`` is the important value: "nothing was compared" and "everything
        matched" must never render as the same answer.
        """
        if not self.risk_compared:
            return None
        return not any(d.field.startswith("risk.") for d in self.differences)

    def summary(self) -> str:
        if not self.differences:
            return "identical on both devices"
        if self.ok:
            return (
                f"{len(self.differences)} numeric difference(s), all within the "
                "documented tolerances; risk semantics unchanged"
            )
        return f"{sum(1 for d in self.differences if not d.tolerated)} difference(s) beyond tolerance"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline_label,
            "candidate": self.candidate_label,
            "ok": self.ok,
            "semantics_preserved": self.semantics_preserved,
            "summary": self.summary(),
            "differences": [d.to_dict() for d in self.differences],
            "notes": list(self.notes),
        }


def compare_workloads(
    baseline: BenchmarkResult, candidate: BenchmarkResult
) -> ParityReport:
    """Compare what two devices actually detected, tracked and emitted."""
    report = ParityReport(baseline.device.label, candidate.device.label)
    left, right = baseline.workload, candidate.workload

    _compare_count(report, "detections", left.detections, right.detections)
    _compare_count(report, "tracks_created", left.tracks_created, right.tracks_created)
    _compare_count(report, "events", left.events, right.events)

    if abs(left.mean_confidence - right.mean_confidence) > CONFIDENCE_TOLERANCE:
        report.differences.append(
            ParityDifference(
                "mean_confidence",
                round(left.mean_confidence, 5),
                round(right.mean_confidence, 5),
                tolerated=False,
                note=f"absolute tolerance {CONFIDENCE_TOLERANCE}",
            )
        )

    left_classes = set(left.class_counts)
    right_classes = set(right.class_counts)
    if left_classes != right_classes:
        report.differences.append(
            ParityDifference(
                "class_labels",
                sorted(left_classes),
                sorted(right_classes),
                tolerated=False,
                note="the set of detected classes must match exactly",
            )
        )
    else:
        for name in sorted(left_classes):
            _compare_count(
                report,
                f"class_counts.{name}",
                left.class_counts[name],
                right.class_counts[name],
            )
    return report


def compare_risk(
    baseline: Optional[Dict[str, Any]],
    candidate: Optional[Dict[str, Any]],
    report: Optional[ParityReport] = None,
) -> ParityReport:
    """Compare the deterministic verdict two devices led to.

    ``baseline`` and ``candidate`` are ``IncidentEvidence.to_dict()`` payloads,
    or ``None`` when a run produced no assessment. These fields have no
    tolerance: a different severity or incident type on a different GPU is a
    correctness failure, not numerical noise.
    """
    report = report or ParityReport("baseline", "candidate")
    if baseline is None and candidate is None:
        report.notes.append(
            "neither device produced a risk assessment on this clip, so the "
            "risk fields were not compared — this is not a passed comparison"
        )
        return report
    if baseline is None or candidate is None:
        report.risk_compared = True
        report.differences.append(
            ParityDifference(
                "risk.assessment",
                "present" if baseline else "none",
                "present" if candidate else "none",
                tolerated=False,
                note="one device produced an incident and the other did not",
            )
        )
        return report

    score_delta = abs(float(baseline["risk_score"]) - float(candidate["risk_score"]))
    if score_delta > RISK_TOLERANCE:
        report.differences.append(
            ParityDifference(
                "risk.risk_score",
                baseline["risk_score"],
                candidate["risk_score"],
                tolerated=False,
                note=f"tolerance {RISK_TOLERANCE}",
            )
        )

    report.risk_compared = True
    report.notes.append(
        f"risk compared: score {baseline['risk_score']} vs "
        f"{candidate['risk_score']}, severity {baseline['severity']} vs "
        f"{candidate['severity']}"
    )
    for key in (
        "severity",
        "incident_type",
        "incident_state",
        "coordinate_space",
        "entities",
    ):
        left = _comparable(baseline.get(key))
        right = _comparable(candidate.get(key))
        if left != right:
            report.differences.append(
                ParityDifference(
                    f"risk.{key}",
                    left,
                    right,
                    tolerated=False,
                    note="risk semantics must not change with the device",
                )
            )
    return report


def _comparable(value: Any) -> Any:
    """Reduce evidence entities to identity + class, dropping float positions."""
    if isinstance(value, list):
        return [
            (item.get("entity_id"), item.get("class_name"))
            if isinstance(item, dict)
            else item
            for item in value
        ]
    return value


def _compare_count(report: ParityReport, label: str, left: int, right: int) -> None:
    if left == right:
        return
    scale = max(abs(left), abs(right), 1)
    relative = abs(left - right) / scale
    report.differences.append(
        ParityDifference(
            label,
            left,
            right,
            tolerated=relative <= COUNT_TOLERANCE,
            note=f"relative difference {relative:.3f}, tolerance {COUNT_TOLERANCE}",
        )
    )
