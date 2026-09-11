"""Synthetic warehouse demo for the M0.3 risk engine.

Runs a scripted incident — a worker walking into a forklift's operating zone
while the forklift approaches — through the real pipeline stages
(tracker → events → temporal memory → risk engine) and prints the analysis.

No video file, no model weights, no network. The scenario is fully
deterministic, so this command prints the same numbers every time and can be
used as a smoke test of the whole reasoning stack::

    python -m app.demo
    python -m app.demo --timeline
    python -m app.demo --json
    python -m app.demo --calibrated     # adds ground-plane measurements (M0.4)
    python -m app.demo --world          # world-space prediction and escalation (M0.5)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .events.generator import EventGenerator
from .memory import TemporalEventMemory
from .perception.trackers.byte_iou import ByteIoUTracker
from .perception.types import Detection
from .calibration import GroundPlaneCalibration
from .calibration.examples import perspective_calibration, warehouse_calibration
from .reasoning import Explainer, RiskConfig, RiskEngine
from .reasoning.kinematics import (
    MotionEstimator,
    ground_closest_approach,
    ground_separation,
)
from .reasoning.models.risk import RiskReport
from .spatial import Zone, ZoneSet

FPS = 10.0
FRAMES = 18

#: The forklift's operating zone, in image pixels for a nominal 960x540 frame.
FORKLIFT_BAY = Zone.from_rect("forklift_bay", (400, 200, 800, 500))


@dataclass
class ScriptedActor:
    """One object moving linearly across the frame."""

    class_name: str
    class_id: int
    start: Tuple[float, float]
    velocity: Tuple[float, float]  # pixels per frame
    size: Tuple[float, float]
    confidence: float = 0.92

    def detection_at(self, frame: int) -> Detection:
        x = self.start[0] + self.velocity[0] * frame
        y = self.start[1] + self.velocity[1] * frame
        return Detection(
            bbox=(x, y, x + self.size[0], y + self.size[1]),
            confidence=self.confidence,
            class_id=self.class_id,
            class_name=self.class_name,
        )


#: The worker walks right, into the bay. The forklift drives left, into them.
WAREHOUSE_ACTORS: Sequence[ScriptedActor] = (
    ScriptedActor("person", 0, (250.0, 300.0), (15.0, 0.0), (40.0, 90.0)),
    ScriptedActor("forklift", 90, (850.0, 320.0), (-20.0, 0.0), (110.0, 70.0), 0.88),
)


def build_warehouse_memory(
    actors: Sequence[ScriptedActor] = WAREHOUSE_ACTORS,
    frames: int = FRAMES,
    fps: float = FPS,
    zones: Optional[ZoneSet] = None,
) -> TemporalEventMemory:
    """Run the scripted scene through tracking → events → temporal memory."""
    zones = zones if zones is not None else ZoneSet([FORKLIFT_BAY])
    tracker = ByteIoUTracker(min_hits=1, max_age=5, iou_threshold=0.1)
    generator = EventGenerator(
        sample_interval=0.1,
        movement_threshold=1.0,
        stationary_threshold=6.0,
        stationary_duration=1.0,
        zones=zones,
    )
    memory = TemporalEventMemory(
        metadata={
            "source": "synthetic warehouse demo",
            "frame_size_px": [960, 540],
            "fps": fps,
        }
    )

    for frame in range(frames):
        timestamp = round(frame / fps, 4)
        detections = [actor.detection_at(frame) for actor in actors]
        tracks = tracker.update(detections, timestamp)
        memory.ingest_many(
            generator.process(
                tracks,
                timestamp=timestamp,
                frame_index=frame,
                lost_tracks=tracker.lost_tracks,
            )
        )
    return memory


def warehouse_config() -> RiskConfig:
    """Risk configuration matching the demo scene."""
    return RiskConfig(
        vehicle_classes=["forklift", "truck", "car"],
        operating_zones=["forklift_bay"],
        zones=ZoneSet([FORKLIFT_BAY]),
    )


#: Severity at which the demo considers the situation "alertable".
ALERT_SEVERITY = "high"


def run_demo(
    at: Optional[float] = None,
    step: float = 0.1,
    calibration: Optional[GroundPlaneCalibration] = None,
) -> RiskReport:
    """Build the scene and assess it.

    With no explicit ``at``, returns the report for the **first moment the
    situation becomes alertable** rather than the final frame. That is the
    operationally interesting question — *when would Sentinel have raised
    this?* — and it keeps a real predicted time-to-incident on the output
    instead of the near-zero value you get once the two have already met.

    Falls back to the last moment in memory if nothing ever crosses the bar.
    """
    memory = build_warehouse_memory()
    engine = RiskEngine(warehouse_config(), calibration=calibration)
    if at is not None:
        return engine.assess(memory, at=at)

    reports = engine.assess_timeline(memory, step=step)
    for report in reports:
        if report.at_or_above(ALERT_SEVERITY):
            return report
    return reports[-1] if reports else engine.assess(memory)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _risk_timeline(step: float = 0.3) -> List[str]:
    """How the risk score develops as the forklift closes."""
    memory = build_warehouse_memory()
    engine = RiskEngine(warehouse_config())
    lines = [
        "RISK DEVELOPMENT OVER TIME",
        "--------------------------",
        f"  {'time':>6}  {'risk':>6}  {'severity':<9}  incident",
    ]
    for report in engine.assess_timeline(memory, step=step):
        top = report.top
        if top is None:
            lines.append(f"  {report.timestamp:6.2f}  {0.0:6.1f}  {'normal':<9}  -")
            continue
        lines.append(
            f"  {report.timestamp:6.2f}  {top.risk_score:6.1f}  "
            f"{top.severity:<9}  {top.incident_type}"
        )
    return lines


def calibration_report() -> List[str]:
    """Demonstrate image -> ground-plane mapping on the demo scene.

    Every number below comes from the synthetic calibration in
    ``app/calibration/examples.py``. It is a mathematical example: no camera
    was involved and nothing here evidences real-world measurement accuracy.
    """
    calibration = warehouse_calibration()
    memory = build_warehouse_memory()
    estimator = MotionEstimator(calibration=calibration)
    at = 0.9  # the moment the demo raises its alert

    lines = [
        "GROUND-PLANE CALIBRATION (M0.4)",
        "-------------------------------",
        f"  calibration    : {calibration.name} ({calibration.quality.point_count} points)",
        f"  source space   : {calibration.source_space}",
        f"  target space   : {calibration.coordinate_space}",
        "  SYNTHETIC EXAMPLE — no camera, no survey, no accuracy claim.",
        "",
        "  Image rectangle      -> World rectangle",
        "    (100,100) (900,100) -> (0,0) (20,0)",
        "    (900,500) (100,500) -> (20,10) (0,10)",
        "",
    ]

    motions = {}
    for entity_id in memory.entities():
        history = [e for e in memory.entity_history(entity_id) if e.timestamp <= at]
        motion = estimator.estimate(history, at, entity_id)
        if motion is not None and motion.world is not None:
            motions[entity_id] = motion

    lines.append(f"  Entities at t={at:.2f}s:")
    lines.append(
        f"    {'entity':<12} {'image px':>16} {'ground m':>16} {'speed':>12}  in region"
    )
    for entity_id, motion in motions.items():
        world = motion.world
        image = f"({motion.position_px[0]:.0f},{motion.position_px[1]:.0f})"
        ground = f"({world.position_m[0]:.2f},{world.position_m[1]:.2f})"
        lines.append(
            f"    {entity_id:<12} {image:>16} {ground:>16} "
            f"{world.speed_m_per_s:>8.2f} m/s  {world.in_calibrated_region}"
        )

    ids = list(motions)
    if len(ids) >= 2:
        a, b = motions[ids[0]], motions[ids[1]]
        separation = ground_separation(a, b)
        approach = ground_closest_approach(a, b)
        lines.append("")
        lines.append("  Pairwise, on the ground plane:")
        lines.append(f"    separation        : {separation:.2f} m")
        if approach is not None:
            lines.append(
                f"    closing speed     : {approach.closing_speed_m_per_s:.2f} m/s"
            )
            lines.append(
                f"    closest approach  : {approach.distance_m:.2f} m "
                f"in {approach.seconds_to_closest_approach:.2f} s"
            )
        lines.append(
            f"    image separation  : "
            f"{((a.position_px[0]-b.position_px[0])**2 + (a.position_px[1]-b.position_px[1])**2)**0.5:.0f} px"
            "   (the same gap, in the other space)"
        )

    lines.append("")
    lines.append("  Why pixels are not enough — a perspective view of a floor:")
    perspective = perspective_calibration()
    for y, label in ((480, "near camera"), (170, "far from camera")):
        left = perspective.image_to_world((400, y))
        right = perspective.image_to_world((500, y))
        lines.append(
            f"    100 px at image y={y:<4} ({label:<15}) = "
            f"{left.distance_to(right):.2f} m on the floor"
        )
    lines.append(
        "    One pixel threshold cannot be correct at both depths. That is the"
    )
    lines.append("    limitation this milestone removes for on-plane points.")

    lines.append("")
    lines.append("  Calibration limitations:")
    for note in calibration.limitations():
        lines.append(f"    - {note}")
    return lines


def world_prediction_report() -> List[str]:
    """Show a situation developing on the calibrated ground plane.

    Walks the "worker enters the forklift bay while the forklift approaches"
    scenario frame by frame, printing the four things that matter at each step:
    the current state, what the constant-velocity predictor expects, how long
    until the unsafe-separation threshold is crossed, and how the risk score
    escalates as evidence accumulates.

    Every number is metres or seconds on a synthetic calibration. It is a
    mathematical example; no camera was involved.
    """
    from .scenarios import load_world

    scenario = load_world("D")
    engine = RiskEngine(scenario.config, calibration=scenario.calibration)
    thresholds = scenario.config.thresholds("ground_plane_meters")

    lines = [
        "WORLD-SPACE PREDICTIVE RISK (M0.5)",
        "----------------------------------",
        f"  scenario       : {scenario.description}",
        f"  coordinate space: ground_plane_meters",
        f"  unsafe separation threshold: "
        f"{thresholds.unsafe_separation:.1f} m",
        "  SYNTHETIC EXAMPLE — no camera, no survey, no accuracy claim.",
        "",
        f"  {'time':>5}  {'sep':>7}  {'closing':>9}  {'min sep':>8}  "
        f"{'t-to-risk':>10}  {'risk':>5}  severity   prediction",
        "  " + "-" * 94,
    ]

    for report in engine.assess_timeline(scenario.memory, step=0.2):
        pairs = [a for a in report.assessments if len(a.involved_entity_ids) == 2]
        if not pairs:
            lines.append(f"  {report.timestamp:5.2f}  {'-':>7}")
            continue
        assessment = max(pairs, key=lambda a: a.risk_score)
        prediction = assessment.details.get("prediction", {}) or {}
        ttr = assessment.time_to_risk

        separation = prediction.get("current_separation_m")
        closing = prediction.get("closing_speed_m_per_s")
        minimum = prediction.get("minimum_separation_m")
        if ttr is None:
            ttr_text = "-"
        elif ttr.status == "already_unsafe":
            ttr_text = "NOW"
        elif ttr.seconds is not None:
            ttr_text = f"{ttr.seconds:.2f}s"
        else:
            ttr_text = "-"

        lines.append(
            f"  {report.timestamp:5.2f}  "
            f"{_fmt_m(separation):>7}  {_fmt_ms(closing):>9}  "
            f"{_fmt_m(minimum):>8}  {ttr_text:>10}  "
            f"{assessment.risk_score:5.1f}  {assessment.severity:<9}  "
            f"{assessment.prediction_outcome}"
        )

    lines.append("")
    lines.append("  Reading the table: separation shrinks, the predictor reports a")
    lines.append("  trajectory conflict before the pair is anywhere near each other,")
    lines.append("  time-to-risk counts down to the threshold crossing, and the risk")
    lines.append("  score escalates as independent factors start to agree.")
    lines.append("")
    lines.append("  The risk score is an ordinal 0-100 ranking, NOT a probability.")
    return lines


def _fmt_m(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.2f}m"


def _fmt_ms(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:+.2f}m/s"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel-demo",
        description=(
            "Run the synthetic warehouse scenario through the M0.3 risk engine. "
            "All measurements are image-space pixels."
        ),
    )
    parser.add_argument(
        "--at", type=float, default=None, help="Assess at this timestamp (seconds)"
    )
    parser.add_argument(
        "--timeline",
        action="store_true",
        help="Also show how risk develops across the whole scene",
    )
    parser.add_argument(
        "--all", action="store_true", help="Show every assessment, not just the top one"
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit the report as JSON instead of text"
    )
    parser.add_argument(
        "--calibrated",
        action="store_true",
        help="Show ground-plane (metre) measurements from the synthetic calibration",
    )
    parser.add_argument(
        "--world",
        action="store_true",
        help="Show world-space prediction, time-to-risk and risk escalation (M0.5)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run the full predictive evaluation harness",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    calibration = warehouse_calibration() if args.calibrated else None
    report = run_demo(at=args.at, calibration=calibration)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        return 0

    explainer = Explainer()
    print(explainer.explain_report(report, limit=None if args.all else 1))

    if args.calibrated:
        print()
        print("\n".join(calibration_report()))

    if args.world:
        print()
        print("\n".join(world_prediction_report()))

    if args.evaluate:
        from .evaluation import ScenarioEvaluator

        print()
        print(ScenarioEvaluator().evaluate_all().to_table())

    if args.timeline:
        print()
        print("\n".join(_risk_timeline()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
