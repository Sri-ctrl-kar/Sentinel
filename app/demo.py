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
from .reasoning import Explainer, RiskConfig, RiskEngine
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


def run_demo(at: Optional[float] = None, step: float = 0.1) -> RiskReport:
    """Build the scene and assess it.

    With no explicit ``at``, returns the report for the **first moment the
    situation becomes alertable** rather than the final frame. That is the
    operationally interesting question — *when would Sentinel have raised
    this?* — and it keeps a real predicted time-to-incident on the output
    instead of the near-zero value you get once the two have already met.

    Falls back to the last moment in memory if nothing ever crosses the bar.
    """
    memory = build_warehouse_memory()
    engine = RiskEngine(warehouse_config())
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
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_demo(at=args.at)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        return 0

    explainer = Explainer()
    print(explainer.explain_report(report, limit=None if args.all else 1))

    if args.timeline:
        print()
        print("\n".join(_risk_timeline()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
