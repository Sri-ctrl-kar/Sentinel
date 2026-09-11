"""Deterministic synthetic scenarios for the M0.3 risk engine.

Each scenario scripts detections frame by frame, pushes them through the real
tracker, the real event generator and the real temporal memory, and returns the
memory plus the config and the moment to assess. No video, no model, no
randomness — the same scenario always produces the same risk score.

Geometry note: every coordinate is an image pixel in a nominal 960x540 frame.
Velocities are stated in pixels per *frame* here and become pixels per second
in the engine via the frame rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.events.generator import EventGenerator
from app.memory import TemporalEventMemory
from app.perception.trackers.byte_iou import ByteIoUTracker
from app.perception.types import Detection
from app.reasoning.config import RiskConfig
from app.spatial import Zone, ZoneSet

FPS = 10.0

#: The operating zone used by the zone-aware scenarios.
FORKLIFT_BAY = Zone.from_rect("forklift_bay", (400, 200, 800, 500))
WAREHOUSE_ZONES = ZoneSet([FORKLIFT_BAY])


@dataclass
class Actor:
    """One scripted object moving linearly across the frame."""

    class_name: str
    class_id: int
    start: Tuple[float, float]  # bbox top-left, pixels
    velocity: Tuple[float, float] = (0.0, 0.0)  # pixels per frame
    size: Tuple[float, float] = (40.0, 90.0)
    confidence: float = 0.92
    visible_frames: Optional[Sequence[int]] = None

    def detection_at(self, frame: int) -> Optional[Detection]:
        if self.visible_frames is not None and frame not in self.visible_frames:
            return None
        x = self.start[0] + self.velocity[0] * frame
        y = self.start[1] + self.velocity[1] * frame
        return Detection(
            bbox=(x, y, x + self.size[0], y + self.size[1]),
            confidence=self.confidence,
            class_id=self.class_id,
            class_name=self.class_name,
        )


def person(start, velocity=(0.0, 0.0), **kwargs) -> Actor:
    return Actor("person", 0, start, velocity, size=(40.0, 90.0), **kwargs)


def forklift(start, velocity=(0.0, 0.0), **kwargs) -> Actor:
    return Actor("forklift", 90, start, velocity, size=(110.0, 70.0), **kwargs)


def build_memory(
    actors: Sequence[Actor],
    frames: int = 20,
    fps: float = FPS,
    zones: Optional[ZoneSet] = None,
) -> TemporalEventMemory:
    """Run scripted actors through tracker → generator → temporal memory.

    Event thresholds are set tight (1px of movement, 0.1s heartbeat) so that
    every frame produces a positioned event. The risk engine needs a dense
    position history to estimate image velocity; a production configuration
    would trade some of that density for a smaller log.
    """
    tracker = ByteIoUTracker(min_hits=1, max_age=5, iou_threshold=0.1)
    generator = EventGenerator(
        sample_interval=0.1,
        movement_threshold=1.0,
        stationary_threshold=6.0,
        stationary_duration=1.0,
        zones=zones,
    )
    memory = TemporalEventMemory()

    for frame in range(frames):
        timestamp = round(frame / fps, 4)
        detections = [d for d in (a.detection_at(frame) for a in actors) if d]
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


@dataclass
class Scenario:
    """A named situation, ready to hand to the engine."""

    name: str
    description: str
    memory: TemporalEventMemory
    config: RiskConfig
    at: float
    expected: Dict[str, Any] = field(default_factory=dict)


def warehouse_config(**overrides: Any) -> RiskConfig:
    """Risk config for the warehouse scenarios."""
    settings: Dict[str, Any] = {
        "vehicle_classes": ["forklift", "truck", "car"],
        "operating_zones": ["forklift_bay"],
        "zones": WAREHOUSE_ZONES,
    }
    settings.update(overrides)
    return RiskConfig(**settings)


# ===========================================================================
# SCENARIO A — moving apart
# ===========================================================================
def scenario_a() -> Scenario:
    """Person and vehicle diverging. Nothing is developing."""
    return Scenario(
        name="A",
        description="person and vehicle moving apart",
        memory=build_memory(
            [
                person((400, 300), velocity=(-25.0, 0.0)),
                forklift((500, 320), velocity=(25.0, 0.0)),
            ],
            frames=16,
        ),
        config=warehouse_config(),
        at=1.5,
        expected={"severity_at_most": "low"},
    )


# ===========================================================================
# SCENARIO B — converging but not intersecting
# ===========================================================================
def scenario_b() -> Scenario:
    """Closing fast, but on parallel lines 200px apart. A near miss."""
    return Scenario(
        name="B",
        description="converging trajectories that miss each other",
        memory=build_memory(
            [
                person((150, 150), velocity=(25.0, 0.0)),
                forklift((650, 310), velocity=(-25.0, 0.0)),
            ],
            frames=16,
        ),
        config=warehouse_config(),
        at=0.7,
        expected={"severity_at_least": "low", "severity_at_most": "medium"},
    )


# ===========================================================================
# SCENARIO C — intersecting trajectories
# ===========================================================================
def scenario_c() -> Scenario:
    """Head-on convergence along the same line. A real collision course."""
    return Scenario(
        name="C",
        description="intersecting trajectories",
        memory=build_memory(
            [
                person((150, 290), velocity=(25.0, 0.0)),
                forklift((650, 300), velocity=(-25.0, 0.0)),
            ],
            frames=16,
        ),
        config=warehouse_config(),
        at=0.7,
        expected={"severity_at_least": "high"},
    )


# ===========================================================================
# SCENARIO D — person in operating zone, vehicle approaching
# ===========================================================================
def scenario_d() -> Scenario:
    """The canonical incident: worker in the bay, forklift bearing down."""
    return Scenario(
        name="D",
        description="person in vehicle operating zone while vehicle approaches",
        memory=build_memory(
            [
                person((250, 300), velocity=(15.0, 0.0)),
                forklift((850, 320), velocity=(-20.0, 0.0)),
            ],
            frames=18,
            zones=WAREHOUSE_ZONES,
        ),
        config=warehouse_config(),
        at=1.5,
        expected={"severity_at_least": "critical"},
    )


# ===========================================================================
# SCENARIO E — many weak signals
# ===========================================================================
def scenario_e() -> Scenario:
    """Nothing alarming on its own; alarming taken together.

    The person has stepped in and out of the forklift bay repeatedly (each
    entry individually minor), and is now outside it with a forklift closing
    slowly at a moderate distance. Every factor scores in the "normal" band
    alone; corroboration is what makes it a situation.
    """
    # The person paces in and out of the bay on a 16-frame triangle wave.
    #
    # The 8px step was originally required to work around the M0.3 tracker
    # fragmentation bug: association scored only on the predicted box, so at a
    # reversal the prediction pointed backwards and the pacer shattered into a
    # crowd of strangers with no history between them. M0.3.1 fixed that (the
    # tracker now falls back to the last observed box), and this path no longer
    # depends on the small step — see tests/test_tracker_identity.py. The
    # geometry is kept as-is so the documented M0.3 scores stay stable.
    # Persistence is only visible when the repetition belongs to ONE entity.
    class _Pacer(Actor):
        def detection_at(self, frame: int):
            phase = frame % 16
            offset = phase if phase <= 8 else 16 - phase
            x = 340.0 + 8.0 * offset  # 340..404; the anchor crosses x=400 at 380
            return Detection(
                bbox=(x, 300.0, x + 40.0, 390.0),
                confidence=0.9,
                class_id=0,
                class_name="person",
            )

    pacer = _Pacer("person", 0, (340.0, 300.0))
    return Scenario(
        name="E",
        description="multiple individually weak signals combine",
        memory=build_memory(
            [pacer, forklift((873, 330), velocity=(-8.5, 0.0))],
            frames=49,
            zones=WAREHOUSE_ZONES,
        ),
        config=warehouse_config(),
        at=4.8,
        expected={"severity_at_least": "high"},
    )


# ===========================================================================
# SCENARIO F — everything stationary
# ===========================================================================
def scenario_f() -> Scenario:
    """A parked forklift and a standing worker. Close, but not developing."""
    return Scenario(
        name="F",
        description="stationary person and stationary vehicle",
        memory=build_memory(
            [person((300, 300)), forklift((400, 320))],
            frames=20,
        ),
        config=warehouse_config(),
        at=1.5,
        expected={"trajectory_score": 0.0, "no_time_to_incident": True},
    )


# ===========================================================================
# SCENARIO G — insufficient history
# ===========================================================================
def scenario_g() -> Scenario:
    """Both entities have only just been seen. The engine must not guess."""
    return Scenario(
        name="G",
        description="insufficient history to predict a trajectory",
        memory=build_memory(
            [
                person((300, 300), velocity=(25.0, 0.0)),
                forklift((450, 320), velocity=(-25.0, 0.0)),
            ],
            frames=2,
        ),
        config=warehouse_config(),
        at=0.1,
        expected={"trajectory_score": 0.0, "insufficient_history": True},
    )


ALL_SCENARIOS = {
    "A": scenario_a,
    "B": scenario_b,
    "C": scenario_c,
    "D": scenario_d,
    "E": scenario_e,
    "F": scenario_f,
    "G": scenario_g,
}


def load(name: str) -> Scenario:
    return ALL_SCENARIOS[name.upper()]()
