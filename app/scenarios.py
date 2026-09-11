"""Deterministic world-space scenarios A-H.

Lives in ``app/`` rather than ``tests/`` because the evaluation harness and the
demo both consume it: these scenarios are the benchmark, not merely fixtures
for one test file.

Each scenario scripts detections through the real tracker, event generator and
temporal memory, then hands the result to the risk engine with (or without) a
calibration. Nothing is mocked and nothing is random.

Geometry is designed in METRES first and projected back into pixels through the
calibration, so the scenario descriptions below say what they mean. The
calibration used is the synthetic warehouse fixture (40 px == 1 m); it is a
mathematical example and no claim of camera accuracy attaches to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .calibration import GroundPlaneCalibration
from .calibration.examples import perspective_calibration, warehouse_calibration
from .events.generator import EventGenerator
from .memory import TemporalEventMemory
from .perception.trackers.byte_iou import ByteIoUTracker
from .perception.types import Detection
from .reasoning.config import RiskConfig
from .spatial import Zone, ZoneSet

FPS = 10.0

#: A forklift operating zone, defined in METRES and converted to pixels below.
BAY_WORLD_RECT = (10.0, 2.0, 19.0, 8.0)  # x1, y1, x2, y2 in metres


def _bay_zone(calibration: GroundPlaneCalibration) -> Zone:
    """The operating zone as an image polygon, derived from its metric extent."""
    x1, y1, x2, y2 = BAY_WORLD_RECT
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return Zone(
        name="forklift_bay",
        polygon=tuple(calibration.world_to_image(c) for c in corners),
    )


@dataclass
class WorldActor:
    """An actor whose motion is specified on the ground plane, in metres."""

    class_name: str
    class_id: int
    start_m: Tuple[float, float]
    velocity_m_per_s: Tuple[float, float] = (0.0, 0.0)
    #: Apparent size in pixels. Kept constant for determinism; the fixture
    #: calibration is affine so size does not vary with depth anyway.
    size_px: Tuple[float, float] = (40.0, 90.0)
    confidence: float = 0.92
    visible_frames: Optional[Sequence[int]] = None

    def detection_at(
        self, frame: int, fps: float, calibration: GroundPlaneCalibration
    ) -> Optional[Detection]:
        if self.visible_frames is not None and frame not in self.visible_frames:
            return None
        seconds = frame / fps
        world = (
            self.start_m[0] + self.velocity_m_per_s[0] * seconds,
            self.start_m[1] + self.velocity_m_per_s[1] * seconds,
        )
        # The actor stands at `world`; its bounding box rests on that point.
        foot_x, foot_y = calibration.world_to_image(world)
        width, height = self.size_px
        return Detection(
            bbox=(foot_x - width / 2, foot_y - height, foot_x + width / 2, foot_y),
            confidence=self.confidence,
            class_id=self.class_id,
            class_name=self.class_name,
        )


def worker(start_m, velocity_m_per_s=(0.0, 0.0), **kwargs) -> WorldActor:
    return WorldActor("person", 0, start_m, velocity_m_per_s, (40.0, 90.0), **kwargs)


def forklift(start_m, velocity_m_per_s=(0.0, 0.0), **kwargs) -> WorldActor:
    return WorldActor("forklift", 90, start_m, velocity_m_per_s, (110.0, 70.0), **kwargs)


def build_memory(
    actors: Sequence[WorldActor],
    calibration: GroundPlaneCalibration,
    frames: int = 24,
    fps: float = FPS,
    zones: Optional[ZoneSet] = None,
) -> TemporalEventMemory:
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
        detections = [
            d
            for d in (a.detection_at(frame, fps, calibration) for a in actors)
            if d is not None
        ]
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
class WorldScenario:
    name: str
    description: str
    memory: TemporalEventMemory
    config: RiskConfig
    at: float
    calibration: Optional[GroundPlaneCalibration]
    expectation: str = ""
    expected_space: str = "ground_plane_meters"
    becomes_unsafe: bool = False


def world_config(calibration: Optional[GroundPlaneCalibration], **overrides):
    settings: Dict[str, Any] = {
        "vehicle_classes": ["forklift", "truck", "car"],
        "operating_zones": ["forklift_bay"],
        "zones": ZoneSet([_bay_zone(calibration)]) if calibration else None,
        "report_threshold": 0.0,
    }
    settings.update(overrides)
    return RiskConfig(**settings)


# ===========================================================================
# A — moving apart
# ===========================================================================
def scenario_a() -> WorldScenario:
    calibration = warehouse_calibration()
    return WorldScenario(
        name="A",
        description="entities moving apart (world space)",
        memory=build_memory(
            [
                worker((8.0, 5.0), (-1.5, 0.0)),
                forklift((11.5, 5.0), (2.5, 0.0)),
            ],
            calibration,
        ),
        config=world_config(calibration),
        at=1.5,
        calibration=calibration,
        expectation="low/no trajectory risk",
    )


# ===========================================================================
# B — approaching but projected to stay safely separated
# ===========================================================================
def scenario_b() -> WorldScenario:
    calibration = warehouse_calibration()
    # Parallel lanes 3.5 m apart: they close in x but never within 3.5 m.
    return WorldScenario(
        name="B",
        description="approaching but projected to remain safely separated",
        memory=build_memory(
            [
                worker((3.0, 2.0), (2.0, 0.0)),
                forklift((17.0, 5.5), (-2.5, 0.0)),
            ],
            calibration,
        ),
        config=world_config(calibration),
        at=1.5,
        calibration=calibration,
        expectation="elevated but not critical",
    )


# ===========================================================================
# C — predicted unsafe closest approach
# ===========================================================================
def scenario_c() -> WorldScenario:
    calibration = warehouse_calibration()
    return WorldScenario(
        name="C",
        description="approaching with predicted unsafe closest approach",
        memory=build_memory(
            [
                worker((4.0, 5.0), (2.0, 0.0)),
                forklift((16.0, 5.0), (-3.0, 0.0)),
            ],
            calibration,
        ),
        config=world_config(calibration),
        at=1.5,
        calibration=calibration,
        expectation="high trajectory-conflict risk",
        becomes_unsafe=True,
    )


# ===========================================================================
# D — operating zone entry while another entity approaches
# ===========================================================================
def scenario_d() -> WorldScenario:
    calibration = warehouse_calibration()
    return WorldScenario(
        name="D",
        description="worker enters the forklift bay while the forklift approaches",
        memory=build_memory(
            [
                worker((8.0, 5.0), (2.5, 0.0)),      # crosses x=10 into the bay
                forklift((18.5, 5.0), (-3.0, 0.0)),
            ],
            calibration,
            zones=ZoneSet([_bay_zone(calibration)]),
        ),
        config=world_config(calibration),
        at=1.2,
        calibration=calibration,
        expectation="higher risk from combined spatial + contextual evidence",
        becomes_unsafe=True,
    )


# ===========================================================================
# E — two stationary entities
# ===========================================================================
def scenario_e() -> WorldScenario:
    calibration = warehouse_calibration()
    return WorldScenario(
        name="E",
        description="two stationary entities",
        memory=build_memory(
            [worker((8.0, 5.0)), forklift((11.5, 5.0))],
            calibration,
        ),
        config=world_config(calibration),
        at=1.5,
        calibration=calibration,
        expectation="no trajectory risk",
    )


# ===========================================================================
# F — insufficient history
# ===========================================================================
def scenario_f() -> WorldScenario:
    calibration = warehouse_calibration()
    return WorldScenario(
        name="F",
        description="insufficient history to support any prediction",
        memory=build_memory(
            [
                worker((6.0, 5.0), (2.0, 0.0)),
                forklift((14.0, 5.0), (-3.0, 0.0)),
            ],
            calibration,
            frames=2,
        ),
        config=world_config(calibration),
        at=0.1,
        calibration=calibration,
        expectation="no fabricated trajectory prediction",
    )


# ===========================================================================
# G — perspective calibration: equal pixel gaps, different world distances
# ===========================================================================
def world_offset_for_pixel_gap(
    calibration: GroundPlaneCalibration,
    depth_m: float,
    start_x_m: float,
    target_px: float,
    tolerance: float = 1e-4,
) -> float:
    """Metres in x that correspond to ``target_px`` at a given ground depth.

    Found by bisection on the calibration itself rather than by assuming any
    pixels-per-metre ratio — under perspective no such constant exists, which
    is precisely what this scenario demonstrates.
    """
    origin_px = calibration.world_to_image((start_x_m, depth_m))

    def gap(offset: float) -> float:
        other = calibration.world_to_image((start_x_m + offset, depth_m))
        return ((other[0] - origin_px[0]) ** 2 + (other[1] - origin_px[1]) ** 2) ** 0.5

    low, high = 0.0, 1.0
    while gap(high) < target_px and high < 1e4:
        high *= 2.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if gap(middle) < target_px:
            low = middle
        else:
            high = middle
        if high - low < tolerance:
            break
    return (low + high) / 2.0


#: Pixel gap used for both pairs in scenario G.
SCENARIO_G_PIXEL_GAP = 110.0
SCENARIO_G_NEAR_DEPTH_M = 2.0
SCENARIO_G_FAR_DEPTH_M = 17.0


def scenario_g() -> WorldScenario:
    """Two stationary pairs with the SAME pixel separation at different depths.

    In image space the two pairs are indistinguishable. On the ground plane one
    is comfortably clear and the other is dangerously close. Any system that
    reasons in pixels must score them identically; a system that reasons on the
    ground plane must not. That is the whole argument for this milestone, made
    as a single assertion.
    """
    calibration = perspective_calibration()
    near_offset = world_offset_for_pixel_gap(
        calibration, SCENARIO_G_NEAR_DEPTH_M, 2.0, SCENARIO_G_PIXEL_GAP
    )
    far_offset = world_offset_for_pixel_gap(
        calibration, SCENARIO_G_FAR_DEPTH_M, 2.0, SCENARIO_G_PIXEL_GAP
    )
    return WorldScenario(
        name="G",
        description="equal pixel gaps representing different world distances",
        memory=build_memory(
            [
                worker((2.0, SCENARIO_G_NEAR_DEPTH_M)),
                forklift((2.0 + near_offset, SCENARIO_G_NEAR_DEPTH_M)),
                worker((2.0, SCENARIO_G_FAR_DEPTH_M)),
                forklift((2.0 + far_offset, SCENARIO_G_FAR_DEPTH_M)),
            ],
            calibration,
            frames=20,
        ),
        config=world_config(calibration),
        at=1.5,
        calibration=calibration,
        expectation="risk decisions use world-space geometry, not pixel geometry",
    )


# ===========================================================================
# H — uncalibrated camera
# ===========================================================================
def scenario_h() -> WorldScenario:
    """The same motion as C, but the engine is given no calibration."""
    calibration = warehouse_calibration()
    memory = build_memory(
        [
            worker((4.0, 5.0), (2.0, 0.0)),
            forklift((16.0, 5.0), (-3.0, 0.0)),
        ],
        calibration,
    )
    return WorldScenario(
        name="H",
        description="uncalibrated camera — image-space fallback",
        memory=memory,
        config=world_config(None, zones=None, operating_zones=[]),
        at=1.5,
        calibration=None,
        expectation="image-space fallback remains functional and labelled",
        expected_space="image_pixels",
        becomes_unsafe=True,
    )


ALL_WORLD_SCENARIOS = {
    "A": scenario_a,
    "B": scenario_b,
    "C": scenario_c,
    "D": scenario_d,
    "E": scenario_e,
    "F": scenario_f,
    "G": scenario_g,
    "H": scenario_h,
}


def load_world(name: str) -> WorldScenario:
    return ALL_WORLD_SCENARIOS[name.upper()]()
