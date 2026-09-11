"""Test-facing re-export of the world-space scenarios.

The scenarios themselves live in ``app/scenarios.py`` because the evaluation
harness and demo consume them too.
"""

from app.scenarios import (  # noqa: F401
    ALL_WORLD_SCENARIOS,
    BAY_WORLD_RECT,
    SCENARIO_G_FAR_DEPTH_M,
    SCENARIO_G_NEAR_DEPTH_M,
    SCENARIO_G_PIXEL_GAP,
    WorldActor,
    WorldScenario,
    build_memory,
    forklift,
    load_world,
    world_config,
    world_offset_for_pixel_gap,
    worker,
)
