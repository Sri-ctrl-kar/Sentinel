"""A synthetic calibration used for demonstration and tests.

READ THIS BEFORE CITING ANY NUMBER FROM HERE
--------------------------------------------
This is a **mathematical example**, not a camera measurement. The image
rectangle was chosen for arithmetic convenience: it maps to exactly 20m x 10m
so that hand-checking the transform is easy. No camera was involved, no floor
was surveyed, and the residuals are zero only because four points always fit a
homography exactly.

It demonstrates that the transform works. It is **not** evidence that Sentinel
measures real distances accurately — that claim requires a real camera and real
surveyed points, and nothing in this repository establishes it.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .planar import GroundPlaneCalibration

#: Image corners of the notional floor rectangle, in pixels.
WAREHOUSE_IMAGE_CORNERS: Sequence[Tuple[float, float]] = (
    (100.0, 100.0),
    (900.0, 100.0),
    (900.0, 500.0),
    (100.0, 500.0),
)

#: The same corners on the floor, in metres.
WAREHOUSE_WORLD_CORNERS: Sequence[Tuple[float, float]] = (
    (0.0, 0.0),
    (20.0, 0.0),
    (20.0, 10.0),
    (0.0, 10.0),
)


def warehouse_calibration() -> GroundPlaneCalibration:
    """A 20m x 10m floor rectangle mapped from an 800x400px image rectangle.

    Deliberately an affine special case (the image rectangle maps to a world
    rectangle with no perspective foreshortening), which makes every expected
    value checkable by hand: 40px along x is 1m, 40px along y is 1m.

    A real overhead-angled camera would produce a trapezoid in the image, and
    the same code handles it — see ``perspective_calibration`` below, which
    exercises the genuinely projective path.
    """
    return GroundPlaneCalibration(
        image_points=WAREHOUSE_IMAGE_CORNERS,
        world_points=WAREHOUSE_WORLD_CORNERS,
        name="synthetic_warehouse",
        description=(
            "Synthetic 20m x 10m floor rectangle. A mathematical example only: "
            "no camera, no survey, no accuracy claim."
        ),
    )


def perspective_calibration() -> GroundPlaneCalibration:
    """A trapezoidal image region, as a real angled camera would see a floor.

    The far edge of the floor is narrower in the image than the near edge, so
    a fixed pixel distance means very different ground distances at the top and
    bottom of the frame. This is the case pixel-space reasoning gets wrong and
    the reason this milestone exists.
    """
    return GroundPlaneCalibration(
        image_points=(
            (350.0, 150.0),  # far-left  (compressed by perspective)
            (650.0, 150.0),  # far-right
            (900.0, 500.0),  # near-right
            (100.0, 500.0),  # near-left
        ),
        world_points=(
            (0.0, 20.0),
            (10.0, 20.0),
            (10.0, 0.0),
            (0.0, 0.0),
        ),
        name="synthetic_perspective",
        description=(
            "Synthetic trapezoidal floor view, 10m wide by 20m deep. A "
            "mathematical example only: no camera, no survey, no accuracy claim."
        ),
    )
