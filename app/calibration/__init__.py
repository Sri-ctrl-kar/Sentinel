"""Ground-plane spatial calibration.

Sits between memory and reasoning in Sentinel's pipeline::

    perception -> tracking -> events -> memory -> CALIBRATION -> reasoning

Its single job is converting image pixels into metres on the floor plane, and
being honest about when it cannot. Nothing here knows about detectors,
trackers, events or memory, and nothing here imports a model runtime — the
homography solver is pure Python for exactly that reason.

**Calibration is optional.** With none supplied the whole system stays in
``image_pixels`` and says so. There is no mode in which pixel coordinates are
quietly reinterpreted as metres.
"""

from .homography import DegenerateConfigurationError, Homography
from .planar import (
    DEFAULT_COLLINEARITY_TOLERANCE,
    MIN_CORRESPONDENCES,
    CalibrationQuality,
    GroundPlaneCalibration,
    GroundPoint,
)

__all__ = [
    "GroundPlaneCalibration",
    "GroundPoint",
    "CalibrationQuality",
    "Homography",
    "DegenerateConfigurationError",
    "MIN_CORRESPONDENCES",
    "DEFAULT_COLLINEARITY_TOLERANCE",
]
