"""Test-facing re-export of the tracking metrics.

They live in ``app/evaluation/tracking.py`` because the benchmark consumes them
too; this module keeps the M0.3.1 test imports unchanged.
"""

from app.evaluation.tracking import (  # noqa: F401
    TrackingMetrics,
    box,
    detection,
    evaluate,
    triangle_wave,
)
