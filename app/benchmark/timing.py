"""Latency aggregation.

A mean throughput figure hides the thing that matters for a safety system: the
frame that took four times as long as the others. So every stage is summarised
with percentiles, and the percentile rule is stated rather than assumed.

Percentile method: **nearest-rank on the sorted samples**. For ``n`` samples,
the ``p``-th percentile is the value at index ``ceil(p/100 * n) - 1``. It
returns an actually-observed measurement (no interpolation between frames,
which would report a latency that never happened), and it is what the common
"p95" in service monitoring means. Numpy's default linear interpolation would
give slightly different numbers; this module deliberately does not depend on
numpy at all.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

#: Percentiles reported for every stage.
REPORTED_PERCENTILES = (50, 95, 99)


def percentile(samples: Sequence[float], p: float) -> float:
    """Nearest-rank percentile of ``samples``.

    ``p`` is in ``[0, 100]``. Raises on an empty sequence: a percentile of
    nothing is not zero, it is undefined, and returning zero would silently
    flatter a run that produced no frames.
    """
    if not samples:
        raise ValueError("percentile of an empty sample set is undefined")
    if not 0 <= p <= 100:
        raise ValueError(f"percentile must be within [0, 100], got {p}")
    ordered = sorted(samples)
    if p == 0:
        return ordered[0]
    rank = math.ceil(p / 100.0 * len(ordered))
    return ordered[max(1, rank) - 1]


@dataclass(frozen=True)
class LatencySummary:
    """One stage's per-frame latency, in milliseconds."""

    label: str
    count: int
    total_seconds: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float

    @property
    def fps(self) -> float:
        """Throughput implied by this stage alone."""
        return self.count / self.total_seconds if self.total_seconds > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "frames": self.count,
            "total_seconds": round(self.total_seconds, 4),
            "fps": round(self.fps, 2),
            "mean_ms": round(self.mean_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "stdev_ms": round(self.stdev_ms, 3),
        }

    def line(self) -> str:
        return (
            f"{self.label:<22} {self.fps:>8.2f} fps  "
            f"mean {self.mean_ms:>8.2f} ms  p50 {self.p50_ms:>8.2f}  "
            f"p95 {self.p95_ms:>8.2f}  p99 {self.p99_ms:>8.2f}"
        )


def summarise(label: str, samples: Sequence[float]) -> LatencySummary:
    """Summarise per-frame durations (in **seconds**) as a latency profile."""
    if not samples:
        return LatencySummary(label, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    milliseconds = [s * 1000.0 for s in samples]
    return LatencySummary(
        label=label,
        count=len(samples),
        total_seconds=float(sum(samples)),
        mean_ms=statistics.fmean(milliseconds),
        p50_ms=percentile(milliseconds, 50),
        p95_ms=percentile(milliseconds, 95),
        p99_ms=percentile(milliseconds, 99),
        min_ms=min(milliseconds),
        max_ms=max(milliseconds),
        stdev_ms=statistics.stdev(milliseconds) if len(milliseconds) > 1 else 0.0,
    )


class Stopwatch:
    """Collects per-frame durations for one stage.

    Uses :func:`time.perf_counter`, the only clock here that is monotonic and
    high-resolution. Wall-clock time would drift; process time would hide the
    GPU wait that is precisely what a GPU benchmark measures.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.samples: List[float] = []
        self._started: Optional[float] = None

    def __enter__(self) -> "Stopwatch":
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self._started is not None:
            self.samples.append(time.perf_counter() - self._started)
            self._started = None

    def record(self, seconds: float) -> None:
        self.samples.append(float(seconds))

    def discard(self, count: int) -> None:
        """Drop the first ``count`` samples — warmup, not measurement."""
        del self.samples[:count]

    def summary(self) -> LatencySummary:
        return summarise(self.label, self.samples)
