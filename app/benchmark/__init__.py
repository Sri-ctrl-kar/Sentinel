"""Perception benchmarking (M0.8).

Measures the workload Sentinel really runs, on whichever device it is pointed
at, and — just as importantly — checks that the answer does not change when the
device does.

    python -m app.benchmark --info
    python -m app.benchmark --video clip.mp4 --device cpu
    python -m app.benchmark --video clip.mp4 --device rocm
    python -m app.benchmark --video clip.mp4 --compare cpu,rocm --parity

This package is a consumer of the pipeline, never a dependency of it: nothing
in ``app/perception``, ``app/reasoning`` or ``app/intelligence`` imports it.
"""

from __future__ import annotations

from .compare import (
    CONFIDENCE_TOLERANCE,
    COUNT_TOLERANCE,
    RISK_TOLERANCE,
    ParityDifference,
    ParityReport,
    Speedup,
    compare_risk,
    compare_workloads,
    speedup,
)
from .harness import (
    BENCHMARK_BANNER,
    DEFAULT_FRAMES,
    DEFAULT_WARMUP_FRAMES,
    BenchmarkConfig,
    BenchmarkError,
    BenchmarkResult,
    WorkloadSummary,
    run_benchmark,
)
from .timing import LatencySummary, Stopwatch, percentile, summarise

__all__ = [
    "BENCHMARK_BANNER",
    "CONFIDENCE_TOLERANCE",
    "COUNT_TOLERANCE",
    "DEFAULT_FRAMES",
    "DEFAULT_WARMUP_FRAMES",
    "RISK_TOLERANCE",
    "BenchmarkConfig",
    "BenchmarkError",
    "BenchmarkResult",
    "LatencySummary",
    "ParityDifference",
    "ParityReport",
    "Speedup",
    "Stopwatch",
    "WorkloadSummary",
    "compare_risk",
    "compare_workloads",
    "percentile",
    "run_benchmark",
    "speedup",
    "summarise",
]
