# Benchmark records

Durable, dated records of benchmarks that were actually run, one file per
result. A record lands here only when it was measured; nothing in this
directory is projected, extrapolated or estimated.

| Record | Milestone | What it measures | Status |
| --- | --- | --- | --- |
| [`m08-amd-mi300x-fp32`](m08-amd-mi300x-fp32.md) ([json](m08-amd-mi300x-fp32.json)) | M0.8 | Perception throughput and correctness parity, CPU vs AMD Instinct MI300X, FP32 | measured on real hardware |

## Conventions

Each record is a pair: a `.md` for people and a `.json` for machines, carrying
the same numbers. Every record states, explicitly:

* the **precision** it was run at — a throughput figure without one is
  meaningless;
* that **model load time is excluded** from throughput, and that warmup frames
  were discarded;
* that the compared devices ran the **same workload** — same weights,
  resolution, thresholds, frame count and clip;
* which metadata was **not recorded**, left blank rather than filled in;
* what the number **is not**, in particular that a single clip on a single host
  is not a general real-world performance claim.

Videos are never committed. A record names the clip it used and, where the clip
is synthetic, how it was generated (`scripts/make_benchmark_clip.py`).

The harness itself emits a complete machine-readable record, including host
metadata and per-stage percentiles:

```bash
python -m app.benchmark --video clip.mp4 --compare cpu,rocm --parity --json
```

That output is the authoritative form; a transcribed record says so and marks
the fields it lacks.
