# Benchmark records

Durable, dated records of benchmarks that were actually run, one file per
result. A record lands here only when it was measured; nothing in this
directory is projected, extrapolated or estimated.

| Record | Recorded | Milestone | What it measures | Status |
| --- | --- | --- | --- | --- |
| [`m08-apple-silicon-cpu-fp32-full-clip-2026-11-06`](m08-apple-silicon-cpu-fp32-full-clip-2026-11-06.md) ([json](m08-apple-silicon-cpu-fp32-full-clip-2026-11-06.json)) | 2026-11-06 | M0.8 | Perception throughput on a local Apple Silicon CPU, FP32, single device, **whole 2105-frame clip** | measured on a local Mac, CPU only — no comparison, no speedup |
| [`m08-apple-silicon-cpu-fp32-2026-11-06`](m08-apple-silicon-cpu-fp32-2026-11-06.md) ([json](m08-apple-silicon-cpu-fp32-2026-11-06.json)) | 2026-11-06 | M0.8 | Perception throughput on a local Apple Silicon CPU, FP32, single device, **first 120 frames** | measured on a local Mac, CPU only — no comparison, no speedup |
| [`m08-amd-mi300x-fp32`](m08-amd-mi300x-fp32.md) ([json](m08-amd-mi300x-fp32.json)) | 2026-09-26 | M0.8 | Perception throughput and correctness parity, CPU vs AMD Instinct MI300X, FP32 | measured on hardware that is no longer available — a past run, not a current result |

The two CPU records are the **same machine, the same clip and the same
settings over different windows** — the harness default of 120 frames, and the
whole 2105-frame recording. Neither supersedes the other and neither computes a
ratio against the other.

Neither is comparable to the **MI300X** record: different clip, different host,
different run. No speedup and no parity verdict may be computed across them,
and no record claims one.

**No AMD run is possible today.** The AMD cloud credits expired and the droplet
was deleted, so `AMD_BENCHMARK = NOT_RUN` (reason: no AMD ROCm device
available) and the MI300X record cannot be reproduced or refreshed. The most
recent benchmark run is the local Apple Silicon CPU run above, which reports
CPU figures only.

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
