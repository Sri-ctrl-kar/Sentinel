# M0.8 — local Apple Silicon CPU, FP32, whole clip

**Status: measured on a local Apple Silicon Mac, CPU only.** Recorded
2026-11-06. Machine-readable form:
[`m08-apple-silicon-cpu-fp32-full-clip-2026-11-06.json`](m08-apple-silicon-cpu-fp32-full-clip-2026-11-06.json).

Sentinel's perception workload — decode → detect → track → generate events —
run on the CPU of an Apple Silicon Mac over the **whole** 2105-frame screen
recording.

**One device.** No second device was benchmarked, so this record contains no
speedup, no parity check and no cross-device comparison of any kind. It says
nothing about AMD ROCm, about any GPU, or about the separate
[MI300X record](m08-amd-mi300x-fp32.md), which is a different run on different
hardware and is not to be read against these numbers.

**Companion record.**
[`m08-apple-silicon-cpu-fp32-2026-11-06`](m08-apple-silicon-cpu-fp32-2026-11-06.md)
is the same machine, the same clip and the same settings over the harness
default window of 120 frames — the opening seconds. This record reads the whole
clip. Both stand on their own, neither supersedes the other, and no ratio
between them is computed here.

## Workload

| | |
| --- | --- |
| Model | YOLOv8n (`yolov8n.pt`, sha256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`) |
| Detector backend | `yolo` (Ultralytics) |
| **Precision** | **FP32** (`half: false` — no FP16, no quantization) |
| Input size | imgsz 640, confidence 0.25, IoU 0.45, no class filter |
| Frames measured | 2100, at stride 1 |
| Warmup frames | 5, run then discarded |
| Frames read in total | 2105 — the entire clip |
| Model load time | 0.092 s, **excluded from every throughput figure below** |
| Batching | none — one frame at a time, as a live camera delivers them |

## Device

| | |
| --- | --- |
| Requested | `cpu` |
| Ran on | CPU (`torch_device: cpu`), **execution verified** |
| Apple Metal (MPS) | **available on this host and deliberately not used** |
| HIP / ROCm / CUDA | none present |

The host has Metal. The run asked for the CPU anyway and the harness verified
that a tensor allocated there, so this is a CPU measurement on a machine that
could have accelerated it — not an accelerator result by another name.

## Host

| | |
| --- | --- |
| OS | Darwin 27.0.0 (arm64) |
| CPU | `arm`, 10 cores |
| RAM | 16.0 GiB |
| Python | 3.14.0 |
| PyTorch | 2.14.0 |
| Ultralytics | 8.4.163 |
| OpenCV | 5.0.0 |
| NumPy | 2.5.3 |
| Chip name, machine model | not recorded |

The harness reports the CPU architecture and core count, not the chip name or
the machine model. Neither is inferred here.

## Clip

| | |
| --- | --- |
| File | `Screen Recording 2026-09-29 at 09.48.17.mov` (not committed) |
| Resolution | 2900 × 1416 |
| Frame rate | 33.47 fps |
| Length | 2105 frames, 62.892 s |
| Measured | 2100 frames — the whole clip, less 5 warmup frames |

The clip's absolute path was a home directory and is omitted; the file name is
kept.

## Throughput

Model load excluded. Percentiles are over the 2100 measured frames.

| stage | FPS | mean | p50 | p95 | p99 | min | max | stdev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inference (detect) | **58.0** | 17.24 ms | 16.899 | 19.113 | 24.431 | 14.584 | 100.333 | 3.337 |
| pipeline (end to end) | **53.39** | 18.73 ms | 18.361 | 20.799 | 26.774 | 15.939 | 102.424 | 3.506 |
| decode | 679.62 | 1.471 ms | 1.415 | 1.716 | 2.216 | 1.219 | 13.877 | 0.393 |
| tracking | 92263.56 | 0.011 ms | 0.006 | 0.026 | 0.032 | 0.002 | 0.076 | 0.009 |
| event generation | 144702.8 | 0.007 ms | 0.001 | 0.023 | 0.03 | 0.001 | 0.059 | 0.008 |

Totals measured: inference 36.2044 s, pipeline 39.3325 s, decode 3.0899 s,
tracking 0.0228 s, events 0.0145 s.

`inference` is one `detector.detect(frame)` call — the forward pass plus
letterboxing, tensor transfer, NMS and box decoding. `pipeline` is decode →
detect → track → generate events, per frame. Model load is in neither.
Detection takes 17.24 ms of the 18.73 ms frame budget; tracking and event
generation together take 0.018 ms.

**The tail.** Over 2100 frames the distribution has a long tail the 120-frame
window never reached: inference p50 16.899 ms against a max of 100.333 ms, and
pipeline p50 18.361 ms against a max of 102.424 ms — roughly six times the
median on the worst frame. The harness captured no cause for it and none is
guessed at here. Read p50 and p95 for typical cost and max for the worst frame
observed.

## What the run produced

| | |
| --- | --- |
| Detections | 1760 |
| Tracks created | 26 |
| Events | 331 |
| Mean detection confidence | 0.51423 |

| class | count | class | count | class | count |
| --- | --- | --- | --- | --- | --- |
| person | 687 | truck | 654 | boat | 152 |
| motorcycle | 72 | surfboard | 58 | car | 41 |
| fire hydrant | 16 | suitcase | 14 | skateboard | 13 |
| dog | 12 | airplane | 10 | horse | 10 |
| bird | 9 | kite | 4 | bench | 3 |
| backpack | 2 | umbrella | 2 | train | 1 |

Eighteen classes, summing to 1760. These are what the detector reported on this
clip. There is no ground truth for this recording, so they are a record of what
ran, not an accuracy score.

## What this is not

* **Not a general real-world performance claim.** One clip, one host, one
  model, one resolution, single stream. It measures this workload on this
  machine; it does not predict what any other deployment will see.
* **Not an accelerated result.** CPU only. No MPS, no ROCm, no CUDA, no FP16,
  no INT8, no quantization, no batching, no ONNX Runtime, MIGraphX or TensorRT.
* **Not a comparison.** A single device was benchmarked. No speedup and no
  parity verdict exist in this record, and none may be computed against the
  MI300X record: different clip, different host, different run.
* **Not interchangeable with the 120-frame record.** Same machine and clip,
  different window — and the content the two windows cover differs.

## Reproducing

On the same machine, with the same clip:

```bash
python -m app.benchmark --info
python -m app.benchmark --video "Screen Recording 2026-09-29 at 09.48.17.mov" \
    --device cpu --frames 2100 --json
```

`--frames 2100` with the default 5 warmup frames reads 2105 — the whole clip.
Asking for 2105 would request 2110 and silently measure 2100 anyway.
`--device cpu` is load-bearing on a Mac: `--device auto` resolves to Metal
where it is available, which would measure something else.
