# M0.8 — local Apple Silicon CPU, FP32

**Status: measured on a local Apple Silicon Mac, CPU only.** Recorded
2026-11-06. Machine-readable form:
[`m08-apple-silicon-cpu-fp32-2026-11-06.json`](m08-apple-silicon-cpu-fp32-2026-11-06.json).

Sentinel's perception workload — decode → detect → track → generate events —
run on the CPU of an Apple Silicon Mac over a screen-recording clip.

**One device.** No second device was benchmarked, so this record contains no
speedup, no parity check and no comparison of any kind. It says nothing about
AMD ROCm, about any GPU, or about the separate
[MI300X record](m08-amd-mi300x-fp32.md), which is a different run on different
hardware and is not to be read against these numbers.

## Workload

| | |
| --- | --- |
| Model | YOLOv8n (`yolov8n.pt`, sha256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`) |
| Detector backend | `yolo` (Ultralytics) |
| **Precision** | **FP32** (`half: false` — no FP16, no quantization) |
| Input size | imgsz 640, confidence 0.25, IoU 0.45, no class filter |
| Frames measured | 120, at stride 1 |
| Warmup frames | 5, run then discarded |
| Model load time | 0.089 s, **excluded from every throughput figure below** |
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
| Measured | the first 120 frames at stride 1 |

Only 120 of the clip's 2105 frames were measured, so these figures describe the
opening seconds, not the whole recording. The clip's absolute path was a home
directory and is omitted; the file name is kept.

## Throughput

Model load excluded. Percentiles are over the 120 measured frames.

| stage | FPS | mean | p50 | p95 | p99 | min | max | stdev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inference (detect) | **55.01** | 18.18 ms | 18.102 | 19.123 | 19.486 | 17.055 | 19.573 | 0.538 |
| pipeline (end to end) | **50.45** | 19.82 ms | 19.755 | 20.726 | 20.958 | 18.706 | 21.589 | 0.553 |
| decode | 621.08 | 1.61 ms | 1.568 | 1.947 | 2.119 | 1.405 | 2.321 | 0.166 |
| tracking | 56548.08 | 0.018 ms | 0.017 | 0.025 | 0.027 | 0.012 | 0.031 | 0.003 |
| event generation | 82002.28 | 0.012 ms | 0.011 | 0.025 | 0.029 | 0.007 | 0.03 | 0.004 |

Totals measured: inference 2.1816 s, pipeline 2.3784 s, decode 0.1932 s,
tracking 0.0021 s, events 0.0015 s.

`inference` is one `detector.detect(frame)` call — the forward pass plus
letterboxing, tensor transfer, NMS and box decoding. `pipeline` is decode →
detect → track → generate events, per frame. Model load is in neither.
Detection takes 18.18 ms of the 19.82 ms frame budget; tracking and event
generation together take 0.030 ms.

## What the run produced

| | |
| --- | --- |
| Detections | 213 |
| Tracks created | 2 |
| Events | 7 |
| Mean detection confidence | 0.6172 |
| Class counts | person 85, truck 120, skateboard 4, surfboard 4 |

These are what the detector reported on this clip. There is no ground truth for
this recording, so they are a record of what ran, not an accuracy score.

## What this is not

* **Not a general real-world performance claim.** One clip, one host, one
  model, one resolution, single stream, 120 frames. It measures this workload
  on this machine; it does not predict what any other deployment will see.
* **Not an accelerated result.** CPU only. No MPS, no ROCm, no CUDA, no FP16,
  no INT8, no quantization, no batching, no ONNX Runtime, MIGraphX or TensorRT.
* **Not a comparison.** A single device was benchmarked. No speedup and no
  parity verdict exist in this record, and none may be computed against the
  MI300X record: different clip, different host, different run.
* **Not a measurement of the whole clip.** 120 of 2105 frames.

## Reproducing

On the same machine, with the same clip:

```bash
python -m app.benchmark --info
python -m app.benchmark --video "Screen Recording 2026-09-29 at 09.48.17.mov" \
    --device cpu --json
```

`--device cpu` is load-bearing on a Mac: `--device auto` resolves to Metal
where it is available, which would measure something else.
