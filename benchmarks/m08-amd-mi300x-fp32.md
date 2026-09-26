# M0.8 — CPU vs AMD Instinct MI300X, FP32

**Status: measured on real AMD hardware.** Machine-readable form:
[`m08-amd-mi300x-fp32.json`](m08-amd-mi300x-fp32.json).

Sentinel's perception workload — the same YOLOv8n model, the same clip, the
same thresholds — run on a CPU and on an AMD Instinct MI300X, with the
deterministic verdict compared afterwards.

## Workload (identical on both devices)

| | |
| --- | --- |
| Model | YOLOv8n |
| **Precision** | **FP32** (no FP16, no quantization) |
| Frames measured | 120 |
| Warmup frames | 5, run then discarded |
| Model load time | measured separately, **excluded from every throughput figure** |
| Video | `/tmp/sentinel_benchmark.mp4` (not committed to this repository) |
| Batching | none — one frame at a time, as a live camera delivers them |

The CPU and ROCm runs used the **same workload**: same weights, same input
resolution, same confidence and IoU thresholds, same class filter, same frame
count, same warmup, same clip. Only the device differed.

## Hardware and runtime

| | |
| --- | --- |
| Accelerator | AMD Instinct MI300X VF |
| Reported as | `AMD ROCm / HIP` (never as CUDA) |
| ROCm / HIP runtime | 7.1.52802 |
| CPU, RAM, OS, Python, PyTorch, Ultralytics versions | not recorded in this report |

Fields the report did not carry are left blank rather than filled in.
`python -m app.benchmark … --json` emits the harness's own complete metadata
(OS, Python, torch, Ultralytics, HIP version, GPU name, CPU, RAM, weights
SHA-256, per-stage percentiles); dropping that file in beside this one would
supersede the gaps above.

## Throughput

| | CPU | AMD MI300X | speedup |
| --- | --- | --- | --- |
| Inference (detect) | **76.03 FPS** | **147.14 FPS** | **1.94×** |
| Pipeline (end to end) | **73.26 FPS** | **138.45 FPS** | **1.89×** |

`inference` is one `detector.detect(frame)` call — the forward pass plus
letterboxing, tensor transfer, NMS and box decoding. `pipeline` is decode →
detect → track → generate events, per frame. Model load is in neither.

## Correctness parity

| | CPU | AMD MI300X |
| --- | --- | --- |
| Detections | 320 | 320 |
| Tracks | 9 | 9 |
| Events | 38 | 38 |
| Mean confidence | 0.6245 | 0.6245 |
| Risk score | 47.44 | 47.44 |
| Severity | medium | medium |

```
correctness parity      : PASS
risk semantics unchanged: YES
```

The deterministic verdict is identical on both devices. That is the result this
milestone was for: the device changed and Sentinel's meaning did not.

## What this is not

* **Not a general real-world performance claim.** One clip, one host, one
  model, one resolution, single stream. It measures this workload on this
  machine; it does not predict what any other deployment will see.
* **Not an FP16 result.** FP32 throughout. No FP16, INT8, quantization,
  batching, ONNX Runtime, MIGraphX or TensorRT was used or measured — see
  *What was NOT optimized* in the repository README.
* **Not a bare-metal number.** The MI300X was a virtual-function (VF)
  instance.

## Reproducing

```bash
python -m app.benchmark --info    # must report HIP available: yes
python -m app.benchmark --video /path/to/clip.mp4 --compare cpu,rocm --parity
```
