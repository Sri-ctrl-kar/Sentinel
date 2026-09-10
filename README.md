# Sentinel

**Predict the incident. Prevent the outcome.**

Sentinel is a multimodal predictive incident-intelligence system.

## Current milestone: M0.1 — Perception → Events

```
video → object detection → object tracking → structured events (JSON)
```

M0.1 delivers the perception layer only. It ingests a local video file, detects
objects frame by frame, tracks them across frames with persistent IDs, and
writes a structured event log to JSON. There is no UI, no reasoning layer, no
risk engine and no audio yet — those are later milestones, and the interfaces
here are shaped so they can be added without a rewrite.

---

## Architecture

```
app/
├── config.py                        PipelineConfig — every knob in one place
├── pipeline.py                      wiring only: frames → detect → track → events
├── main.py                          CLI entry point
├── perception/
│   ├── types.py                     Detection, Track, Frame  (framework-free)
│   ├── video.py                     VideoSource — decoding + real timestamps
│   ├── detector.py                  Detector ABC + backend registry
│   ├── backends/
│   │   ├── yolo_ultralytics.py      YOLOv8  ← the ONLY file importing torch
│   │   ├── blob.py                  OpenCV colour-blob detector (no weights)
│   │   └── mock.py                  scripted/synthetic detector (tests)
│   ├── tracker.py                   Tracker ABC + strategy registry
│   └── trackers/byte_iou.py         ByteTrack-style IoU tracker
├── events/
│   ├── schema.py                    Event / EventLog + action vocabulary
│   └── generator.py                 track state → structured events
└── storage/memory.py                event store + JSON persistence
```

The rule that makes the rest of the roadmap possible: **model-specific code
never escapes `perception/backends/`**. `pipeline.py` talks to the `Detector`
and `Tracker` interfaces, so moving inference to AMD ROCm — or to ONNX Runtime,
or to a remote inference service — is a new file in `backends/`, not a
refactor.

### Model choices

| Layer | Choice | Why |
| --- | --- | --- |
| Detection | **YOLOv8n** (Ultralytics) | Real-time on CPU, faster still on GPU; COCO classes already cover people, vehicles and equipment; exports cleanly to ONNX for the ROCm path. |
| Tracking | **ByteTrack-style IoU association** | Two-stage matching recovers objects through partial occlusion, so an entity keeps its ID when it briefly disappears — which is the whole point of tracking for incident prediction. |

The tracker is implemented in-repo (`app/perception/trackers/byte_iou.py`,
numpy-free, ~300 lines): no ReID network on the critical path, no extra
dependency, and fully unit-testable without model weights. It keeps ByteTrack's
two-stage association and a constant-velocity motion model, substituting an
exponentially smoothed linear predictor for the Kalman filter.

---

## Setup

Python 3.9+.

### Minimal install (no model weights, no torch)

Enough to run the pipeline with the `blob` and `mock` backends and to run the
whole test suite:

```bash
pip install -r requirements-dev.txt
```

### With the YOLOv8 detector

Install torch **first**, picking the build that matches your hardware:

```bash
# AMD ROCm 6.2
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# NVIDIA CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then:

```bash
pip install -r requirements-yolo.txt
```

`yolov8n.pt` is downloaded automatically on first run.

---

## Usage

```bash
python -m app.main path/to/video.mp4 -o data/events.json
```

Common options:

```bash
# Faster pass: every 3rd frame, small model, 480px inference
python -m app.main clip.mp4 --stride 3 --imgsz 480

# Only care about people and vehicles
python -m app.main clip.mp4 --classes person car truck bus forklift

# Force a device (ROCm reports itself as "cuda" — see below)
python -m app.main clip.mp4 --device cuda --half

# No weights available? Run the whole pipeline on a synthetic clip
python scripts/generate_demo_video.py -o data/demo/demo.mp4
python -m app.main data/demo/demo.mp4 --detector blob --classes person truck car
```

Run `python -m app.main --help` for the full list.

### Using it as a library

```python
from app.config import PipelineConfig
from app.pipeline import run_pipeline

result = run_pipeline(PipelineConfig(video_path="clip.mp4", output_path="events.json"))

print(result.memory.summary())
for event in result.memory.by_entity("person_1"):
    print(event.timestamp, event.action, event.position)
```

---

## Event format

One JSON document per run:

```json
{
  "schema_version": "0.1",
  "metadata": {
    "video": { "path": "clip.mp4", "width": 960, "height": 540, "fps": 20.0, "frame_count": 60 },
    "detector": { "backend": "ultralytics-yolo", "model": "yolov8n.pt", "device": "cuda", "accelerator": "rocm" },
    "tracker": "byte_iou",
    "frames_processed": 60,
    "summary": { "total_events": 27, "unique_entities": 5 }
  },
  "event_count": 27,
  "events": [
    {
      "event_id": "evt_000001",
      "timestamp": 0.05,
      "frame_index": 0,
      "entity_id": "person_1",
      "track_id": 1,
      "action": "appeared",
      "position": [151.17, 397.99],
      "bbox": [48.7, 235.5, 253.6, 560.4],
      "attributes": {
        "class_name": "person",
        "class_id": 0,
        "confidence": 0.9019,
        "track_age": 1,
        "first_seen": 0.05
      }
    }
  ]
}
```

`entity_id` is the persistent identity: every event about `person_1` refers to
the same physical object for as long as the tracker can follow it.

### Actions

| Action | Meaning |
| --- | --- |
| `appeared` | First frame in which the tracker confirmed this entity. |
| `moved` | The entity travelled more than `--movement-threshold` pixels since its last motion anchor. |
| `detected` | Heartbeat: re-states the entity's state every `--sample-interval` seconds when it isn't moving. |
| `disappeared` | The tracker gave up on the entity, or the video ended. Carries `visible_for_seconds`. |

A per-frame dump would be enormous and mostly redundant. These four actions
keep the log small enough to hand to a reasoning layer later while still
recording when each entity started, moved and stopped.

---

## AMD / ROCm notes

- **`torch.cuda` *is* the ROCm API.** On a ROCm build, `torch.cuda.is_available()`
  returns `True` and the device string is still `"cuda"`. `--device auto`
  therefore resolves correctly on AMD with no branching. Sentinel inspects
  `torch.version.hip` only to *report* which stack is live — the event log's
  `metadata.detector.accelerator` reads `rocm`, `cuda`, `mps` or `cpu`.
- **Install is the only real difference.** Use the ROCm wheel index above;
  `requirements-yolo.txt` deliberately leaves torch unpinned so pip cannot
  silently replace a ROCm build with the default PyPI (CUDA) one.
- **FP16 is a flag, not a code path.** `--half` enables FP16 inference, which
  MI-series CDNA hardware handles natively. It is ignored on CPU.
- **No CUDA-only code anywhere.** No custom kernels, no TensorRT, no pycuda.
- **MIGraphX / ONNX Runtime is the next step, not a rewrite.** The `Detector`
  ABC returns plain `Detection` objects, so an ONNX-EP backend is a new file in
  `app/perception/backends/`. Export with
  `yolo export model=yolov8n.pt format=onnx`.

---

## Tests

```bash
python -m pytest tests/ -q
```

The suite runs with **no model weights, no GPU and no network**: the `mock`
backend covers tracking and event generation, and the `blob` backend drives a
genuine end-to-end run (real video decoding, real tracking, real JSON output)
against a synthetic clip whose ground truth is known. Tests needing OpenCV skip
cleanly if it isn't installed.

---

## Scope

**In M0.1:** video ingestion, detection, tracking, persistent IDs, structured
events, JSON output, tests.

**Not yet:** frontend, LLM reasoning, risk scoring, audio, agents,
counterfactual simulation.
