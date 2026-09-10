# Sentinel

**Predict the incident. Prevent the outcome.**

Sentinel is a multimodal predictive incident-intelligence system.

## Current milestone: M0.2 — Temporal Event Memory

```
video → detection → tracking → structured events → temporal memory (queryable)
```

**M0.1** delivered perception: video in, detections tracked with persistent IDs,
a structured event log out.

**M0.2** adds the layer Sentinel will actually reason over. Frame-level tracking
output is folded into a persistent, chronological memory: every entity has a
timeline, every timeline is queryable, and state at any past moment can be
reconstructed. The event vocabulary grows to cover dwelling (`stationary`) and
image-space regions (`entered_zone` / `exited_zone`).

There is still no UI, no LLM reasoning, no risk engine and no audio — those are
later milestones, and the interfaces here are shaped so they can be added
without a rewrite.

---

## Architecture

```
app/
├── config.py                        PipelineConfig — every knob in one place
├── pipeline.py                      wiring only: frames → detect → track → events → memory
├── main.py                          CLI entry point
├── spatial.py                       pixel-space geometry + Zone / ZoneSet
│
├── perception/                      ── LAYER 1: what is on screen now
│   ├── types.py                     Detection, Track, Frame  (framework-free)
│   ├── video.py                     VideoSource — decoding + real timestamps
│   ├── detector.py                  Detector ABC + backend registry
│   ├── backends/
│   │   ├── yolo_ultralytics.py      YOLOv8  ← the ONLY file importing torch
│   │   ├── blob.py                  OpenCV colour-blob detector (no weights)
│   │   └── mock.py                  scripted/synthetic detector (tests)
│   ├── tracker.py                   Tracker ABC + strategy registry
│   └── trackers/byte_iou.py         ByteTrack-style IoU tracker
│
├── events/                          ── LAYER 2: what is worth recording
│   ├── schema.py                    Event / EventLog + action vocabulary
│   └── generator.py                 track state → structured events
│
├── memory/                          ── LAYER 3: what has happened (M0.2)
│   ├── temporal.py                  TemporalEventMemory, EntityTimeline, EntityState
│   └── query.py                     EventFilter — composable query criteria
│
├── storage/memory.py                flat event log + JSON persistence
└── reasoning/                       ── LAYER 4: what it means (NOT YET BUILT)
```

Two rules make the rest of the roadmap possible, and both are enforced by
`tests/test_memory_layering.py` rather than by convention:

1. **Model-specific code never escapes `perception/backends/`.** `pipeline.py`
   talks to the `Detector` and `Tracker` interfaces, so moving inference to AMD
   ROCm — or to ONNX Runtime, or to a remote service — is a new file in
   `backends/`, not a refactor.
2. **`app/memory/` imports only `app.events.schema` and `app.spatial`.** It
   never imports perception, torch, OpenCV or numpy. A temporal memory that
   knew what a YOLO tensor looked like could not survive a detector change —
   and the reasoning layer will sit on top of `memory/`, not on top of
   `perception/`.

Data flows strictly downward. Layer 3 consumes `Event` objects and has no idea
a camera was involved.

### Temporal memory (M0.2)

`TemporalEventMemory` is a chronological index over the event stream. It holds
the same events as the flat log; the difference is what you can ask it.

| Query | Method |
| --- | --- |
| Chronological event stream | `memory.stream()` |
| Events for one entity | `memory.entity_history("person_1")` |
| Events in a time interval | `memory.between(2.0, 5.0)` — `[start, end)` |
| Latest state of an entity | `memory.latest_state("person_1")` |
| State at a past moment | `memory.state_at("person_1", at=3.5)` |
| Entities currently present | `memory.present_entities()` (or `at=3.5`) |
| Who is inside a zone | `memory.zone_occupancy("loading_bay")` |
| One entity's timeline + state | `memory.timeline("person_1")` |

Every query accepts the same criteria (`entity_id`, `actions`, `class_names`,
`zone`, `track_id`, `start`, `end`, `min_confidence`), so they compose:

```python
memory.between(2.0, 5.0, entity_id="person_1", actions=["moved"])
```

Two design decisions worth knowing:

- **State is derived, never authored.** `EntityState` is a fold over an
  entity's events. That is what makes `state_at()` possible and what keeps
  out-of-order ingestion correct — a replayed or merged log still produces the
  right answer.
- **Insertion preserves chronology.** Events may arrive out of order; the
  stream always reads chronologically, ordered by
  `(timestamp, event_id, arrival)`.

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

# Watch image-space zones, and print the resulting timeline
python -m app.main clip.mp4 --zone loading_bay=380,180,640,320 --timeline

# Also write the temporal memory (events + folded per-entity state)
python -m app.main clip.mp4 --memory-output data/memory.json

# No weights available? Run the whole pipeline on a synthetic clip
python scripts/generate_demo_video.py -o data/demo/demo.mp4
python -m app.main data/demo/demo.mp4 --detector blob --classes person truck car
```

Run `python -m app.main --help` for the full list.

### Using it as a library

```python
from app.config import PipelineConfig
from app.pipeline import run_pipeline
from app.spatial import Zone, ZoneSet

result = run_pipeline(
    PipelineConfig(
        video_path="clip.mp4",
        output_path="events.json",
        zones=ZoneSet([Zone.from_rect("loading_bay", (380, 180, 640, 320))]),
    )
)

memory = result.temporal            # the M0.2 temporal event memory

print(memory.summary())
print(memory.entities())            # ['person_1', 'truck_2', ...]
print(memory.present_entities(at=3.5))
print(memory.zone_occupancy("loading_bay", at=3.5))

for event in memory.entity_history("person_1"):
    print(event.timestamp, event.action, event.position)   # pixels!

state = memory.latest_state("person_1")
print(state.class_name, state.duration, state.zones, state.present)
```

`result.memory` remains the flat event log (what gets written to
`events.json`); `result.temporal` is the queryable index over the same events.

---

## Event format

One JSON document per run:

```json
{
  "schema_version": "0.2",
  "coordinate_space": "image_pixels",
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
      "event_id": "evt_000004",
      "timestamp": 1.25,
      "frame_index": 25,
      "entity_id": "person_1",
      "track_id": 1,
      "action": "entered_zone",
      "position": [151.17, 397.99],
      "bbox": [48.7, 235.5, 253.6, 560.4],
      "coordinate_space": "image_pixels",
      "zones": ["loading_bay"],
      "attributes": {
        "class_name": "person",
        "class_id": 0,
        "confidence": 0.9019,
        "track_age": 25,
        "zone": "loading_bay"
      }
    }
  ]
}
```

`entity_id` is the persistent identity: every event about `person_1` refers to
the same physical object for as long as the tracker can follow it.

`coordinate_space` is stated on every event, and it is always `image_pixels`.
`position` and `bbox` are pixel coordinates in the decoded frame — read the
[known limitations](#coordinates-are-pixels-and-pixels-are-not-metres) before
computing anything from them.

`--memory-output` writes a second document with the same events plus a folded
`entities` map — each entity's class, latest position, zones, first/last seen,
duration and presence.

### Actions

| Action | Meaning |
| --- | --- |
| `appeared` | First frame in which the tracker confirmed this entity. |
| `moved` | The entity travelled more than `--movement-threshold` pixels since its last motion anchor. |
| `stationary` | The entity stayed within `--stationary-threshold` pixels for `--stationary-duration` seconds. Fires **once per stationary episode**, and re-arms after the entity moves again. |
| `entered_zone` | The entity's anchor point crossed into a configured zone. Names the zone in `attributes.zone`. |
| `exited_zone` | The entity's anchor point left a zone. |
| `detected` | Heartbeat: re-states the entity's state every `--sample-interval` seconds when it isn't moving or dwelling. |
| `disappeared` | The tracker gave up on the entity, or the video ended. Carries `visible_for_seconds`. |

A per-frame dump would be enormous and mostly redundant. These actions keep the
log small enough to hand to a reasoning layer later while still recording when
each entity started, moved, settled and stopped.

Two boundary behaviours that matter for correctness:

- An entity that **appears already inside a zone** gets an `entered_zone` event
  (marked `on_appearance: true`). Without it, occupancy reconstructed from the
  event stream would silently miss it.
- An entity that **disappears while inside a zone** gets an `exited_zone` event
  (marked `on_disappearance: true`) *before* its `disappeared` event. Without
  it, zone occupancy would leak entities that are long gone.

### Zones

A zone is a named polygon **in the image**, configured per camera framing:

```bash
# inline, repeatable
python -m app.main clip.mp4 --zone loading_bay=380,180,640,320 --zone walkway=0,240,380,384

# or from a file — see data/zones.example.json
python -m app.main clip.mp4 --zones data/zones.example.json
```

Membership is tested against one **anchor point** of the bounding box, not the
whole box. The default anchor is `bottom_center` — the mid-point of the box's
lower edge — because for an upright object that is the closest cheap proxy for
where it touches the ground. Using the centroid would put a tall person "in"
a floor zone they are merely standing next to. `anchor: "center"` is available
per zone. Zones may overlap; an entity in three zones emits three
`entered_zone` events.

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

The suite runs with **no model weights, no GPU and no network**. Temporal
memory tests drive scripted synthetic detections through the *real* tracker,
generator and memory, so they are exactly reproducible; the `blob` backend
drives a genuine end-to-end run (real video decoding, real tracking, real JSON
output) against a synthetic clip whose ground truth is known. Tests needing
OpenCV skip cleanly if it isn't installed.

`tests/test_memory_layering.py` parses the source tree and fails the build if
`app/memory/` ever imports a model library or the perception layer. The
architectural boundary is checked, not just documented.

---

## Known limitations

These are real constraints of the current build, not TODO placeholders.

### Coordinates are pixels, and pixels are not metres

Nothing in Sentinel knows the camera's intrinsics, height, tilt, or the scene's
ground plane. Therefore:

- A pixel distance is **not** a real-world distance. 50px near the camera may
  be centimetres; 50px at the horizon may be tens of metres.
- `--movement-threshold` and `--stationary-threshold` are pixel values whose
  meaning changes with resolution, zoom and how far the subject is from the
  camera. A single threshold cannot be correct across the whole frame.
- `speed_px_per_frame` is pixel speed, not physical speed.
- Zone geometry is tied to one exact camera framing. Move or re-zoom the
  camera and every zone must be redrawn.

Every event carries `coordinate_space: "image_pixels"` and pixel-valued config
keys are suffixed `_px`, so the day world coordinates arrive it is an explicit,
greppable migration rather than a silent change of units. Real-world units
require a homography or camera calibration, which is **not** in scope.

### Tracking and identity

- **No re-identification.** An entity gone longer than `--track-max-age`
  returns with a new ID and a new timeline. Occlusion shorter than that is
  handled; a person leaving and re-entering the frame is two entities.
- **No cross-camera identity.** One video, one namespace of entity IDs.
- **Entity IDs are per-run.** `person_1` in two runs is two different people.

### Events and memory

- **`stationary` means "did not move in pixel space".** A person shifting
  weight on the spot and a parked vehicle are indistinguishable to it.
- **Zone membership is a single point test.** An entity straddling a boundary
  is in or out by its anchor point alone; there is no partial overlap or
  dwell-based hysteresis, so an entity oscillating on a boundary line will emit
  repeated enter/exit pairs.
- **Memory is per-run and in-process.** It persists to JSON, but there is no
  database, no indexing across runs, and the whole history is held in RAM.
  Fine for clips; not yet for a continuous camera feed.
- **`present_entities(at=...)` and `state_at()` replay an entity's history**,
  so they are O(events) per call rather than indexed.

### Not yet built

Frontend, LLM reasoning, risk scoring, audio, autonomous agents, counterfactual
simulation. `app/reasoning/` is still the M0 stub.

---

## Scope

**In M0.1:** video ingestion, detection, tracking, persistent IDs, structured
events, JSON output, tests.

**In M0.2:** temporal event memory, per-entity timelines, `stationary` and zone
events, the query API, explicit pixel-space coordinates, layering enforced by
test.

**Not yet:** frontend, LLM reasoning, risk scoring, audio, agents,
counterfactual simulation.
