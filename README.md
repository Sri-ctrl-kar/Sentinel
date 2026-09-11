# Sentinel

**Predict the incident. Prevent the outcome.**

Sentinel is a multimodal predictive incident-intelligence system.

## Current milestone: M0.3 — Predictive Risk Engine

```
video → detection → tracking → events → temporal memory → risk assessment → predicted incident
```

**M0.1** delivered perception: video in, detections tracked with persistent IDs,
a structured event log out.

**M0.2** added temporal memory: a persistent, chronological, queryable record of
what every entity has done.

**M0.3** is the first milestone that *reasons*. It reads the temporal memory and
assesses **developing situations** — a worker and a forklift converging, someone
standing where a machine works, a separation collapsing, a pattern of repeated
violations — producing a scored, explained, structured prediction before
anything has happened.

Try it with no video, no weights and no GPU:

```bash
python -m app.demo --timeline
```

There is still no UI, no LLM, no audio and no autonomous agents — those are
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
├── reasoning/                       ── LAYER 4: what it means (M0.3)
│   ├── risk_engine.py               RiskEngine — orchestration and aggregation
│   ├── config.py                    RiskConfig — every threshold, documented
│   ├── kinematics.py                image-space velocity + closest approach
│   ├── explainer.py                 renders assessments as readable analysis
│   ├── models/risk.py               RiskAssessment, FactorScore, RiskReport
│   └── factors/                     one narrow question each, independently testable
│       ├── proximity.py             ProximityFactor, ClosingSpeedFactor
│       ├── trajectory.py            TrajectoryFactor (the only predictor)
│       ├── zone.py                  ZoneFactor — person in a machine's zone
│       └── persistence.py           PersistenceFactor, EscalationFactor
│
├── storage/memory.py                flat event log + JSON persistence
└── demo.py                          synthetic warehouse scenario (no video needed)
```

Two rules make the rest of the roadmap possible, and both are enforced by
`tests/test_memory_layering.py` rather than by convention:

1. **Model-specific code never escapes `perception/backends/`.** `pipeline.py`
   talks to the `Detector` and `Tracker` interfaces, so moving inference to AMD
   ROCm — or to ONNX Runtime, or to a remote service — is a new file in
   `backends/`, not a refactor.
2. **`app/memory/` and `app/reasoning/` import only `app.events.schema` and
   `app.spatial`.** Neither imports perception, torch, OpenCV or numpy. A risk
   engine that knew what a YOLO tensor looked like could not survive a detector
   change. The layering test spawns a clean interpreter and asserts that
   `import app.reasoning` does not load a model runtime at all.

Data flows strictly downward. Layers 3 and 4 consume `Event` objects and have
no idea a camera was involved. Geometry is **not** duplicated: the risk engine
reuses `app/spatial.py` for distance and zone containment, and that reuse is
itself asserted by a test.

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

### Risk model (M0.3)

The score is a weighted sum of independently computed factors, times a
corroboration multiplier. Nothing is learned, nothing is hidden:

```
base_points  = Σ over {proximity, closing_speed, trajectory, zone}
                   factor.score × factor.weight        →   0 .. 100
persistence  = persistence.score × 15                  →   0 ..  15
subtotal     = base_points + persistence               →   0 .. 115
multiplier   = escalation multiplier by signal count   →   1.00 .. 1.35
risk_score   = clamp(0, 100, subtotal × multiplier)
```

| Factor | Weight | Question it answers | Formula |
| --- | ---: | --- | --- |
| `proximity` | 30 | How close are they *now*? | Linear from 0 at `interaction_radius_px` (260) to 1 at `critical_radius_px` (70) |
| `closing_speed` | 25 | How fast is the gap shrinking? | Linear from 0 at 10px/s to 1 at `closing_speed_reference_px_per_s` (160) |
| `trajectory` | 25 | Will their paths actually conflict, and how soon? | `spatial × temporal` from the closest-approach solution |
| `zone` | 20 | Is a person where a machine works? | 0.60 unaccompanied → 1.00 with the vehicle in the same zone |
| `persistence` | 15 | Is this a repeated pattern? | `violations / 3`, capped at 1 |
| `escalation` | ×1.00–1.35 | Do independent signals agree? | Lookup by count of factors scoring ≥ 0.20 |

Three deliberate properties:

- **The four base weights sum to exactly 100**, so no single factor can reach
  `critical` alone. Proximity maxes out at 30 points — the "low" band. Two
  things being near each other is not an incident, and the arithmetic says so.
- **`trajectory` multiplies its spatial and temporal terms** rather than
  averaging them. Both must hold: a conflict 40 seconds out is not urgent, and
  an imminent closest approach that misses by 300px is not a conflict.
  Averaging would let either alone carry the factor — exactly the false
  positive to avoid.
- **Escalation amplifies, it cannot invent.** `0 × 1.35` is still `0`.
  Corroboration can promote a borderline situation across a severity band; it
  can never manufacture risk that no factor observed.

Severity bands: `normal` 0–19, `low` 20–39, `medium` 40–64, `high` 65–89,
`critical` 90–100. Wide at the bottom, narrow at the top — the difference
between 5 and 15 is noise, the difference between 85 and 95 is the difference
between "watch this" and "act now".

### Supported incident types

| Incident type | Detected when |
| --- | --- |
| `PERSON_VEHICLE_COLLISION_RISK` | Predicted closest approach falls inside `conflict_radius_px`, or a person is in an operating zone with the vehicle converging |
| `PERSON_IN_VEHICLE_OPERATING_ZONE` | A person's anchor point is inside a configured operating zone |
| `CONVERGING_TRAJECTORIES` | Paths converge but miss by more than the conflict radius — a near miss, deliberately *not* called a collision |
| `RAPID_SEPARATION_DECREASE` | The gap is closing fast with no predicted path conflict |
| `PERSISTENT_ZONE_VIOLATION` | The same entity repeatedly enters a watched zone |
| `ESCALATING_SITUATION` | Several independently weak signals corroborate |
| `CLOSE_PROXIMITY` | Entities are near each other but nothing is developing — informational, never a prediction |

### Example output

`python -m app.demo` — a worker walking into a forklift bay as the forklift
approaches, with no video file or model weights involved:

```
SENTINEL INCIDENT ANALYSIS
==========================

Risk: 93/100
Severity: CRITICAL
Incident: PERSON_VEHICLE_COLLISION_RISK
Confidence: 0.94
At: t=0.90s

Entities:
  person_1
  forklift_2

Evidence:
  - separation is shrinking at 350px/s in image space (saturates at 160px/s)
  - image-space trajectories converge to 10px in 0.9s
  - person_1 is inside operating zone forklift_bay; forklift_2 is in the same zone
  - person_1 entered forklift_bay once in the last 60s
  - 4 independent signals agree (closing_speed, trajectory, zone, persistence); escalating by ×1.30

Predicted time-to-risk: 0.9 seconds

Recommended intervention:
  slow/stop the vehicle and redirect the person out of its path

Scoring breakdown (score x weight = points):
  proximity        0.00 x  30.0 =   0.00  (confidence 0.00)
  closing_speed    1.00 x  25.0 =  25.00  (confidence 0.90)
  trajectory       0.85 x  25.0 =  21.19  (confidence 0.90)
  zone             1.00 x  20.0 =  20.00  (confidence 1.00)
  persistence      0.33 x  15.0 =   5.00  (confidence 1.00)
  subtotal                         71.19
  escalation      x1.30 (4 corroborating signals)
  RISK SCORE                       92.55

Coordinate space: image_pixels
  All distances are IMAGE PIXELS and all speeds are PIXELS PER SECOND.
  These are not physical distances or speeds; no camera calibration
  or ground-plane homography is applied. Times are real seconds.
```

Note what drives the score: proximity contributes **nothing** — the two are
still ~300px apart. The alert comes from motion and context, 0.9 seconds before
the predicted conflict. That is the difference between recording an incident and
predicting one.

`python -m app.demo --timeline` shows the situation building:

```
RISK DEVELOPMENT OVER TIME
--------------------------
    time    risk  severity   incident
    0.00     0.0  normal     -
    0.30    48.1  medium     PERSON_VEHICLE_COLLISION_RISK
    0.60    49.4  medium     PERSON_VEHICLE_COLLISION_RISK
    0.90    92.5  critical   PERSON_VEHICLE_COLLISION_RISK
    1.20   100.0  critical   PERSON_VEHICLE_COLLISION_RISK
    1.50   100.0  critical   PERSON_VEHICLE_COLLISION_RISK
```

The jump at 0.9s is the worker crossing into the bay: the zone factor engages
and a fourth signal joins the corroboration count. Before that, the engine had
already been flagging the collision course at medium risk for half a second.

### Why this is deterministic and explainable

- **No learned model, no randomness, no wall clock.** Every number comes from
  a documented formula over recorded events. Re-running the engine on the same
  memory at the same timestamp produces byte-identical JSON — asserted by test,
  including after re-ingesting the events in reverse order.
- **Every weight is named and justified** in `app/reasoning/config.py`, next to
  the reasoning for its default. There are no magic numbers inside factor
  implementations.
- **Every factor is individually inspectable.** A `FactorScore` carries its
  score, weight, contribution, confidence, a plain-English rationale, the raw
  measurements it used, and the event IDs that evidence it. The printed
  breakdown lets you re-add the score by hand.
- **Evidence is checkable.** Cited event IDs are asserted to exist in memory and
  to predate the assessment — the engine cannot cite the future.
- **Refusals are explicit.** When the engine cannot predict, it says so in a
  field (`insufficient_history`, `both_stationary`) rather than returning a
  confident-looking zero.

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

# Assess risk over a clip: report the worst moment in the footage
python -m app.main clip.mp4 --zone forklift_bay=400,200,800,500 \
    --operating-zone forklift_bay --risk

# No weights available? Run the whole pipeline on a synthetic clip
python scripts/generate_demo_video.py -o data/demo/demo.mp4
python -m app.main data/demo/demo.mp4 --detector blob --classes person truck car
```

### Risk demo (no video, no weights, no GPU)

```bash
python -m app.demo              # the moment Sentinel would have raised the alert
python -m app.demo --timeline   # how the risk score develops across the scene
python -m app.demo --json       # the structured assessment
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

And to reason over it:

```python
from app.reasoning import Explainer, RiskConfig, RiskEngine

engine = RiskEngine(RiskConfig(operating_zones=["loading_bay"], zones=zones))

report = engine.assess(memory, at=3.5)
if report.top:
    print(report.top.risk_score, report.top.severity, report.top.incident_type)
    print(Explainer().explain(report.top))

# Or scan a whole clip for its worst moment
worst = max(engine.assess_timeline(memory, step=0.5), key=lambda r: r.max_score)
```

The engine takes a `TemporalEventMemory` — nothing else. It never sees a frame,
a detection or a track.

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

### Fixed in M0.3.1 — tracker identity fragmentation

**The bug.** The tracker scored association on a single measurement,
`iou(predicted_box, detection)`, which made the motion model a *hard
constraint* rather than evidence. At a direction reversal the predicted box
sits `2 × step` from the detection while the last observed box sits only
`1 × step` away, so a correct match was vetoed by a wrong prediction — the
tracker performed *worse than having no motion model at all*. The orphaned
track then coasted further the wrong way each frame and could never recover,
minting a new ID at every reversal. Fragmentation began exactly where the
geometry says: once `2 × step` exceeded the object's own width.

This was found during M0.3 validation, where a worker pacing in and out of a
zone shattered into **7 identities**, hiding the repeated-violation pattern the
persistence factor exists to detect.

**The fix.** Association now considers three tiers of evidence, each consulted
only for pairs the previous tier left unmatched:

| Tier | Evidence | Handles |
| ---: | --- | --- |
| 0 | IoU against the **predicted** box | every ordinary frame; resolves crossings |
| 1 | IoU against the **last observed** box | direction reversals, where prediction points backwards |
| 2 | Normalised **centre distance** + size consistency | motion faster than the object's own size |

The ordering is the whole design: relaxation applies exactly where prediction
failed, and never where it succeeded. `max_age` was not touched, and no
appearance model was added.

Tiering rather than `max(predicted_iou, observed_iou)` matters, and the
difference is not theoretical — taking the maximum fixed the reversal but
introduced identity swaps at crossings, because one track's stale observed box
can overlap the *other* object's detection better than its own predicted box
overlaps its own. Consuming every confident predicted match first means the
ambiguous evidence is never reached while the motion model is still working.

Measured on the original reproduction (`tests/test_tracker_identity.py`):

| | Before | After |
| --- | ---: | ---: |
| Ground-truth entities | 2 | 2 |
| Track IDs created | 7 | **2** |
| ID switches | 5 | **0** |
| False merges | 0 | 0 |

Reversal is now clean at every step size from 5px to 80px on a 40px-wide box
(it previously broke at 20px), while crossing, parallel, occlusion and
beyond-`max_age` behaviour are unchanged.

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

`tests/test_tracker_identity.py` scores the tracker against ground truth using
`tests/tracking_metrics.py`, which reports track IDs created, ID switches and
false merges. Ground truth is used **only to grade the result** — the tracker
is handed nothing but detections, so a test can never flatter it by leaking the
answer.

`tests/test_memory_layering.py` parses the source tree and fails the build if
`app/memory/` or `app/reasoning/` ever imports a model library or the perception
layer, and spawns a clean interpreter to prove `import app.reasoning` loads no
model runtime. The architectural boundary is checked, not just documented.

The M0.3 scenarios (`tests/scenarios.py`) script detections frame by frame
through the real tracker, event generator, temporal memory and risk engine —
so the scores they assert are the ones the real stack produces, not fixtures:

| Scenario | Situation | Result |
| --- | --- | --- |
| A | Person and vehicle moving apart | 0.0 normal — no assessment reported |
| B | Converging, but on lines that miss | 50.4 medium `CONVERGING_TRAJECTORIES` |
| C | Trajectories intersect | 72.4 high `PERSON_VEHICLE_COLLISION_RISK`, 0.37s warning |
| D | Person in operating zone, vehicle approaching | 100.0 critical, 0.31s warning |
| E | Five weak signals corroborating | 79.2 high — every factor under 20 points alone |
| F | Both entities stationary | 19.7 normal `CLOSE_PROXIMITY`, trajectory refused |
| G | Insufficient history | 19.7 normal, trajectory refused, no prediction |

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
- **Crossing objects of the same class can still swap** if one of them is
  also mispredicted at the moment they overlap. The tiered association below
  resolves the ordinary crossing correctly, but overlap plus a bad prediction
  is genuinely ambiguous without appearance features.
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

### The risk engine (M0.3)

- **Thresholds are pixel values and must be re-tuned per camera.** The defaults
  assume a roughly 640-960px-wide frame with subjects at mid-depth. A different
  mounting height, focal length or resolution needs different numbers. There is
  no auto-calibration.
- **A single set of radii cannot be right across the whole frame.** Because
  pixels-per-metre varies with depth, `critical_radius_px = 70` is generous near
  the camera and far too small at the horizon. This is the single largest source
  of both false positives and false negatives.
- **Trajectory prediction assumes constant image velocity.** People change
  direction; vehicles turn; the camera may pan. The prediction horizon is capped
  at 6 seconds for this reason, and the engine refuses to predict at all without
  3 samples spanning 0.2s.
- **Weights are engineering judgement, not empirical.** They are principled and
  documented, but they have not been validated against real incident data,
  because no labelled dataset was in scope for M0.3. Treat the absolute score as
  a ranking signal, not a calibrated probability.
- **Only person-vehicle pairs and lone people are assessed.** Vehicle-vehicle
  conflicts, person-person interactions and static hazards are not modelled.
- **No occlusion reasoning.** Two entities may be far apart in image space while
  physically adjacent (one behind a rack), or overlap in the image while metres
  apart in depth.
- **Interventions are a static lookup table** keyed by incident type. The engine
  does not reason about what to do — deliberately, at M0.3.
- **Assessment is O(people x vehicles) per moment**, and each moment replays
  entity history. Fine for clips; a busy scene at a fine time step will be slow.

### Not yet built

Frontend, LLM reasoning, audio, autonomous agents, counterfactual simulation,
database persistence.

---

## Scope

**In M0.1:** video ingestion, detection, tracking, persistent IDs, structured
events, JSON output, tests.

**In M0.2:** temporal event memory, per-entity timelines, `stationary` and zone
events, the query API, explicit pixel-space coordinates, layering enforced by
test.

**In M0.3:** predictive risk engine, image-space kinematics, five factor types,
deterministic explainable scoring, structured risk assessments, the synthetic
warehouse demo.

**In M0.3.1:** tracker identity fix — tiered association evidence, ground-truth
identity metrics, and regression tests for reversal, occlusion, crossing and
crowding.

**Not yet:** frontend, LLM reasoning, audio, agents, counterfactual simulation,
database.

---

## What remains for M0.4

M0.3 deliberately stops at a deterministic, explainable score. The obvious next
steps, in rough order of value:

1. **Ground-plane calibration.** A homography per camera would turn every pixel
   measurement in this repo into a real one, and would fix the depth problem
   that currently limits every threshold. This is the highest-value change
   available and it makes the risk model meaningfully more accurate rather than
   merely more elaborate.
2. **Validation against labelled footage.** The weights are defensible but
   unvalidated. Precision/recall against real near-miss data would turn the
   scoring model from judgement into evidence.
3. **The LLM reasoning layer**, sitting on top of the risk engine — explaining
   *why* a situation developed and what to do, with the deterministic score as
   its grounding. The `Explainer` interface is the seam it plugs into.
4. **Re-identification**, so an entity that leaves and re-enters keeps its
   history and its persistence record.
5. **Broader incident modelling**: vehicle-vehicle conflicts, static hazards,
   occlusion-aware separation.
6. **The frontend**, once there is something stable to display.
