"""The perception benchmark.

Measures the workload Sentinel actually runs — the same YOLO backend, the same
tracker, the same event generator — rather than a synthetic matrix multiply
that would tell you about the hardware and nothing about this system.

What is measured, and what is deliberately not
----------------------------------------------
Three clocks, reported separately so no cost is hidden:

``model_load_seconds``
    Constructing the detector: reading weights from disk, building the graph,
    moving parameters onto the device. **Excluded from every throughput
    figure** and reported on its own line. Any model download happens before
    this (the harness requires the weights to exist), so a download can never
    land inside a measurement.
``inference``
    One call to ``detector.detect(frame)``. This is the model *plus* its
    preprocessing and postprocessing — letterboxing, tensor transfer, NMS, box
    decoding. Those costs are real and are not separated out, because a user
    cannot skip them either.
``pipeline``
    Decode → detect → track → generate events, per frame. The difference
    between this and ``inference`` is exactly what the deterministic layers
    cost.

Warmup frames are run and then discarded: the first inference on any device
includes lazy allocator and kernel setup, and on a GPU it is often an order of
magnitude slower than the steady state.

Everything is fixed across devices — weights, resolution, confidence, IoU,
class filter, frame count, stride, video — so a CPU and an AMD run differ only
in where the arithmetic happened.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..accel import (
    DeviceSpec,
    EnvironmentInfo,
    file_digest,
    probe_environment,
    resolve_device,
    verify_execution,
)
from ..config import PipelineConfig
from ..events.generator import EventGenerator
from ..perception.detector import create_detector
from ..perception.tracker import create_tracker
from ..perception.video import VideoSource
from .timing import LatencySummary, Stopwatch, summarise

#: Frames run before measurement starts. Engineering-selected: enough to cover
#: allocator warmup and the first kernel compile on a GPU, cheap enough not to
#: dominate a short clip.
DEFAULT_WARMUP_FRAMES = 5

#: Frames measured when the caller does not say. Long enough for a stable p95.
DEFAULT_FRAMES = 120

BENCHMARK_BANNER = (
    "SENTINEL PERCEPTION BENCHMARK",
    "measures the real pipeline; model load time is excluded from throughput",
)


class BenchmarkError(RuntimeError):
    """The benchmark could not run — missing video, missing weights, no device."""


@dataclass
class BenchmarkConfig:
    """Everything that must be identical between two comparable runs."""

    video: str
    device: str = "auto"
    weights: str = "yolov8n.pt"
    detector: str = "yolo"
    imgsz: int = 640
    confidence: float = 0.25
    iou: float = 0.45
    classes: Optional[List[str]] = None
    frames: int = DEFAULT_FRAMES
    warmup: int = DEFAULT_WARMUP_FRAMES
    stride: int = 1
    half: bool = False

    def __post_init__(self) -> None:
        if self.frames <= 0:
            raise BenchmarkError("frames must be positive")
        if self.warmup < 0:
            raise BenchmarkError("warmup cannot be negative")

    @property
    def total_frames(self) -> int:
        """Frames actually decoded: warmup is run, then thrown away."""
        return self.frames + self.warmup

    def workload_key(self) -> Dict[str, Any]:
        """The fields two runs must share to be comparable at all."""
        return {
            "video": os.path.basename(self.video),
            "detector": self.detector,
            "weights": os.path.basename(self.weights),
            "imgsz": self.imgsz,
            "confidence": self.confidence,
            "iou": self.iou,
            "classes": list(self.classes) if self.classes else None,
            "frames": self.frames,
            "warmup": self.warmup,
            "stride": self.stride,
            "half": self.half,
        }

    def to_dict(self) -> Dict[str, Any]:
        payload = self.workload_key()
        payload["device_requested"] = self.device
        return payload

    def to_pipeline_config(self, device: DeviceSpec) -> PipelineConfig:
        """The very same configuration object the production pipeline uses."""
        return PipelineConfig(
            video_path=self.video,
            stride=self.stride,
            max_frames=self.total_frames,
            detector=self.detector,
            weights=self.weights,
            device=device.torch_device,
            imgsz=self.imgsz,
            half=self.half,
            confidence=self.confidence,
            nms_iou=self.iou,
            classes=list(self.classes) if self.classes else None,
        )


@dataclass
class WorkloadSummary:
    """What the run actually saw, so two devices can be compared for parity."""

    detections: int = 0
    tracks_created: int = 0
    events: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    mean_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detections": self.detections,
            "tracks_created": self.tracks_created,
            "events": self.events,
            "class_counts": dict(sorted(self.class_counts.items())),
            "mean_confidence": round(self.mean_confidence, 5),
        }


@dataclass
class BenchmarkResult:
    """One device's measured result. Serialisable; no live objects."""

    config: BenchmarkConfig
    device: DeviceSpec
    environment: EnvironmentInfo
    inference: LatencySummary
    pipeline: LatencySummary
    decode: LatencySummary
    tracking: LatencySummary
    events: LatencySummary
    model_load_seconds: float
    workload: WorkloadSummary
    device_verified: bool
    device_detail: str = ""
    weights_sha256: Optional[str] = None
    video_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def inference_fps(self) -> float:
        return self.inference.fps

    @property
    def pipeline_fps(self) -> float:
        return self.pipeline.fps

    @property
    def accelerated(self) -> bool:
        """True only when a GPU was asked for *and* proven to have run it."""
        return self.device.is_accelerator and self.device_verified

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "device": self.device.to_dict(),
            "device_verified": self.device_verified,
            "device_detail": self.device_detail,
            "environment": self.environment.to_dict(),
            "weights_sha256": self.weights_sha256,
            "video": self.video_metadata,
            "model_load_seconds": round(self.model_load_seconds, 3),
            "timings": {
                "inference": self.inference.to_dict(),
                "pipeline": self.pipeline.to_dict(),
                "decode": self.decode.to_dict(),
                "tracking": self.tracking.to_dict(),
                "events": self.events.to_dict(),
            },
            "workload": self.workload.to_dict(),
        }

    def lines(self) -> List[str]:
        rows = [
            f"device            : {self.device.describe()}",
            f"device verified   : {'yes' if self.device_verified else 'NO'} "
            f"({self.device_detail})",
            f"model load        : {self.model_load_seconds:.3f} s "
            "(excluded from throughput below)",
            f"frames measured   : {self.inference.count} "
            f"(+{self.config.warmup} warmup, discarded)",
            "",
            self.inference.line(),
            self.pipeline.line(),
            self.decode.line(),
            self.tracking.line(),
            self.events.line(),
        ]
        return rows


# ---------------------------------------------------------------------------
def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run the perception pipeline on one device and measure it.

    Raises :class:`BenchmarkError` rather than degrading quietly: a benchmark
    that falls back to the CPU when the GPU is missing produces a number that
    looks like a result and is not one.
    """
    if not os.path.isfile(config.video):
        raise BenchmarkError(f"video not found: {config.video}")

    device = resolve_device(config.device)
    verified, detail = verify_execution(device)
    if device.is_accelerator and not verified:
        raise BenchmarkError(
            f"{device.label} was selected but could not execute a tensor "
            f"operation ({detail}); refusing to report it as accelerated"
        )

    environment = probe_environment()
    weights_path = config.weights if os.path.isfile(config.weights) else None
    if config.detector == "yolo" and weights_path is None:
        raise BenchmarkError(
            f"weights not found: {config.weights}. Download them before "
            "benchmarking — model download must never land inside a timing."
        )

    # Build exactly what the pipeline builds. Going through PipelineConfig
    # rather than hand-rolling the arguments is what makes this a benchmark of
    # Sentinel rather than a benchmark of a YOLO call that resembles it.
    pipeline_config = config.to_pipeline_config(device)

    load_started = time.perf_counter()
    detector = create_detector(
        pipeline_config.detector,
        classes=pipeline_config.classes,
        min_confidence=pipeline_config.confidence,
        **pipeline_config.detector_kwargs(),
    )
    model_load_seconds = time.perf_counter() - load_started

    tracker = create_tracker(
        pipeline_config.tracker, **pipeline_config.tracker_kwargs()
    )
    generator = EventGenerator(
        sample_interval=pipeline_config.sample_interval,
        movement_threshold=pipeline_config.movement_threshold,
        stationary_threshold=pipeline_config.stationary_threshold,
        stationary_duration=pipeline_config.stationary_duration,
    )

    decode = Stopwatch("decode")
    inference = Stopwatch("inference (detect)")
    tracking = Stopwatch("tracking")
    events = Stopwatch("event generation")
    pipeline_samples: List[float] = []

    workload = WorkloadSummary()
    confidences: List[float] = []
    track_ids: set = set()
    video_metadata: Dict[str, Any] = {}

    try:
        with VideoSource(
            config.video, stride=config.stride, max_frames=config.total_frames
        ) as source:
            video_metadata = source.metadata.to_dict()
            previous = time.perf_counter()
            for index, frame in enumerate(source):
                decoded_at = time.perf_counter()
                decode.record(decoded_at - previous)

                detect_started = time.perf_counter()
                detections = detector.detect(frame.image)
                detect_done = time.perf_counter()
                inference.record(detect_done - detect_started)

                tracks = tracker.update(detections, frame.timestamp)
                track_done = time.perf_counter()
                tracking.record(track_done - detect_done)

                produced = generator.process(
                    tracks,
                    timestamp=frame.timestamp,
                    frame_index=frame.index,
                    lost_tracks=tracker.lost_tracks,
                )
                frame_done = time.perf_counter()
                events.record(frame_done - track_done)
                # Decode through events: everything one frame costs.
                pipeline_samples.append(frame_done - previous)

                if index >= config.warmup:
                    workload.detections += len(detections)
                    workload.events += len(produced)
                    track_ids.update(t.track_id for t in tracks)
                    for detection in detections:
                        name = detection.class_name or str(detection.class_id)
                        workload.class_counts[name] = (
                            workload.class_counts.get(name, 0) + 1
                        )
                        confidences.append(detection.confidence)
                previous = time.perf_counter()
    finally:
        detector.close()

    measured = len(inference.samples)
    if measured <= config.warmup:
        raise BenchmarkError(
            f"the clip yielded {measured} frames, which is not more than the "
            f"{config.warmup} warmup frames; use a longer video or fewer warmup frames"
        )

    for watch in (decode, inference, tracking, events):
        watch.discard(config.warmup)
    pipeline_samples = pipeline_samples[config.warmup :]

    workload.tracks_created = len(track_ids)
    workload.mean_confidence = (
        sum(confidences) / len(confidences) if confidences else 0.0
    )

    return BenchmarkResult(
        config=config,
        device=device,
        environment=environment,
        inference=inference.summary(),
        pipeline=summarise("pipeline (end to end)", pipeline_samples),
        decode=decode.summary(),
        tracking=tracking.summary(),
        events=events.summary(),
        model_load_seconds=model_load_seconds,
        workload=workload,
        device_verified=verified,
        device_detail=detail,
        weights_sha256=file_digest(weights_path),
        video_metadata=video_metadata,
    )
