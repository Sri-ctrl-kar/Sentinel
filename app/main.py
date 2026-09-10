"""Sentinel M0.1 CLI: run the perception pipeline over a local video file."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

from .config import DEFAULT_CLASSES, PipelineConfig
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel",
        description="Sentinel M0.1 — video -> detection -> tracking -> structured events",
    )
    parser.add_argument("video", help="Path to a local video file")
    parser.add_argument(
        "-o",
        "--output",
        default="data/events.json",
        help="Where to write the event JSON (default: data/events.json)",
    )

    detection = parser.add_argument_group("detection")
    detection.add_argument(
        "--detector",
        default="yolo",
        help="Detection backend: yolo | blob | mock (default: yolo)",
    )
    detection.add_argument(
        "--weights", default="yolov8n.pt", help="YOLO weights (default: yolov8n.pt)"
    )
    detection.add_argument(
        "--device",
        default="auto",
        help="auto | cpu | cuda (ROCm also reports as 'cuda') | mps",
    )
    detection.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    detection.add_argument(
        "--half", action="store_true", help="FP16 inference (GPU only; good on AMD CDNA)"
    )
    detection.add_argument(
        "--confidence", type=float, default=0.25, help="Detection confidence floor"
    )
    detection.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help=f"Class names to keep (default: {len(DEFAULT_CLASSES)} incident-relevant "
        "COCO classes). Pass --classes with no values to keep everything.",
    )

    tracking = parser.add_argument_group("tracking")
    tracking.add_argument("--tracker", default="byte_iou", help="Tracking strategy")
    tracking.add_argument(
        "--track-max-age",
        type=int,
        default=30,
        help="Frames a track may coast unmatched before it is dropped",
    )
    tracking.add_argument(
        "--track-min-hits",
        type=int,
        default=2,
        help="Matched frames before a track is reported",
    )
    tracking.add_argument(
        "--track-iou",
        type=float,
        default=0.3,
        help="Minimum IoU for detection/track association",
    )

    events = parser.add_argument_group("events")
    events.add_argument(
        "--sample-interval",
        type=float,
        default=1.0,
        help="Seconds between 'detected' heartbeat events per entity",
    )
    events.add_argument(
        "--movement-threshold",
        type=float,
        default=40.0,
        help="Pixels of travel before a 'moved' event is emitted",
    )

    stream = parser.add_argument_group("stream")
    stream.add_argument(
        "--stride", type=int, default=1, help="Process every Nth frame (default: 1)"
    )
    stream.add_argument(
        "--max-frames", type=int, default=None, help="Stop after N processed frames"
    )
    stream.add_argument(
        "--start-time", type=float, default=0.0, help="Seek to this offset (seconds)"
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    # ``--classes`` with no values means "keep every class"; omitting the flag
    # entirely means "use the incident-relevant default set".
    if args.classes is None:
        classes = list(DEFAULT_CLASSES)
    elif len(args.classes) == 0:
        classes = None
    else:
        classes = list(args.classes)

    return PipelineConfig(
        video_path=args.video,
        stride=args.stride,
        max_frames=args.max_frames,
        start_time=args.start_time,
        detector=args.detector,
        weights=args.weights,
        device=args.device,
        imgsz=args.imgsz,
        half=args.half,
        confidence=args.confidence,
        classes=classes,
        tracker=args.tracker,
        track_iou_threshold=args.track_iou,
        track_max_age=args.track_max_age,
        track_min_hits=args.track_min_hits,
        sample_interval=args.sample_interval,
        movement_threshold=args.movement_threshold,
        output_path=args.output,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = config_from_args(args)
    try:
        result = run_pipeline(config)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (IOError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary = result.memory.summary()
    print(
        json.dumps(
            {
                "video": config.video_path,
                "output": config.output_path,
                "frames_processed": result.frames_processed,
                "processing_fps": round(result.fps, 2),
                "detector": result.metadata.get("detector"),
                **summary,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
