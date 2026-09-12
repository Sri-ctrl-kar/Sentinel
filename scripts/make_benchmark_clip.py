#!/usr/bin/env python3
"""Build a benchmark clip by panning a window across a still photograph.

The benchmark needs a clip with *real* content: a synthetic scene of coloured
rectangles runs the detector just as hard, but finds nothing, so detection
counts, tracking and event generation are all trivially zero and a CPU/GPU
parity check compares nothing to nothing.

Panning a crop window across one photograph gives genuine detections that move
smoothly frame to frame — enough for the tracker and the event generator to do
real work — while keeping the input reproducible and tiny. No video is
committed to this repository, and neither is the source image.

    python scripts/make_benchmark_clip.py --image photo.jpg --out /tmp/clip.mp4

Any photograph with people or vehicles in it will do.
"""

from __future__ import annotations

import argparse
import math
import os
import sys


def generate(
    image_path: str,
    out_path: str,
    frames: int = 125,
    width: int = 640,
    height: int = 384,
    fps: float = 25.0,
) -> str:
    import cv2

    source = cv2.imread(image_path)
    if source is None:
        raise SystemExit(f"could not read image: {image_path}")

    # Scale so the crop window can travel a meaningful distance in both axes.
    scale = max(1.6 * width / source.shape[1], 1.6 * height / source.shape[0])
    if scale > 1.0:
        source = cv2.resize(
            source,
            (int(source.shape[1] * scale), int(source.shape[0] * scale)),
            interpolation=cv2.INTER_LINEAR,
        )

    max_x = max(source.shape[1] - width, 1)
    max_y = max(source.shape[0] - height, 1)

    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise SystemExit(f"could not open video writer for {out_path}")

    try:
        for index in range(frames):
            phase = index / max(frames - 1, 1)
            # A slow diagonal sweep: monotonic in x, a gentle sine in y, so the
            # motion is smooth and deterministic rather than jittery.
            x = int(phase * max_x)
            y = int((0.5 - 0.5 * math.cos(math.pi * phase)) * max_y)
            writer.write(source[y : y + height, x : x + width])
    finally:
        writer.release()
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="source photograph")
    parser.add_argument("--out", required=True, help="output .mp4 path")
    parser.add_argument("--frames", type=int, default=125)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args(argv)

    path = generate(
        args.image, args.out, args.frames, args.width, args.height, args.fps
    )
    size = os.path.getsize(path)
    print(f"wrote {path} ({args.frames} frames, {size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
