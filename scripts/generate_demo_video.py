"""Generate a small synthetic video for smoke-testing the pipeline.

Draws coloured rectangles moving along known paths. Two uses:

* run the whole pipeline end-to-end with the ``blob`` detector on a machine
  with no model weights and no GPU;
* have a fixture whose ground truth (how many objects, moving which way) is
  known exactly, so event output can be asserted against it.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple


def generate(
    path: str,
    width: int = 640,
    height: int = 384,
    fps: int = 20,
    seconds: float = 4.0,
) -> str:
    import cv2
    import numpy as np

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    total_frames = int(fps * seconds)
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise IOError(f"Could not open video writer for {path}")

    try:
        for index in range(total_frames):
            frame = np.full((height, width, 3), 40, dtype=np.uint8)
            # Static floor band, so the scene isn't uniformly flat.
            cv2.rectangle(frame, (0, height - 60), (width, height), (70, 70, 70), -1)

            progress = index / max(1, total_frames - 1)

            # Red "person" walking left to right across the whole frame.
            px = int(20 + progress * (width - 100))
            cv2.rectangle(frame, (px, 140), (px + 46, 250), (0, 0, 220), -1)

            # Blue "truck" crossing right to left, entering after 25% of the clip.
            if progress > 0.25:
                tx = int(width - 140 - (progress - 0.25) * (width * 1.1))
                if tx > -130:
                    cv2.rectangle(frame, (tx, 210), (tx + 120, 285), (220, 60, 0), -1)

            # Green "car" parked, appears for the middle half of the clip only.
            if 0.3 < progress < 0.8:
                cv2.rectangle(frame, (430, 90), (540, 150), (0, 200, 0), -1)

            writer.write(frame)
    finally:
        writer.release()
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default="data/demo/demo.mp4")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=4.0)
    args = parser.parse_args()

    path = generate(
        args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        seconds=args.seconds,
    )
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
