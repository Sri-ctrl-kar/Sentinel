"""Generate a small synthetic video for smoke-testing the pipeline.

Draws coloured rectangles moving along known paths. Two uses:

* run the whole pipeline end-to-end with the ``blob`` detector on a machine
  with no model weights and no GPU;
* have a fixture whose ground truth (how many objects, moving which way) is
  known exactly, so event output can be asserted against it.

``--annotations`` additionally writes a clip annotation file derived from the
**drawing commands themselves**. That matters for benchmarking: the ground
truth comes from what the renderer drew, not from what the pipeline detected,
so evaluating the pipeline against it is not circular. It is still a rendered
scene — perfect contrast, no motion blur, no occlusion — and the benchmark says
so wherever the numbers appear.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional, Tuple


#: Actors the renderer draws, as (id, class, hue-label) plus a box function of
#: progress. The annotation writer and the drawing code share these, so the two
#: cannot drift apart.
def _person_box(progress: float, width: int, height: int):
    x = int(20 + progress * (width - 100))
    return (x, 140, x + 46, 250)


def _truck_box(progress: float, width: int, height: int):
    if progress <= 0.25:
        return None
    x = int(width - 140 - (progress - 0.25) * (width * 1.1))
    return (x, 210, x + 120, 285) if x > -130 else None


def _car_box(progress: float, width: int, height: int):
    if not (0.3 < progress < 0.8):
        return None
    return (430, 90, 540, 150)


RENDERED_ACTORS = (
    ("person_gt", "person", _person_box),
    ("truck_gt", "truck", _truck_box),
    ("car_gt", "car", _car_box),
)


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
            colours = {
                "person_gt": (0, 0, 220),    # red
                "truck_gt": (220, 60, 0),    # blue
                "car_gt": (0, 200, 0),       # green
            }
            for entity_id, _class_name, box_of in RENDERED_ACTORS:
                box = box_of(progress, width, height)
                if box is None:
                    continue
                cv2.rectangle(
                    frame, (box[0], box[1]), (box[2], box[3]), colours[entity_id], -1
                )

            writer.write(frame)
    finally:
        writer.release()
    return path


def build_annotation(
    video_path: str,
    width: int = 640,
    height: int = 384,
    fps: int = 20,
    seconds: float = 4.0,
    name: str = "rendered_bay",
    annotation_path: Optional[str] = None,
    box_every: int = 5,
) -> dict:
    """Ground truth for the rendered clip, taken from the drawing commands.

    Boxes are emitted every ``box_every`` frames — the annotation format
    interpolates between them — which is also roughly what a human annotating
    by hand would produce.
    """
    total_frames = int(fps * seconds)
    entities = []
    for entity_id, class_name, box_of in RENDERED_ACTORS:
        boxes = {}
        for index in range(total_frames):
            if index % box_every and index != total_frames - 1:
                continue
            progress = index / max(1, total_frames - 1)
            box = box_of(progress, width, height)
            if box is not None:
                boxes[str(index)] = [float(v) for v in box]
        if boxes:
            entities.append(
                {"id": entity_id, "class_name": class_name, "boxes": boxes}
            )

    video = video_path
    if annotation_path:
        # Store the media path relative to the annotation, so the pair can move
        # together.
        video = os.path.relpath(
            os.path.abspath(video_path),
            os.path.dirname(os.path.abspath(annotation_path)),
        )

    return {
        "name": name,
        "video": video,
        "fps": float(fps),
        "notes": (
            "RENDERED clip, not real footage. Ground truth derived from the "
            "renderer's drawing commands, independent of the pipeline."
        ),
        "entities": entities,
        "events": _rendered_events(total_frames, fps, width, height),
    }


def _rendered_events(total_frames: int, fps: int, width: int, height: int) -> list:
    """Appearance and disappearance events implied by the drawing commands.

    Only the events the renderer can state with certainty are annotated.
    Movement and dwell thresholds are pipeline configuration, not facts about
    the scene, so they are deliberately not asserted here.
    """
    events = []
    for entity_id, _class_name, box_of in RENDERED_ACTORS:
        visible = [
            index
            for index in range(total_frames)
            if box_of(index / max(1, total_frames - 1), width, height) is not None
        ]
        if not visible:
            continue
        events.append(
            {"entity": entity_id, "action": "appeared", "time": visible[0] / fps}
        )
        events.append(
            {
                "entity": entity_id,
                "action": "disappeared",
                "time": visible[-1] / fps,
            }
        )
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default="data/demo/demo.mp4")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument(
        "--annotations",
        default=None,
        metavar="PATH",
        help="Also write ground-truth annotations derived from the drawing commands",
    )
    args = parser.parse_args()

    path = generate(
        args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        seconds=args.seconds,
    )
    print(f"Wrote {path}")

    if args.annotations:
        payload = build_annotation(
            path,
            width=args.width,
            height=args.height,
            fps=args.fps,
            seconds=args.seconds,
            annotation_path=args.annotations,
        )
        directory = os.path.dirname(os.path.abspath(args.annotations))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.annotations, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(f"Wrote {args.annotations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
