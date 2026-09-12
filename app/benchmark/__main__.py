"""``python -m app.benchmark`` — diagnostics, benchmark, comparison, parity.

Three things this command will not do:

* claim acceleration it did not verify — a GPU run must prove a tensor
  executed on the device, or the run is refused;
* silently fall back to the CPU when a named device is missing — it stops and
  says ``AMD_BENCHMARK = NOT_RUN`` with the reason;
* report a speedup it did not measure.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from ..accel import (
    DeviceUnavailable,
    available_devices,
    probe_environment,
    resolve_device,
    verify_execution,
)
from .compare import compare_risk, compare_workloads, speedup
from .harness import (
    BENCHMARK_BANNER,
    DEFAULT_FRAMES,
    DEFAULT_WARMUP_FRAMES,
    BenchmarkConfig,
    BenchmarkError,
    BenchmarkResult,
    run_benchmark,
)

WIDTH = 72


def rule(char: str = "-") -> str:
    return char * WIDTH


def banner() -> List[str]:
    return [rule("="), f"  {BENCHMARK_BANNER[0]}", f"  {BENCHMARK_BANNER[1]}", rule("=")]


# ---------------------------------------------------------------------------
def info_lines(device_request: str = "auto") -> List[str]:
    """The ``--info`` diagnostic: what is here, and can the model use it?"""
    environment = probe_environment()
    lines = [rule("="), "  SENTINEL DEVICE DIAGNOSTICS", rule("=")]
    lines.extend(environment.lines())
    lines.append("")

    lines.append("selectable devices:")
    for spec in available_devices():
        ok, detail = verify_execution(spec)
        mark = "ok " if ok else "NO "
        lines.append(f"  [{mark}] {spec.describe():<46} {detail}")
    lines.append("")

    try:
        selected = resolve_device(device_request)
    except DeviceUnavailable as exc:
        lines.append(f"selected device   : unavailable — {exc}")
        return lines

    ok, detail = verify_execution(selected)
    lines.append(f"selected device   : {selected.describe()}")
    lines.append(f"tensor executes   : {'yes' if ok else 'NO'} ({detail})")
    lines.append(f"model can execute : {_model_check(selected)}")
    lines.append("")
    if environment.has_amd_gpu:
        lines.append("AMD ROCm / HIP is present and visible to PyTorch.")
    else:
        lines.append("AMD_BENCHMARK = NOT_RUN")
        lines.append("REASON = AMD ROCm device unavailable")
    return lines


def _model_check(spec: Any) -> str:
    """Can Ultralytics actually place a model on this device, right now?"""
    try:
        from ultralytics import YOLO  # noqa: F401
    except ImportError:
        return "unknown — ultralytics is not installed"
    try:
        import torch
    except ImportError:  # pragma: no cover - ultralytics implies torch
        return "unknown — torch is not installed"
    try:
        torch.zeros(1, 3, 32, 32, device=spec.torch_device)
    except Exception as exc:  # pragma: no cover - hardware-dependent
        return f"no — {type(exc).__name__}: {exc}"
    return f"yes — an image tensor allocates on {spec.torch_device}"


# ---------------------------------------------------------------------------
def result_block(result: BenchmarkResult) -> List[str]:
    lines = [rule("="), f"  {result.device.label}", rule("=")]
    lines.extend(result.lines())
    lines.append("")
    lines.append(
        f"detections {result.workload.detections}  "
        f"tracks {result.workload.tracks_created}  "
        f"events {result.workload.events}  "
        f"mean conf {result.workload.mean_confidence:.4f}"
    )
    return lines


def run_one(args: argparse.Namespace, device: str) -> BenchmarkResult:
    config = BenchmarkConfig(
        video=args.video,
        device=device,
        weights=args.weights,
        detector=args.detector,
        imgsz=args.imgsz,
        confidence=args.confidence,
        iou=args.iou,
        classes=args.classes,
        frames=args.frames,
        warmup=args.warmup,
        stride=args.stride,
        half=args.half,
    )
    return run_benchmark(config)


def evidence_for(result: BenchmarkResult, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """Run the full deterministic stack on one device, for a parity check.

    Imported here rather than at module scope: the parity check is the only
    part of benchmarking that touches the reasoning layers at all.
    """
    from ..intelligence import evidence_from_assessment
    from ..pipeline import PerceptionPipeline
    from ..reasoning import RiskEngine

    from ..main import zones_from_args
    from ..reasoning import RiskConfig

    pipeline_config = result.config.to_pipeline_config(result.device)
    pipeline_config.zones = zones_from_args(args)
    pipeline = PerceptionPipeline(config=pipeline_config)
    try:
        outcome = pipeline.run(args.video)
    finally:
        pipeline.close()
    if outcome.temporal is None:
        return None
    risk_config = RiskConfig(
        operating_zones=list(args.operating_zone),
        zones=pipeline_config.zones,
    )
    engine = RiskEngine(risk_config)
    # The worst moment in the clip, not the last one: entities have usually
    # left the frame by the final timestamp, so assessing only the end would
    # compare two empty scenes and call it parity.
    worst = None
    for report in engine.assess_timeline(outcome.temporal, step=args.risk_step):
        for assessment in report.assessments:
            if worst is None or assessment.risk_score > worst.risk_score:
                worst = assessment
    if worst is None:
        return None
    return evidence_from_assessment(worst, events=outcome.temporal.events).to_dict()


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.benchmark",
        description=(
            "Benchmark Sentinel's perception workload on CPU or an AMD ROCm GPU."
        ),
    )
    parser.add_argument("--video", help="local video file to benchmark")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto | cpu | rocm | cuda | mps (a named device is never faked)",
    )
    parser.add_argument(
        "--compare",
        help="comma-separated devices to benchmark and compare, e.g. cpu,rocm",
    )
    parser.add_argument(
        "--parity",
        action="store_true",
        help="also run the full deterministic stack on each device and diff the verdict",
    )
    parser.add_argument("--info", action="store_true", help="device diagnostics only")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--detector", default="yolo")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument(
        "--zone",
        action="append",
        default=[],
        help="NAME=X1,Y1,X2,Y2 image-space zone, repeatable (for --parity)",
    )
    parser.add_argument(
        "--zones", default=None, help="zone definitions from a JSON file (for --parity)"
    )
    parser.add_argument(
        "--operating-zone",
        action="append",
        default=[],
        help="zone name where vehicles operate, repeatable (for --parity)",
    )
    parser.add_argument(
        "--risk-step",
        type=float,
        default=0.5,
        help="seconds between risk assessments when scanning for --parity",
    )
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_FRAMES)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--half",
        action="store_true",
        help="FP16. Off by default: M0.8 establishes an FP32 baseline first",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.info or not args.video:
        lines = info_lines(args.device)
        if not args.video and not args.info:
            lines.append("")
            lines.append("no --video given; nothing was benchmarked")
        print("\n".join(lines))
        return 0

    devices = (
        [d.strip() for d in args.compare.split(",") if d.strip()]
        if args.compare
        else [args.device]
    )

    results: List[BenchmarkResult] = []
    payload: Dict[str, Any] = {"runs": []}
    blocks: List[str] = banner()

    for device in devices:
        try:
            result = run_one(args, device)
        except (BenchmarkError, DeviceUnavailable) as exc:
            if device.lower() in ("rocm", "amd"):
                print("AMD_BENCHMARK = NOT_RUN", file=sys.stderr)
                print(f"REASON = {exc}", file=sys.stderr)
            else:
                print(f"error: {exc}", file=sys.stderr)
            if len(devices) == 1:
                return 2
            continue
        results.append(result)
        payload["runs"].append(result.to_dict())
        blocks.extend([""] + result_block(result))

    if not results:
        return 2

    if len(results) >= 2:
        baseline, candidate = results[0], results[1]
        ratio = speedup(baseline, candidate)
        payload["speedup"] = ratio.to_dict()
        blocks.extend(["", rule("="), "  SPEEDUP", rule("=")])
        blocks.extend(ratio.lines())
        if not candidate.accelerated:
            blocks.append(
                "note: the candidate device is not a verified accelerator, so "
                "this is a device-to-device comparison, not an acceleration claim"
            )

        parity = compare_workloads(baseline, candidate)
        if args.parity:
            parity = compare_risk(
                evidence_for(baseline, args), evidence_for(candidate, args), parity
            )
        payload["parity"] = parity.to_dict()
        blocks.extend(["", rule("="), "  CORRECTNESS PARITY", rule("=")])
        blocks.append(f"{'PASS' if parity.ok else 'FAIL'} — {parity.summary()}")
        for difference in parity.differences:
            blocks.append(f"  {difference}")
        for note in parity.notes:
            blocks.append(f"  note: {note}")
        preserved = parity.semantics_preserved
        if preserved is None:
            blocks.append(
                "risk semantics unchanged: NOT COMPARED"
                + ("" if args.parity else " (pass --parity to compare them)")
            )
        else:
            blocks.append(
                f"risk semantics unchanged: {'YES' if preserved else 'NO'}"
            )

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("\n".join(blocks))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
