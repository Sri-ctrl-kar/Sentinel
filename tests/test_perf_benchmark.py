"""The perception benchmark: configuration, timing, schema, parity (M0.8 C-K).

None of these tests needs a GPU, model weights, or a video: the pieces that can
be checked deterministically — percentile arithmetic, warmup handling, the
result schema, the comparison tolerances — are checked here, and the parts that
need hardware are refused loudly rather than faked.
"""

from __future__ import annotations

import json

import pytest

from app.accel.device import CPU, KIND_ROCM, DeviceSpec
from app.benchmark import (
    CONFIDENCE_TOLERANCE,
    COUNT_TOLERANCE,
    RISK_TOLERANCE,
    BenchmarkConfig,
    BenchmarkError,
    BenchmarkResult,
    LatencySummary,
    ParityReport,
    Stopwatch,
    WorkloadSummary,
    compare_risk,
    compare_workloads,
    percentile,
    run_benchmark,
    speedup,
    summarise,
)
from app.accel import probe_environment
from app.benchmark.__main__ import build_parser, info_lines, main

ROCM_GPU = DeviceSpec(
    kind=KIND_ROCM,
    torch_device="cuda",
    name="AMD Instinct MI300X",
    index=0,
    runtime_version="6.2",
)


def make_result(
    device=CPU,
    inference_ms=40.0,
    pipeline_ms=42.0,
    frames=100,
    verified=True,
    workload=None,
    config=None,
) -> BenchmarkResult:
    """A fabricated result, for exercising comparison and serialisation."""
    inference = summarise("inference (detect)", [inference_ms / 1000.0] * frames)
    pipeline = summarise("pipeline (end to end)", [pipeline_ms / 1000.0] * frames)
    empty = summarise("decode", [0.0005] * frames)
    return BenchmarkResult(
        config=config or BenchmarkConfig(video="clip.mp4", frames=frames),
        device=device,
        environment=probe_environment(),
        inference=inference,
        pipeline=pipeline,
        decode=empty,
        tracking=empty,
        events=empty,
        model_load_seconds=1.25,
        workload=workload or WorkloadSummary(120, 4, 30, {"person": 100, "bus": 20}, 0.61),
        device_verified=verified,
        device_detail="fixture",
    )


# ---------------------------------------------------------------------------
# Percentiles and timing aggregation
# ---------------------------------------------------------------------------
def test_percentiles_use_nearest_rank_on_observed_samples():
    samples = list(range(1, 101))
    assert percentile(samples, 50) == 50
    assert percentile(samples, 95) == 95
    assert percentile(samples, 99) == 99
    assert percentile(samples, 100) == 100
    assert percentile(samples, 0) == 1


def test_a_percentile_is_always_a_value_that_was_measured():
    """No interpolation: reporting a latency no frame had would be fiction."""
    samples = [1.0, 2.0, 100.0]
    for p in (50, 95, 99):
        assert percentile(samples, p) in samples


def test_percentiles_are_order_independent():
    assert percentile([9, 1, 5, 3, 7], 50) == percentile([1, 3, 5, 7, 9], 50) == 5


def test_a_percentile_of_nothing_is_undefined_not_zero():
    with pytest.raises(ValueError, match="undefined"):
        percentile([], 50)


def test_an_out_of_range_percentile_is_rejected():
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_the_tail_is_visible_in_p95():
    """A mean alone would hide the slow frame; that is why p95 is reported."""
    samples = [0.010] * 95 + [0.500] * 5
    summary = summarise("detect", samples)
    assert summary.mean_ms == pytest.approx(34.5, abs=0.1)
    assert summary.p50_ms == pytest.approx(10.0)
    assert summary.p95_ms == pytest.approx(10.0)
    assert summary.p99_ms == pytest.approx(500.0)
    assert summary.max_ms == pytest.approx(500.0)


def test_a_summary_reports_throughput_consistent_with_its_samples():
    summary = summarise("detect", [0.04] * 50)
    assert summary.count == 50
    assert summary.total_seconds == pytest.approx(2.0)
    assert summary.fps == pytest.approx(25.0)
    assert summary.mean_ms == pytest.approx(40.0)


def test_an_empty_summary_is_zero_everywhere_rather_than_an_error():
    summary = summarise("detect", [])
    assert summary.count == 0
    assert summary.fps == 0.0
    assert summary.to_dict()["frames"] == 0


def test_the_stopwatch_records_and_discards_warmup():
    watch = Stopwatch("detect")
    for value in (0.5, 0.1, 0.1, 0.1):
        watch.record(value)
    watch.discard(1)
    assert watch.samples == [0.1, 0.1, 0.1]
    assert watch.summary().mean_ms == pytest.approx(100.0)


def test_the_stopwatch_works_as_a_context_manager():
    watch = Stopwatch("detect")
    with watch:
        pass
    assert len(watch.samples) == 1
    assert watch.samples[0] >= 0.0


def test_a_latency_summary_serialises_every_reported_statistic():
    payload = summarise("detect", [0.01, 0.02, 0.03]).to_dict()
    for key in (
        "label",
        "frames",
        "fps",
        "mean_ms",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "min_ms",
        "max_ms",
        "stdev_ms",
        "total_seconds",
    ):
        assert key in payload


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def test_warmup_frames_are_additional_to_the_measured_frames():
    config = BenchmarkConfig(video="clip.mp4", frames=100, warmup=5)
    assert config.total_frames == 105


def test_a_non_positive_frame_count_is_rejected():
    with pytest.raises(BenchmarkError):
        BenchmarkConfig(video="clip.mp4", frames=0)


def test_a_negative_warmup_is_rejected():
    with pytest.raises(BenchmarkError):
        BenchmarkConfig(video="clip.mp4", warmup=-1)


def test_the_workload_key_holds_everything_that_must_match_between_devices():
    key = BenchmarkConfig(video="a/clip.mp4", device="cpu").workload_key()
    for field in (
        "video",
        "weights",
        "imgsz",
        "confidence",
        "iou",
        "classes",
        "frames",
        "warmup",
        "stride",
        "half",
    ):
        assert field in key
    assert "device_requested" not in key, "the device is what differs, not the workload"


def test_the_workload_key_ignores_the_device_so_two_devices_are_comparable():
    cpu = BenchmarkConfig(video="clip.mp4", device="cpu").workload_key()
    gpu = BenchmarkConfig(video="clip.mp4", device="rocm").workload_key()
    assert cpu == gpu


def test_the_benchmark_config_produces_the_pipeline_config_the_app_uses():
    """The benchmark must measure Sentinel, not something that resembles it."""
    config = BenchmarkConfig(
        video="clip.mp4", imgsz=512, confidence=0.4, iou=0.5, classes=["person"]
    )
    pipeline_config = config.to_pipeline_config(CPU)
    assert pipeline_config.detector == "yolo"
    assert pipeline_config.imgsz == 512
    assert pipeline_config.confidence == 0.4
    assert pipeline_config.nms_iou == 0.5
    assert pipeline_config.classes == ["person"]
    assert pipeline_config.device == "cpu"


def test_fp16_is_off_by_default():
    """M0.8 establishes an FP32 baseline; precision changes are never silent."""
    assert BenchmarkConfig(video="clip.mp4").half is False
    assert build_parser().parse_args([]).half is False


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
def test_a_missing_video_is_refused():
    with pytest.raises(BenchmarkError, match="video not found"):
        run_benchmark(BenchmarkConfig(video="/no/such/clip.mp4", device="cpu"))


def test_a_missing_weights_file_is_refused_rather_than_downloaded_mid_timing(tmp_path):
    """A download inside a measurement would silently ruin it."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not really a video, but it exists")
    with pytest.raises(BenchmarkError, match="weights not found"):
        run_benchmark(
            BenchmarkConfig(
                video=str(clip), device="cpu", weights=str(tmp_path / "nope.pt")
            )
        )


def test_requesting_an_unavailable_device_fails_the_run(tmp_path):
    from app.accel import DeviceUnavailable

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    with pytest.raises((BenchmarkError, DeviceUnavailable)):
        run_benchmark(BenchmarkConfig(video=str(clip), device="rocm"))


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------
def test_the_result_schema_is_json_serialisable_and_complete():
    payload = make_result().to_dict()
    json.dumps(payload)  # must not raise
    assert set(payload) >= {
        "config",
        "device",
        "device_verified",
        "environment",
        "model_load_seconds",
        "timings",
        "workload",
        "video",
    }
    assert set(payload["timings"]) == {
        "inference",
        "pipeline",
        "decode",
        "tracking",
        "events",
    }


def test_model_load_time_is_reported_separately_from_throughput():
    result = make_result()
    assert result.model_load_seconds == 1.25
    assert "excluded from throughput" in "\n".join(result.lines())
    # The load time is not in any timing series.
    assert result.inference.total_seconds < result.model_load_seconds * 100


def test_inference_and_pipeline_are_reported_separately():
    result = make_result(inference_ms=40.0, pipeline_ms=50.0)
    assert result.inference_fps == pytest.approx(25.0)
    assert result.pipeline_fps == pytest.approx(20.0)
    assert result.pipeline_fps < result.inference_fps


def test_environment_metadata_records_what_is_needed_to_reproduce_a_run():
    payload = make_result().to_dict()["environment"]
    for key in (
        "os",
        "python",
        "torch",
        "ultralytics",
        "opencv",
        "hip",
        "cuda",
        "gpus",
        "cpu",
        "cpu_count",
        "ram_gb",
    ):
        assert key in payload


def test_environment_metadata_contains_no_secrets():
    """Hardware facts only: no credentials, no user, no filesystem paths."""
    payload = json.dumps(probe_environment().to_dict()).lower()
    for marker in ("key", "token", "password", "secret", "/home/", "/root/"):
        assert marker not in payload


def test_a_run_only_counts_as_accelerated_when_the_device_was_verified():
    assert make_result(device=ROCM_GPU, verified=True).accelerated is True
    assert make_result(device=ROCM_GPU, verified=False).accelerated is False
    assert make_result(device=CPU, verified=True).accelerated is False


# ---------------------------------------------------------------------------
# Speedup
# ---------------------------------------------------------------------------
def test_speedup_is_the_throughput_ratio():
    cpu = make_result(inference_ms=40.0, pipeline_ms=50.0)
    gpu = make_result(device=ROCM_GPU, inference_ms=10.0, pipeline_ms=20.0)
    ratio = speedup(cpu, gpu)
    assert ratio.inference == pytest.approx(4.0)
    assert ratio.pipeline == pytest.approx(2.5)
    assert ratio.candidate_label == "AMD ROCm / HIP"


def test_comparing_two_different_workloads_is_refused():
    cpu = make_result(config=BenchmarkConfig(video="a.mp4", frames=100))
    gpu = make_result(
        device=ROCM_GPU, config=BenchmarkConfig(video="a.mp4", frames=50), frames=50
    )
    with pytest.raises(ValueError, match="different workloads"):
        speedup(cpu, gpu)


def test_speedup_serialises_both_sides_not_just_the_ratio():
    payload = speedup(make_result(), make_result(device=ROCM_GPU)).to_dict()
    assert "baseline_inference_fps" in payload
    assert "candidate_inference_fps" in payload


# ---------------------------------------------------------------------------
# Correctness parity
# ---------------------------------------------------------------------------
def test_identical_workloads_compare_clean():
    report = compare_workloads(make_result(), make_result(device=ROCM_GPU))
    assert report.ok
    assert report.differences == []


def test_a_small_detection_count_difference_is_tolerated_and_reported():
    """Kernel-level float differences can move a box across the threshold."""
    left = make_result(workload=WorkloadSummary(1000, 4, 30, {"person": 1000}, 0.6))
    right = make_result(
        device=ROCM_GPU, workload=WorkloadSummary(1010, 4, 30, {"person": 1010}, 0.6)
    )
    report = compare_workloads(left, right)
    assert report.ok
    assert report.differences, "a tolerated difference is still reported"
    assert all(d.tolerated for d in report.differences)


def test_a_large_detection_count_difference_is_not_tolerated():
    left = make_result(workload=WorkloadSummary(100, 4, 30, {"person": 100}, 0.6))
    right = make_result(
        device=ROCM_GPU, workload=WorkloadSummary(150, 4, 30, {"person": 150}, 0.6)
    )
    report = compare_workloads(left, right)
    assert not report.ok


def test_the_count_tolerance_boundary_is_inclusive():
    left = make_result(workload=WorkloadSummary(100, 0, 0, {}, 0.0))
    right = make_result(
        device=ROCM_GPU,
        workload=WorkloadSummary(int(100 * (1 + COUNT_TOLERANCE)), 0, 0, {}, 0.0),
    )
    assert compare_workloads(left, right).ok


def test_a_different_set_of_classes_is_never_tolerated():
    left = make_result(workload=WorkloadSummary(100, 4, 30, {"person": 100}, 0.6))
    right = make_result(
        device=ROCM_GPU, workload=WorkloadSummary(100, 4, 30, {"truck": 100}, 0.6)
    )
    report = compare_workloads(left, right)
    assert not report.ok
    assert any(d.field == "class_labels" for d in report.differences)


def test_a_confidence_drift_beyond_tolerance_is_flagged():
    left = make_result(workload=WorkloadSummary(100, 4, 30, {"person": 100}, 0.60))
    right = make_result(
        device=ROCM_GPU,
        workload=WorkloadSummary(100, 4, 30, {"person": 100}, 0.60 + 2 * CONFIDENCE_TOLERANCE),
    )
    assert not compare_workloads(left, right).ok


# ---------------------------------------------------------------------------
# Risk parity — the claim that actually matters
# ---------------------------------------------------------------------------
EVIDENCE = {
    "risk_score": 63.3,
    "severity": "medium",
    "incident_type": "PERSON_VEHICLE_COLLISION_RISK",
    "incident_state": "observed",
    "coordinate_space": "image_pixels",
    "entities": [{"entity_id": "person_3", "class_name": "person"}],
}


def test_the_same_risk_verdict_on_both_devices_passes():
    report = compare_risk(dict(EVIDENCE), dict(EVIDENCE))
    assert report.ok
    assert report.semantics_preserved is True
    assert report.risk_compared is True


def test_a_changed_risk_score_is_never_tolerated():
    report = compare_risk(dict(EVIDENCE), dict(EVIDENCE, risk_score=70.0))
    assert not report.ok
    assert report.semantics_preserved is False


def test_a_risk_score_within_rounding_is_accepted():
    report = compare_risk(
        dict(EVIDENCE), dict(EVIDENCE, risk_score=63.3 + RISK_TOLERANCE / 2)
    )
    assert report.ok


@pytest.mark.parametrize(
    "field,value",
    [
        ("severity", "high"),
        ("incident_type", "CONVERGING_TRAJECTORIES"),
        ("incident_state", "imminent"),
        ("coordinate_space", "ground_plane_meters"),
    ],
)
def test_any_changed_risk_semantic_fails_parity(field, value):
    report = compare_risk(dict(EVIDENCE), dict(EVIDENCE, **{field: value}))
    assert not report.ok
    assert report.semantics_preserved is False


def test_a_different_set_of_involved_entities_fails_parity():
    other = dict(EVIDENCE, entities=[{"entity_id": "person_9", "class_name": "person"}])
    assert not compare_risk(dict(EVIDENCE), other).ok


def test_an_incident_on_one_device_only_fails_parity():
    assert not compare_risk(dict(EVIDENCE), None).ok
    assert not compare_risk(None, dict(EVIDENCE)).ok


def test_no_incident_on_either_device_is_reported_as_untested_not_as_a_pass():
    """A comparison that never happened must not read as one that succeeded."""
    report = compare_risk(None, None)
    assert report.ok
    assert report.risk_compared is False
    assert report.semantics_preserved is None
    assert any("not compared" in note for note in report.notes)


def test_a_parity_report_serialises():
    payload = compare_risk(dict(EVIDENCE), dict(EVIDENCE)).to_dict()
    assert payload["ok"] is True
    assert payload["semantics_preserved"] is True
    assert "notes" in payload


def test_parity_reports_summarise_themselves_honestly():
    clean = ParityReport("CPU", "AMD ROCm / HIP")
    assert "identical" in clean.summary()


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------
def test_the_diagnostic_command_reports_the_runtime(capsys):
    assert main(["--info"]) == 0
    out = capsys.readouterr().out
    for label in ("PyTorch", "HIP available", "HIP version", "selected device"):
        assert label in out


def test_the_diagnostic_says_not_run_when_no_amd_device_exists(capsys):
    """The required wording, so a reader is never left to assume."""
    main(["--info"])
    out = capsys.readouterr().out
    if "AMD ROCm / HIP is present" not in out:
        assert "AMD_BENCHMARK = NOT_RUN" in out
        assert "REASON = AMD ROCm device unavailable" in out


def test_the_diagnostic_names_the_accelerator_vendor_correctly(capsys):
    main(["--info"])
    out = capsys.readouterr().out
    assert "AMD ROCm / HIP" in out or "NVIDIA CUDA" in out or "CPU" in out


def test_without_a_video_nothing_is_benchmarked(capsys):
    assert main([]) == 0
    assert "nothing was benchmarked" in capsys.readouterr().out


def test_requesting_rocm_without_hardware_prints_not_run(capsys):
    from app.accel import gpu_kind

    if gpu_kind() == "rocm":  # pragma: no cover - only on an AMD machine
        pytest.skip("this machine has an AMD GPU")
    code = main(["--video", "/no/such/file.mp4", "--device", "rocm"])
    captured = capsys.readouterr()
    assert code == 2
    assert "AMD_BENCHMARK = NOT_RUN" in captured.err
    assert "REASON" in captured.err


def test_the_cli_exposes_the_documented_flags():
    args = build_parser().parse_args(
        ["--video", "clip.mp4", "--device", "rocm", "--frames", "50"]
    )
    assert args.video == "clip.mp4"
    assert args.device == "rocm"
    assert args.frames == 50


def test_info_lines_handle_an_unavailable_device_request():
    lines = "\n".join(info_lines("rocm"))
    assert "unavailable" in lines or "AMD ROCm / HIP" in lines


# ---------------------------------------------------------------------------
# The benchmark -> pipeline device boundary (M0.8 regression)
# ---------------------------------------------------------------------------
# The bug: to_pipeline_config() handed PipelineConfig the *torch* device string,
# which is "cuda" on ROCm. PipelineConfig.device is a request that gets
# re-resolved downstream, and app.accel reads "cuda" as "NVIDIA, specifically" —
# so a ROCm benchmark asked Sentinel to run on CUDA and was correctly refused.
# What must cross this boundary is the semantic kind.
from app.accel import device as accel_module  # noqa: E402
from app.benchmark.harness import semantic_device  # noqa: E402
from app.perception.backends import yolo_ultralytics as yolo_backend  # noqa: E402
from test_accel_device import fake_torch  # noqa: E402

ROCM_SPEC = DeviceSpec(
    kind="rocm", torch_device="cuda", name="AMD Instinct MI300X", index=0,
    runtime_version="6.2.41134",
)


@pytest.fixture
def rocm_host(monkeypatch):
    """Make this machine look like a single-GPU ROCm host to app.accel."""
    torch_module = fake_torch(
        cuda_available=True, hip="6.2.41134", names=("AMD Instinct MI300X",)
    )
    monkeypatch.setattr(accel_module, "_torch", lambda: torch_module)
    return torch_module


def test_a_rocm_device_reaches_the_pipeline_as_rocm_not_as_cuda():
    """The required regression: kind crosses the boundary, not torch_device."""
    config = BenchmarkConfig(video="clip.mp4", device="rocm").to_pipeline_config(
        ROCM_SPEC
    )
    assert config.device == "rocm"
    assert config.device != "cuda"


def test_the_semantic_device_of_a_rocm_spec_is_rocm():
    assert semantic_device(ROCM_SPEC) == "rocm"


def test_a_rocm_benchmark_config_is_accepted_by_sentinels_own_resolver(rocm_host):
    """End to end: the config the harness builds must actually run on AMD.

    Asserting the string alone would not prove much — the point is that the
    downstream resolver, the one that refused before, now resolves it, hands
    torch the "cuda" string a ROCm build wants, and still reports AMD.
    """
    config = BenchmarkConfig(video="clip.mp4", device="rocm").to_pipeline_config(
        accel_module.resolve_device("rocm")
    )
    assert config.device == "rocm"
    assert yolo_backend.resolve_device(config.device) == "cuda"
    assert yolo_backend.describe_accelerator() == "rocm"


def test_the_boundary_no_longer_refuses_a_rocm_run(rocm_host):
    """The exact failure that was reported: DeviceUnavailable on an AMD host."""
    from app.accel import DeviceUnavailable

    config = BenchmarkConfig(video="clip.mp4", device="rocm").to_pipeline_config(
        accel_module.resolve_device("rocm")
    )
    try:
        yolo_backend.resolve_device(config.device)
    except DeviceUnavailable as exc:  # pragma: no cover - the bug, if it returns
        pytest.fail(f"the benchmark's own device request was refused: {exc}")


def test_rocm_is_still_never_reported_as_cuda(rocm_host):
    """Fixing the request string must not blur the reporting."""
    spec = accel_module.resolve_device("rocm")
    assert spec.kind == "rocm"
    assert spec.label == "AMD ROCm / HIP"
    assert "NVIDIA" not in spec.describe()
    assert spec.torch_device == "cuda", "torch still needs the CUDA API string"


@pytest.mark.parametrize(
    "spec,expected",
    [
        (CPU, "cpu"),
        (DeviceSpec(kind="cuda", torch_device="cuda", name="RTX 4090", index=0), "cuda"),
        (DeviceSpec(kind="mps", torch_device="mps", name="mps"), "mps"),
        (ROCM_SPEC, "rocm"),
    ],
)
def test_every_device_kind_crosses_the_boundary_as_itself(spec, expected):
    config = BenchmarkConfig(video="clip.mp4").to_pipeline_config(spec)
    assert config.device == expected


def test_an_explicit_device_index_survives_the_boundary():
    """"cuda:1" collapsed to "cuda" would time GPU 0 and report GPU 1."""
    spec = DeviceSpec(kind="rocm", torch_device="cuda:1", name="MI300X", index=1)
    config = BenchmarkConfig(video="clip.mp4", device="cuda:1").to_pipeline_config(spec)
    assert config.device == "cuda:1"


def test_no_torch_device_string_ever_reaches_a_pipeline_config():
    """Guard the boundary at the source level, so it cannot regress quietly.

    The rule is narrow and exact: ``torch_device`` is the right thing to hand
    *torch* (``__main__`` allocates its probe tensor with it) and the wrong
    thing to hand ``PipelineConfig``. So this checks every PipelineConfig
    construction in the package, not every mention of the attribute.
    """
    import os
    import re

    import app.benchmark

    package = os.path.dirname(os.path.abspath(app.benchmark.__file__))
    constructions = []
    for name in sorted(os.listdir(package)):
        if not name.endswith(".py"):
            continue
        path = os.path.join(package, name)
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        for match in re.finditer(r"PipelineConfig\(", source):
            depth, index = 0, match.end() - 1
            while index < len(source):
                depth += {"(": 1, ")": -1}.get(source[index], 0)
                if depth == 0:
                    break
                index += 1
            constructions.append((f"app/benchmark/{name}", source[match.end() : index]))

    assert constructions, "expected at least one PipelineConfig construction"
    for where, body in constructions:
        assert "torch_device" not in body, (
            f"{where} builds a PipelineConfig from a torch device string; "
            f"PipelineConfig.device is a semantic request — see semantic_device()."
        )


def test_the_parity_path_uses_the_same_fixed_boundary():
    """--parity builds its pipeline through to_pipeline_config, so it inherits."""
    import inspect

    from app.benchmark.__main__ import evidence_for

    source = inspect.getsource(evidence_for)
    assert "to_pipeline_config" in source
    assert "torch_device" not in source
