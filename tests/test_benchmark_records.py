"""The durable benchmark records under benchmarks/ (M0.8 evidence).

A published throughput number is a claim, and these tests hold that claim to
the rules the records themselves state: the precision is named, model load is
excluded, both devices ran the same workload, the JSON and the prose carry the
same figures, and nothing is dressed up as more general than it is.

The point is not to re-measure anything — the numbers came from real hardware
this repository cannot reach. The point is that they cannot quietly lose their
qualifiers, or drift apart between the two files, as the README is edited.
"""

from __future__ import annotations

import json
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDS = os.path.join(ROOT, "benchmarks")
AMD_RECORD = "m08-amd-mi300x-fp32"
CPU_RECORD = "m08-apple-silicon-cpu-fp32-2026-11-06"
FULL_RECORD = "m08-apple-silicon-cpu-fp32-full-clip-2026-11-06"


def read(name: str) -> str:
    with open(os.path.join(RECORDS, name), "r", encoding="utf-8") as handle:
        return handle.read()


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(read(f"{AMD_RECORD}.json"))


@pytest.fixture(scope="module")
def prose() -> str:
    return read(f"{AMD_RECORD}.md")


@pytest.fixture(scope="module")
def cpu_record() -> dict:
    return json.loads(read(f"{CPU_RECORD}.json"))


@pytest.fixture(scope="module")
def cpu_prose() -> str:
    return read(f"{CPU_RECORD}.md")


@pytest.fixture(scope="module")
def full_record() -> dict:
    return json.loads(read(f"{FULL_RECORD}.json"))


@pytest.fixture(scope="module")
def full_prose() -> str:
    return read(f"{FULL_RECORD}.md")


@pytest.fixture(scope="module")
def readme() -> str:
    with open(os.path.join(ROOT, "README.md"), "r", encoding="utf-8") as handle:
        return handle.read()


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------
def test_the_records_directory_has_an_index():
    index = read("README.md")
    assert AMD_RECORD in index
    assert "measured" in index.lower()


def test_each_record_is_a_json_and_prose_pair():
    for suffix in (".json", ".md"):
        assert os.path.isfile(os.path.join(RECORDS, AMD_RECORD + suffix))


def test_the_record_declares_what_kind_of_record_it_is(record):
    """A transcribed report must say so rather than pose as harness output."""
    assert record["record_kind"] == "transcribed_report"
    assert record["status"] == "measured_on_real_hardware"
    assert "app.benchmark" in record["record_note"]


# ---------------------------------------------------------------------------
# The disclosures that make a number readable
# ---------------------------------------------------------------------------
def test_the_precision_is_stated_and_is_fp32(record, prose, readme):
    assert record["workload"]["precision"] == "FP32"
    assert record["workload"]["fp16_used"] is False
    assert "FP32" in prose
    assert "**Precision** | **FP32**" in prose


def test_no_fp16_claim_is_made_anywhere_in_the_record(record, prose):
    """FP16 may be named only to say it was not used."""
    payload = json.dumps(record)
    for text in (payload, prose):
        for line in text.splitlines():
            if "FP16" in line or "fp16" in line:
                assert any(
                    marker in line.lower()
                    for marker in ("no fp16", "not ", "false", "never")
                ), f"unqualified FP16 mention: {line.strip()}"


def test_model_load_is_excluded_from_throughput(record, prose):
    assert record["workload"]["model_load_excluded_from_throughput"] is True
    assert "excluded from every throughput figure" in prose


def test_warmup_is_recorded_and_discarded(record, prose):
    assert record["workload"]["warmup_frames"] == 5
    assert record["workload"]["warmup_excluded_from_measurement"] is True
    assert "discarded" in prose


def test_the_two_devices_ran_the_same_workload(record, prose):
    assert record["workload"]["identical_on_both_devices"] is True
    assert "same workload" in prose
    assert "Only\nthe device differed" in prose or "Only the device differed" in prose


def test_the_record_refuses_to_generalise(record, prose):
    """No single-clip run may read as a statement about deployments."""
    assert any("general real-world performance" in c for c in record["caveats"])
    lowered = prose.lower()
    assert "not a general real-world performance claim" in lowered
    assert "does not predict" in lowered


def test_metadata_the_run_did_not_capture_is_blank_not_invented(record):
    hardware = record["hardware"]
    for field in hardware["not_recorded"]:
        assert hardware.get(field, None) is None, (
            f"{field} is listed as not recorded but carries a value"
        )


def test_the_video_is_named_and_not_committed(record):
    assert record["workload"]["video"] == "/tmp/sentinel_benchmark.mp4"
    assert record["workload"]["video_committed_to_repository"] is False
    assert not os.path.exists(os.path.join(RECORDS, "sentinel_benchmark.mp4"))


# ---------------------------------------------------------------------------
# The numbers themselves
# ---------------------------------------------------------------------------
MEASURED = {
    "cpu": {"inference_fps": 76.03, "pipeline_fps": 73.26},
    "rocm": {"inference_fps": 147.14, "pipeline_fps": 138.45},
}


@pytest.mark.parametrize("device", sorted(MEASURED))
def test_the_measured_throughput_is_preserved_exactly(record, device):
    for field, value in MEASURED[device].items():
        assert record["results"][device][field] == value


def test_the_stated_speedups_match_the_stated_throughput(record):
    """The ratios must follow from the figures, to the precision published."""
    speedup = record["speedup"]
    for stage, key in (("inference", "inference_fps"), ("pipeline", "pipeline_fps")):
        ratio = record["results"]["rocm"][key] / record["results"]["cpu"][key]
        assert round(ratio, 2) == speedup[stage]


def test_the_speedup_says_what_it_is_a_ratio_of(record):
    basis = record["speedup"]["basis"]
    assert "same host" in basis and "same workload" in basis


def test_the_parity_figures_are_identical_on_both_devices(record):
    cpu, rocm = record["results"]["cpu"], record["results"]["rocm"]
    for field in ("detections", "tracks", "events", "mean_confidence", "risk_score", "severity"):
        assert cpu[field] == rocm[field], f"{field} differs between devices"
    assert (cpu["detections"], cpu["tracks"], cpu["events"]) == (320, 9, 38)
    assert cpu["mean_confidence"] == 0.6245
    assert cpu["risk_score"] == 47.44
    assert cpu["severity"] == "medium"


def test_parity_is_recorded_as_a_pass_with_risk_semantics_unchanged(record):
    assert record["parity"]["verdict"] == "PASS"
    assert record["parity"]["risk_semantics_unchanged"] is True


def test_the_hardware_and_runtime_are_named(record, prose):
    assert record["hardware"]["accelerator"] == "AMD Instinct MI300X VF"
    assert record["hardware"]["rocm_hip_runtime"] == "7.1.52802"
    assert record["hardware"]["accelerator_label"] == "AMD ROCm / HIP"
    assert "MI300X" in prose
    assert "7.1.52802" in prose


def test_the_accelerator_is_never_labelled_cuda(record, prose):
    assert "CUDA" not in json.dumps(record)
    assert record["hardware"]["accelerator_kind"] == "rocm"


def test_the_frame_count_is_recorded(record, prose):
    assert record["workload"]["frames_measured"] == 120
    assert record["workload"]["model"] == "YOLOv8n"


# ---------------------------------------------------------------------------
# The README must agree with the record
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "figure", ["76.03", "147.14", "73.26", "138.45", "1.94", "1.89", "47.44", "7.1.52802"]
)
def test_every_published_figure_appears_in_the_readme(readme, figure):
    assert figure in readme


def test_the_readme_links_to_the_durable_record(readme):
    assert f"benchmarks/{AMD_RECORD}.md" in readme
    assert f"benchmarks/{AMD_RECORD}.json" in readme


def test_the_readme_keeps_the_fp32_and_model_load_qualifiers(readme):
    amd_section = readme[readme.index("### AMD benchmark results") :]
    amd_section = amd_section[: amd_section.index("### Correctness parity")]
    assert "FP32" in amd_section
    assert "Model load time is excluded" in amd_section
    assert "same workload" in amd_section
    assert "not a claim about general" in amd_section


def test_the_readme_does_not_conflate_the_two_cpu_baselines(readme):
    """29.90 FPS (container) and 76.03 FPS (AMD host) are different machines."""
    assert "not** the AMD host" in readme
    assert "29.90" in readme and "76.03" in readme


def test_the_readme_dates_the_amd_run_to_hardware_that_is_gone(readme):
    """The figures stay; the implication that the device is still there does not.

    This replaces an assertion that the README no longer said `NOT_RUN`, which
    held while an AMD host existed. The credits expired and the droplet was
    deleted, so `NOT_RUN` is the current state again and the recorded figures
    have to read as a dated past run rather than a live capability.
    """
    amd_section = readme[readme.index("### AMD benchmark results") :]
    amd_section = amd_section[: amd_section.index("### Correctness parity")]
    assert "no longer available" in amd_section
    assert "NOT_RUN" in amd_section
    assert "**Measured on real AMD hardware.**" not in amd_section


def test_the_readme_leads_with_the_current_cpu_only_status(readme):
    """A reader must meet the current state before the MI300X figures."""
    status = readme[readme.index("### Current benchmark status") :]
    status = status[: status.index("### Why Sentinel benefits")]
    assert "CPU only" in status
    assert "AMD_BENCHMARK = NOT_RUN" in status
    assert "no AMD ROCm device available" in status
    assert readme.index("### Current benchmark status") < readme.index(
        "### AMD benchmark results"
    )


def test_the_record_index_says_no_amd_run_is_possible_now():
    index = read("README.md")
    assert "AMD_BENCHMARK = NOT_RUN" in index
    assert "no longer available" in index


# ===========================================================================
# The local Apple Silicon CPU record (2026-11-06)
#
# Transcribed from one `--device cpu --json` run. These tests do not re-measure
# anything: they hold the record to the same rules as its sibling — figures
# preserved exactly, disclosures present, nothing generalised — and add the one
# rule a single-device record needs, which is that it compares itself to
# nothing.
# ===========================================================================
CPU_TIMINGS = {
    "inference": {
        "fps": 55.01, "mean_ms": 18.18, "p50_ms": 18.102, "p95_ms": 19.123,
        "p99_ms": 19.486, "min_ms": 17.055, "max_ms": 19.573, "stdev_ms": 0.538,
        "total_seconds": 2.1816,
    },
    "pipeline": {
        "fps": 50.45, "mean_ms": 19.82, "p50_ms": 19.755, "p95_ms": 20.726,
        "p99_ms": 20.958, "min_ms": 18.706, "max_ms": 21.589, "stdev_ms": 0.553,
        "total_seconds": 2.3784,
    },
    "decode": {
        "fps": 621.08, "mean_ms": 1.61, "p50_ms": 1.568, "p95_ms": 1.947,
        "p99_ms": 2.119, "min_ms": 1.405, "max_ms": 2.321, "stdev_ms": 0.166,
        "total_seconds": 0.1932,
    },
    "tracking": {
        "fps": 56548.08, "mean_ms": 0.018, "p50_ms": 0.017, "p95_ms": 0.025,
        "p99_ms": 0.027, "min_ms": 0.012, "max_ms": 0.031, "stdev_ms": 0.003,
        "total_seconds": 0.0021,
    },
    "events": {
        "fps": 82002.28, "mean_ms": 0.012, "p50_ms": 0.011, "p95_ms": 0.025,
        "p99_ms": 0.029, "min_ms": 0.007, "max_ms": 0.03, "stdev_ms": 0.004,
        "total_seconds": 0.0015,
    },
}


def test_the_cpu_record_is_a_json_and_prose_pair():
    for suffix in (".json", ".md"):
        assert os.path.isfile(os.path.join(RECORDS, CPU_RECORD + suffix))


def test_the_cpu_record_is_indexed_with_its_date():
    index = read("README.md")
    assert CPU_RECORD in index
    assert "2026-11-06" in index


@pytest.mark.parametrize("stage", sorted(CPU_TIMINGS))
def test_every_cpu_timing_is_preserved_exactly(cpu_record, stage):
    recorded = cpu_record["timings"][stage]
    for field, value in CPU_TIMINGS[stage].items():
        assert recorded[field] == value, f"{stage}.{field} drifted"
    assert recorded["frames"] == 120


@pytest.mark.parametrize("stage", ["inference", "pipeline"])
def test_the_cpu_throughput_follows_from_the_measured_totals(cpu_record, stage):
    """The FPS must be the frame count over the time actually spent.

    Only the two slow stages are checked: `total_seconds` is rounded to four
    decimals, which at 0.0021 s dominates the ratio and would make the fast
    stages fail an exactness test for no reason.
    """
    timing = cpu_record["timings"][stage]
    assert round(timing["frames"] / timing["total_seconds"], 2) == timing["fps"]
    assert round(timing["total_seconds"] / timing["frames"] * 1000, 2) == timing["mean_ms"]


def test_the_cpu_percentiles_are_ordered(cpu_record):
    for stage, timing in cpu_record["timings"].items():
        assert timing["min_ms"] <= timing["p50_ms"] <= timing["p95_ms"], stage
        assert timing["p95_ms"] <= timing["p99_ms"] <= timing["max_ms"], stage


def test_the_cpu_class_counts_account_for_every_detection(cpu_record):
    results = cpu_record["results"]
    assert sum(results["class_counts"].values()) == results["detections"] == 213
    assert results["tracks_created"] == 2
    assert results["events"] == 7
    assert results["mean_confidence"] == 0.6172


def test_the_cpu_record_names_the_device_it_verified(cpu_record, cpu_prose):
    device = cpu_record["device"]
    assert device["requested"] == "cpu"
    assert device["kind"] == "cpu"
    assert device["torch_device"] == "cpu"
    assert device["verified"] is True
    assert "CPU" in cpu_prose


def test_the_cpu_record_says_metal_was_available_and_unused(cpu_record, cpu_prose):
    """A CPU number from a machine with a GPU has to say the GPU sat idle."""
    accelerator = cpu_record["accelerator"]
    assert accelerator["mps_available"] is True
    assert accelerator["mps_used"] is False
    assert accelerator["used"] is False
    assert "deliberately not used" in cpu_prose


def test_the_cpu_record_is_fp32_and_says_so(cpu_record, cpu_prose):
    assert cpu_record["workload"]["precision"] == "FP32"
    assert cpu_record["workload"]["fp16_used"] is False
    assert "**FP32**" in cpu_prose


def test_the_cpu_record_excludes_model_load_and_discards_warmup(cpu_record, cpu_prose):
    workload = cpu_record["workload"]
    assert workload["model_load_excluded_from_throughput"] is True
    assert workload["model_load_seconds"] == 0.089
    assert workload["warmup_frames"] == 5
    assert workload["warmup_excluded_from_measurement"] is True
    assert "excluded from every throughput figure" in cpu_prose
    assert "discarded" in cpu_prose


def test_the_cpu_record_states_how_much_of_the_clip_it_measured(cpu_record, cpu_prose):
    """120 frames of a 62.892s clip is not the clip, and must not read as it."""
    clip = cpu_record["clip"]
    assert clip["frame_count"] == 2105
    assert clip["duration_seconds"] == 62.892
    assert cpu_record["workload"]["frames_measured"] == 120
    assert clip["frame_count"] > cpu_record["workload"]["frames_measured"]
    assert "2105" in cpu_prose and "120" in cpu_prose


def test_the_cpu_clip_is_named_but_not_committed(cpu_record):
    workload = cpu_record["workload"]
    assert workload["video"] == "Screen Recording 2026-09-29 at 09.48.17.mov"
    assert workload["video_committed_to_repository"] is False
    assert workload["video_absolute_path_recorded"] is False
    assert not os.path.exists(os.path.join(RECORDS, workload["video"]))


def test_the_cpu_record_carries_no_home_directory_path(cpu_record, cpu_prose):
    """The path the harness reported was a home directory; only the name is kept."""
    payload = json.dumps(cpu_record)
    for text in (payload, cpu_prose):
        assert "/Users/" not in text


MI300X_FIGURES = (147.14, 138.45, 1.94, 1.89, 76.03, 73.26, 47.44)


def _numeric_leaves(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _numeric_leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from _numeric_leaves(item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)


def test_the_cpu_record_compares_itself_to_nothing(cpu_record, cpu_prose):
    """A single-device run has no speedup, no parity and no second device.

    MI300X may be *named* — saying the two records are not comparable is the
    point — but none of its figures may appear, and no ratio may exist here.
    Values are compared as numbers, not as substrings: decode's p95 is 1.947 ms,
    which contains "1.94" and is nobody's speedup.
    """
    assert cpu_record["comparison"] is None
    assert "speedup" not in json.dumps(cpu_record).lower().replace(
        "no speedup", ""
    ), "a ratio field survives in a single-device record"
    assert set(cpu_record["results"]) == {
        "detections",
        "tracks_created",
        "events",
        "class_counts",
        "mean_confidence",
    }

    leaked = set(_numeric_leaves(cpu_record)) & set(MI300X_FIGURES)
    assert not leaked, f"MI300X figures present as values: {sorted(leaked)}"

    for figure in MI300X_FIGURES:
        pattern = rf"(?<![\d.]){re.escape(f'{figure:g}')}(?![\d])"
        assert not re.search(pattern, cpu_prose), f"MI300X figure {figure} in the prose"

    flat = " ".join(cpu_prose.split())
    assert "No second device was benchmarked" in flat
    assert "not to be read against these numbers" in flat


def test_the_cpu_record_claims_no_acceleration(cpu_record, cpu_prose):
    for text in (json.dumps(cpu_record), cpu_prose):
        assert "ROCm" not in text or "no ROCm" in text or "No MPS, no ROCm" in text
    assert cpu_record["accelerator"]["rocm_build"] is False
    assert cpu_record["accelerator"]["gpu_kind"] is None
    assert cpu_record["accelerator"]["gpus"] == []


def test_the_cpu_record_refuses_to_generalise(cpu_record, cpu_prose):
    assert any("general real-world performance" in c for c in cpu_record["caveats"])
    lowered = cpu_prose.lower()
    assert "not a general real-world performance claim" in lowered
    assert "does not predict" in lowered


def test_the_cpu_record_records_the_host_and_marks_what_it_lacks(cpu_record, cpu_prose):
    host = cpu_record["host"]
    assert host["os"] == "Darwin 27.0.0 (arm64)"
    assert host["cpu"] == "arm" and host["cpu_count"] == 10 and host["ram_gb"] == 16.0
    assert host["python"] == "3.14.0" and host["torch"] == "2.14.0"
    assert host["ultralytics"] == "8.4.163"
    for field in host["not_recorded"]:
        assert field not in host or host.get(field) is None
    assert "not recorded" in cpu_prose


def test_the_cpu_record_names_the_weights_it_ran(cpu_record, cpu_prose):
    digest = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
    assert cpu_record["workload"]["weights_sha256"] == digest
    assert cpu_record["workload"]["weights"] == "yolov8n.pt"
    assert digest in cpu_prose


def test_the_two_records_stay_separate(cpu_record, record):
    """Neither record may quietly absorb the other's hardware or figures."""
    assert cpu_record["record_id"] != record["record_id"]
    assert cpu_record["device"]["kind"] == "cpu"
    assert record["hardware"]["accelerator_kind"] == "rocm"
    assert cpu_record["results"]["detections"] != record["results"]["cpu"]["detections"]


# ===========================================================================
# The whole-clip Apple Silicon CPU record (2026-11-06)
#
# Same machine, clip and settings as the 120-frame record, over 2105 frames
# instead of 125. It is an addition, not a replacement: the tests below check
# its own figures, and that it neither absorbs nor supersedes its sibling.
# ===========================================================================
FULL_TIMINGS = {
    "inference": {
        "fps": 58.0, "mean_ms": 17.24, "p50_ms": 16.899, "p95_ms": 19.113,
        "p99_ms": 24.431, "min_ms": 14.584, "max_ms": 100.333, "stdev_ms": 3.337,
        "total_seconds": 36.2044,
    },
    "pipeline": {
        "fps": 53.39, "mean_ms": 18.73, "p50_ms": 18.361, "p95_ms": 20.799,
        "p99_ms": 26.774, "min_ms": 15.939, "max_ms": 102.424, "stdev_ms": 3.506,
        "total_seconds": 39.3325,
    },
    "decode": {
        "fps": 679.62, "mean_ms": 1.471, "p50_ms": 1.415, "p95_ms": 1.716,
        "p99_ms": 2.216, "min_ms": 1.219, "max_ms": 13.877, "stdev_ms": 0.393,
        "total_seconds": 3.0899,
    },
    "tracking": {
        "fps": 92263.56, "mean_ms": 0.011, "p50_ms": 0.006, "p95_ms": 0.026,
        "p99_ms": 0.032, "min_ms": 0.002, "max_ms": 0.076, "stdev_ms": 0.009,
        "total_seconds": 0.0228,
    },
    "events": {
        "fps": 144702.8, "mean_ms": 0.007, "p50_ms": 0.001, "p95_ms": 0.023,
        "p99_ms": 0.03, "min_ms": 0.001, "max_ms": 0.059, "stdev_ms": 0.008,
        "total_seconds": 0.0145,
    },
}

FULL_CLASS_COUNTS = {
    "airplane": 10, "backpack": 2, "bench": 3, "bird": 9, "boat": 152,
    "car": 41, "dog": 12, "fire hydrant": 16, "horse": 10, "kite": 4,
    "motorcycle": 72, "person": 687, "skateboard": 13, "suitcase": 14,
    "surfboard": 58, "train": 1, "truck": 654, "umbrella": 2,
}


def test_the_full_clip_record_is_a_json_and_prose_pair():
    for suffix in (".json", ".md"):
        assert os.path.isfile(os.path.join(RECORDS, FULL_RECORD + suffix))


def test_the_full_clip_record_is_indexed_beside_its_sibling():
    index = read("README.md")
    assert FULL_RECORD in index
    assert CPU_RECORD in index, "the 120-frame record must stay in the index"
    assert "whole 2105-frame clip" in index
    assert "first 120 frames" in index


@pytest.mark.parametrize("stage", sorted(FULL_TIMINGS))
def test_every_full_clip_timing_is_preserved_exactly(full_record, stage):
    recorded = full_record["timings"][stage]
    for field, value in FULL_TIMINGS[stage].items():
        assert recorded[field] == value, f"{stage}.{field} drifted"
    assert recorded["frames"] == 2100


@pytest.mark.parametrize("stage", ["inference", "pipeline", "decode"])
def test_the_full_clip_throughput_follows_from_the_measured_totals(full_record, stage):
    """FPS and mean must be the frame count over the time actually spent.

    The two sub-millisecond stages are excluded: their `total_seconds` is
    rounded to four decimals, which at 0.0145 s dominates the ratio.
    """
    timing = full_record["timings"][stage]
    assert round(timing["frames"] / timing["total_seconds"], 2) == pytest.approx(
        timing["fps"], abs=0.01
    )
    assert round(timing["total_seconds"] / timing["frames"] * 1000, 3) == pytest.approx(
        timing["mean_ms"], abs=0.001
    )


def test_the_full_clip_percentiles_are_ordered(full_record):
    for stage, timing in full_record["timings"].items():
        assert timing["min_ms"] <= timing["p50_ms"] <= timing["p95_ms"], stage
        assert timing["p95_ms"] <= timing["p99_ms"] <= timing["max_ms"], stage


def test_the_full_clip_record_discloses_its_tail(full_record, full_prose):
    """A max six times the median is the kind of thing a record must not bury."""
    inference = full_record["timings"]["inference"]
    assert inference["max_ms"] / inference["p50_ms"] > 5
    assert "tail" in full_record["timing_note"].lower()
    assert "100.333" in full_prose and "102.424" in full_prose


def test_the_full_clip_class_counts_account_for_every_detection(full_record):
    results = full_record["results"]
    assert results["class_counts"] == FULL_CLASS_COUNTS
    assert sum(results["class_counts"].values()) == results["detections"] == 1760
    assert results["tracks_created"] == 26
    assert results["events"] == 331
    assert results["mean_confidence"] == 0.51423


def test_the_full_clip_record_read_the_whole_clip(full_record, full_prose):
    workload, clip = full_record["workload"], full_record["clip"]
    assert workload["frames_measured"] == 2100
    assert workload["warmup_frames"] == 5
    assert workload["frames_read"] == 2105
    assert workload["frames_read"] == clip["frame_count"]
    assert clip["fully_covered"] is True
    assert "the entire clip" in full_prose or "the whole clip" in full_prose


def test_the_full_clip_record_names_the_device_it_verified(full_record):
    device = full_record["device"]
    assert device["requested"] == "cpu"
    assert device["kind"] == "cpu"
    assert device["verified"] is True


def test_the_full_clip_record_says_metal_was_available_and_unused(full_record, full_prose):
    accelerator = full_record["accelerator"]
    assert accelerator["mps_available"] is True
    assert accelerator["mps_used"] is False
    assert accelerator["used"] is False
    assert "deliberately not used" in full_prose


def test_the_full_clip_record_is_fp32_and_excludes_model_load(full_record, full_prose):
    workload = full_record["workload"]
    assert workload["precision"] == "FP32"
    assert workload["fp16_used"] is False
    assert workload["model_load_excluded_from_throughput"] is True
    assert workload["model_load_seconds"] == 0.092
    assert workload["warmup_excluded_from_measurement"] is True
    assert "**FP32**" in full_prose
    assert "excluded from every throughput figure" in full_prose


def test_the_full_clip_record_carries_no_home_directory_path(full_record, full_prose):
    assert full_record["workload"]["video_absolute_path_recorded"] is False
    for text in (json.dumps(full_record), full_prose):
        assert "/Users/" not in text


def test_the_full_clip_record_compares_itself_to_no_other_device(full_record, full_prose):
    assert full_record["comparison"] is None
    leaked = set(_numeric_leaves(full_record)) & set(MI300X_FIGURES)
    assert not leaked, f"MI300X figures present as values: {sorted(leaked)}"
    for figure in MI300X_FIGURES:
        pattern = rf"(?<![\d.]){re.escape(f'{figure:g}')}(?![\d])"
        assert not re.search(pattern, full_prose), f"MI300X figure {figure} in the prose"


def test_the_full_clip_record_points_at_its_sibling_without_ranking_it(
    full_record, full_prose
):
    """Added alongside, not instead of: it must say so and compute no ratio."""
    companion = full_record["companion_record"]
    assert companion["record_id"] == CPU_RECORD
    flat = " ".join(full_prose.split())
    assert "neither supersedes the other" in flat
    assert "no ratio between them is computed" in flat
    # The sibling's headline figures must not be restated or ratioed here.
    for figure in (55.01, 50.45):
        assert figure not in set(_numeric_leaves(full_record))


def test_the_120_frame_record_is_untouched_by_the_full_clip_run(record, cpu_record):
    """The earlier record keeps its own window and figures, unchanged."""
    assert cpu_record["workload"]["frames_measured"] == 120
    assert cpu_record["timings"]["inference"]["fps"] == 55.01
    assert cpu_record["timings"]["pipeline"]["fps"] == 50.45
    assert cpu_record["results"]["detections"] == 213
    assert "companion_record" not in cpu_record
    assert record["hardware"]["accelerator_kind"] == "rocm"


def test_the_three_records_stay_distinct(cpu_record, full_record, record):
    ids = {
        cpu_record["record_id"],
        full_record["record_id"],
        record["record_id"],
    }
    assert len(ids) == 3
    assert cpu_record["results"]["detections"] != full_record["results"]["detections"]
