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

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDS = os.path.join(ROOT, "benchmarks")
AMD_RECORD = "m08-amd-mi300x-fp32"


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


def test_the_readme_no_longer_claims_the_amd_benchmark_was_not_run(readme):
    amd_section = readme[readme.index("### AMD benchmark results") :]
    amd_section = amd_section[: amd_section.index("### Correctness parity")]
    assert "NOT_RUN" not in amd_section
    assert "Measured on real AMD hardware" in amd_section
