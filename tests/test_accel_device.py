"""Device selection and AMD/ROCm detection (M0.8 sections A, B, J).

Every test here runs on any machine: the torch runtime is faked, so ROCm
behaviour is exercised in full with no AMD hardware present. That is the point
— the AMD path must be testable and reviewable before it ever meets a GPU.
"""

from __future__ import annotations

import types

import pytest

from app.accel import device as accel
from app.accel.device import (
    CPU,
    KIND_CPU,
    KIND_CUDA,
    KIND_LABELS,
    KIND_MPS,
    KIND_ROCM,
    DeviceSpec,
    DeviceUnavailable,
    available_devices,
    gpu_kind,
    hip_version,
    is_rocm_build,
    resolve_device,
    verify_execution,
)


def fake_torch(
    *,
    cuda_available: bool = False,
    hip: str = None,
    cuda: str = None,
    names=(),
    mps: bool = False,
):
    """A stand-in torch module shaped like the real one.

    A ROCm build is exactly this: ``cuda.is_available()`` true, ``version.hip``
    set, ``version.cuda`` empty. That the CUDA namespace answers for an AMD GPU
    is not a mistake in the fake — it is how ROCm PyTorch really behaves.
    """
    module = types.ModuleType("torch")
    module.__version__ = "2.14.0+rocm6.2" if hip else "2.14.0+cu130"

    cuda_ns = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        device_count=lambda: len(names),
        get_device_name=lambda index: names[index],
    )
    module.cuda = cuda_ns
    module.version = types.SimpleNamespace(hip=hip, cuda=cuda)
    module.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: mps)
    )
    return module


ROCM = fake_torch(
    cuda_available=True, hip="6.2.41134", names=("AMD Instinct MI300X",)
)
NVIDIA = fake_torch(
    cuda_available=True, cuda="13.0", names=("NVIDIA GeForce RTX 4090",)
)
CPU_ONLY = fake_torch(cuda_available=False, cuda="13.0")


@pytest.fixture
def no_torch(monkeypatch):
    """Simulate a machine with no PyTorch installed at all.

    Passing ``torch_module=None`` would mean "detect it", which on this machine
    finds the real torch — so the absence has to be simulated at the lookup.
    """
    monkeypatch.setattr(accel, "_torch", lambda: None)


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------
def test_cpu_is_always_selectable():
    assert resolve_device("cpu", torch_module=CPU_ONLY) == CPU
    assert resolve_device("cpu", torch_module=ROCM).kind == KIND_CPU


def test_cpu_is_selectable_without_torch_at_all(no_torch):
    """Sentinel must run on a machine that has never seen a model runtime."""
    assert resolve_device("cpu").kind == KIND_CPU
    assert resolve_device("auto").kind == KIND_CPU
    assert available_devices() == [CPU]


def test_auto_falls_back_to_cpu_when_no_accelerator_exists():
    assert resolve_device("auto", torch_module=CPU_ONLY).kind == KIND_CPU


def test_the_cpu_spec_reports_itself_as_cpu():
    assert CPU.label == "CPU"
    assert not CPU.is_accelerator
    assert not CPU.is_amd


# ---------------------------------------------------------------------------
# ROCm
# ---------------------------------------------------------------------------
def test_a_rocm_build_is_detected_through_the_hip_version():
    assert is_rocm_build(ROCM)
    assert hip_version(ROCM) == "6.2.41134"
    assert not is_rocm_build(NVIDIA)
    assert hip_version(NVIDIA) is None


def test_an_amd_gpu_is_never_reported_as_nvidia_cuda():
    """The whole reason this module exists."""
    spec = resolve_device("auto", torch_module=ROCM)
    assert spec.kind == KIND_ROCM
    assert spec.label == "AMD ROCm / HIP"
    assert "CUDA" not in spec.describe()
    assert "NVIDIA" not in spec.describe()
    assert spec.is_amd


def test_an_amd_gpu_still_uses_the_cuda_device_string_for_torch():
    """ROCm PyTorch expects "cuda"; reporting and addressing are different jobs."""
    spec = resolve_device("rocm", torch_module=ROCM)
    assert spec.torch_device == "cuda"
    assert spec.kind == KIND_ROCM


def test_an_amd_gpu_reports_its_name_and_hip_runtime():
    spec = resolve_device("rocm", torch_module=ROCM)
    assert spec.name == "AMD Instinct MI300X"
    assert spec.runtime_version == "6.2.41134"
    assert "6.2.41134" in spec.describe()


def test_auto_prefers_the_accelerator_over_the_cpu():
    assert resolve_device("auto", torch_module=ROCM).kind == KIND_ROCM
    assert resolve_device("auto", torch_module=NVIDIA).kind == KIND_CUDA


def test_available_devices_lists_the_cpu_alongside_the_gpu():
    devices = available_devices(torch_module=ROCM)
    assert [d.kind for d in devices] == [KIND_CPU, KIND_ROCM]


def test_a_nvidia_gpu_is_reported_as_nvidia():
    spec = resolve_device("auto", torch_module=NVIDIA)
    assert spec.kind == KIND_CUDA
    assert spec.label == "NVIDIA CUDA"
    assert spec.runtime_version == "13.0"


# ---------------------------------------------------------------------------
# Refusals — a named device is never quietly substituted
# ---------------------------------------------------------------------------
def test_requesting_rocm_without_a_gpu_raises_rather_than_falling_back():
    with pytest.raises(DeviceUnavailable, match="no GPU is available"):
        resolve_device("rocm", torch_module=CPU_ONLY)


def test_requesting_rocm_on_an_nvidia_machine_refuses_to_run_on_cuda():
    """A benchmark that silently ran on the other vendor would be worthless."""
    with pytest.raises(DeviceUnavailable, match="refusing to run"):
        resolve_device("rocm", torch_module=NVIDIA)


def test_requesting_cuda_on_an_amd_machine_refuses_too():
    with pytest.raises(DeviceUnavailable, match="refusing to run"):
        resolve_device("cuda", torch_module=ROCM)


def test_requesting_rocm_without_torch_raises(no_torch):
    with pytest.raises(DeviceUnavailable):
        resolve_device("rocm")


def test_an_unknown_device_name_is_rejected_with_the_valid_ones():
    with pytest.raises(DeviceUnavailable, match="unknown device"):
        resolve_device("gpu-please", torch_module=ROCM)
    with pytest.raises(DeviceUnavailable, match="rocm"):
        resolve_device("nvidia", torch_module=ROCM)


def test_mps_is_refused_when_unavailable():
    with pytest.raises(DeviceUnavailable, match="MPS"):
        resolve_device("mps", torch_module=CPU_ONLY)


def test_mps_resolves_when_present():
    spec = resolve_device("mps", torch_module=fake_torch(mps=True))
    assert spec.kind == KIND_MPS
    assert spec.torch_device == "mps"


# ---------------------------------------------------------------------------
# Explicit torch device strings
# ---------------------------------------------------------------------------
def test_an_explicit_device_index_keeps_the_correct_vendor_label():
    torch_module = fake_torch(
        cuda_available=True, hip="6.2", names=("MI300X", "MI300X")
    )
    spec = resolve_device("cuda:1", torch_module=torch_module)
    assert spec.kind == KIND_ROCM
    assert spec.torch_device == "cuda:1"
    assert spec.index == 1


def test_multiple_gpus_are_addressed_individually():
    torch_module = fake_torch(cuda_available=True, hip="6.2", names=("a", "b"))
    devices = available_devices(torch_module=torch_module)
    assert [d.torch_device for d in devices] == ["cpu", "cuda:0", "cuda:1"]


def test_a_runtime_that_cannot_name_its_devices_is_still_selectable():
    """Failing to read a label is not failing to have a device."""
    torch_module = fake_torch(cuda_available=True, hip="6.2")
    del torch_module.cuda.device_count
    assert accel.device_names(torch_module) == []
    assert resolve_device("rocm", torch_module=torch_module).kind == KIND_ROCM


# ---------------------------------------------------------------------------
# Proof of execution
# ---------------------------------------------------------------------------
def test_execution_on_the_cpu_is_verified_without_torch(no_torch):
    ok, detail = verify_execution(CPU)
    assert ok is True
    assert "torch" in detail


def test_a_gpu_claim_is_not_accepted_without_torch(no_torch):
    spec = DeviceSpec(kind=KIND_ROCM, torch_device="cuda", name="MI300X")
    ok, _detail = verify_execution(spec)
    assert ok is False


def test_gpu_kind_is_none_without_torch(no_torch):
    assert gpu_kind() is None


def test_execution_is_verified_by_actually_computing_something():
    """The check must fail when the runtime lies about the result."""

    class BrokenTensor:
        device = "cuda"

        def __matmul__(self, other):
            return self

        def sum(self):
            return self

        def item(self):
            return 0.0

    torch_module = fake_torch(cuda_available=True, hip="6.2", names=("MI300X",))
    torch_module.ones = lambda *a, **k: BrokenTensor()
    spec = resolve_device("rocm", torch_module=torch_module)
    ok, detail = verify_execution(spec, torch_module=torch_module)
    assert ok is False
    assert "expected" in detail


def test_execution_verification_reports_where_the_tensor_landed():
    class Tensor:
        device = "cuda:0"

        def __matmul__(self, other):
            return self

        def sum(self):
            return self

        def item(self):
            return 512.0

    torch_module = fake_torch(cuda_available=True, hip="6.2", names=("MI300X",))
    torch_module.ones = lambda *a, **k: Tensor()
    spec = resolve_device("rocm", torch_module=torch_module)
    ok, detail = verify_execution(spec, torch_module=torch_module)
    assert ok is True
    assert "cuda:0" in detail


def test_execution_failure_is_reported_not_raised():
    torch_module = fake_torch(cuda_available=True, hip="6.2", names=("MI300X",))

    def explode(*args, **kwargs):
        raise RuntimeError("HIP error: no ROCm-capable device is detected")

    torch_module.ones = explode
    spec = resolve_device("rocm", torch_module=torch_module)
    ok, detail = verify_execution(spec, torch_module=torch_module)
    assert ok is False
    assert "HIP error" in detail


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def test_every_kind_has_a_distinct_human_label():
    labels = [KIND_LABELS[k] for k in (KIND_CPU, KIND_ROCM, KIND_CUDA, KIND_MPS)]
    assert len(set(labels)) == len(labels)
    assert KIND_LABELS[KIND_ROCM] != KIND_LABELS[KIND_CUDA]


def test_a_device_spec_serialises():
    spec = resolve_device("rocm", torch_module=ROCM)
    payload = spec.to_dict()
    assert payload["kind"] == "rocm"
    assert payload["label"] == "AMD ROCm / HIP"
    assert payload["torch_device"] == "cuda"


def test_gpu_kind_is_none_on_a_cpu_only_machine():
    assert gpu_kind(CPU_ONLY) is None
