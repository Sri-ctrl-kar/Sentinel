"""Device selection for the perception layer.

Sentinel spends essentially all of its compute in one place. Measured on a
4-core CPU over 160 frames of 640x384 video: detection 33.87 ms/frame,
decoding 0.64 ms, tracking 0.01 ms, event generation under 0.01 ms. Detection
is 98% of wall-clock and everything downstream is free. So this module exists
to put *detection* on an accelerator, and nothing else.

AMD is a first-class target here, which takes some care, because ROCm builds of
PyTorch expose the HIP runtime through the CUDA API surface:
``torch.cuda.is_available()`` returns ``True`` and the device string is still
``"cuda"``. Two consequences, both handled here:

* Sentinel must never *report* an AMD GPU as NVIDIA CUDA. The two are told
  apart by ``torch.version.hip``, which is set only on ROCm builds, and the
  distinction is carried in :attr:`DeviceSpec.kind`.
* A user asking for ``rocm`` on a CUDA build must get an error, not a silent
  run on the wrong stack. "It ran" and "it ran where I asked" are different
  claims, and a benchmark that confuses them is worthless.

Nothing here imports torch at module scope: importing this module on a machine
with no torch at all is free, and CPU-only operation is never at risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

#: Device kinds Sentinel distinguishes. These are *reporting* labels; the
#: string torch is given is :attr:`DeviceSpec.torch_device`.
KIND_CPU = "cpu"
KIND_ROCM = "rocm"
KIND_CUDA = "cuda"
KIND_MPS = "mps"

KINDS = (KIND_CPU, KIND_ROCM, KIND_CUDA, KIND_MPS)

#: What a human should see for each kind. An AMD card is never called CUDA.
KIND_LABELS = {
    KIND_CPU: "CPU",
    KIND_ROCM: "AMD ROCm / HIP",
    KIND_CUDA: "NVIDIA CUDA",
    KIND_MPS: "Apple Metal (MPS)",
}

#: Requests accepted from a CLI or config, beyond an explicit ``cuda:1``.
REQUESTS = ("auto",) + KINDS


class DeviceUnavailable(RuntimeError):
    """A specific device was asked for and is not present.

    Deliberately not a fallback: silently running on the CPU after a user asked
    for a GPU produces benchmark numbers that mean nothing.
    """


@dataclass(frozen=True)
class DeviceSpec:
    """A resolved compute device, and how to talk about it.

    ``torch_device`` is what torch and Ultralytics are given — for AMD that is
    still ``"cuda"``, because that is the string a ROCm build expects.
    ``kind`` is what Sentinel reports, and for AMD that is ``rocm``.
    """

    kind: str
    torch_device: str
    name: str = ""
    index: Optional[int] = None
    #: HIP runtime version on ROCm builds; CUDA toolkit version on NVIDIA.
    runtime_version: Optional[str] = None

    @property
    def label(self) -> str:
        return KIND_LABELS.get(self.kind, self.kind)

    @property
    def is_accelerator(self) -> bool:
        return self.kind != KIND_CPU

    @property
    def is_amd(self) -> bool:
        return self.kind == KIND_ROCM

    def describe(self) -> str:
        parts = [self.label]
        if self.name:
            parts.append(self.name)
        if self.runtime_version:
            parts.append(f"runtime {self.runtime_version}")
        return " — ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "torch_device": self.torch_device,
            "name": self.name,
            "index": self.index,
            "runtime_version": self.runtime_version,
        }


CPU = DeviceSpec(kind=KIND_CPU, torch_device="cpu", name="cpu")


# ---------------------------------------------------------------------------
# Runtime inspection
# ---------------------------------------------------------------------------
def _torch() -> Optional[Any]:
    """The torch module, or ``None``. Never raises: torch is optional."""
    try:
        import torch  # noqa: WPS433 - deliberately lazy and optional
    except ImportError:
        return None
    return torch


def hip_version(torch_module: Optional[Any] = None) -> Optional[str]:
    """The HIP runtime version, set only on ROCm builds of PyTorch."""
    torch_module = torch_module or _torch()
    if torch_module is None:
        return None
    version = getattr(getattr(torch_module, "version", None), "hip", None)
    return str(version) if version else None


def cuda_version(torch_module: Optional[Any] = None) -> Optional[str]:
    """The CUDA toolkit version, set only on NVIDIA builds."""
    torch_module = torch_module or _torch()
    if torch_module is None:
        return None
    version = getattr(getattr(torch_module, "version", None), "cuda", None)
    return str(version) if version else None


def is_rocm_build(torch_module: Optional[Any] = None) -> bool:
    """Is this a ROCm build of PyTorch? (Independent of a GPU being present.)"""
    return hip_version(torch_module) is not None


def gpu_kind(torch_module: Optional[Any] = None) -> Optional[str]:
    """``rocm`` or ``cuda`` when a GPU is usable, else ``None``.

    Both stacks answer ``torch.cuda.is_available()``; only ``torch.version.hip``
    separates them.
    """
    torch_module = torch_module or _torch()
    if torch_module is None:
        return None
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return None
    return KIND_ROCM if is_rocm_build(torch_module) else KIND_CUDA


def mps_available(torch_module: Optional[Any] = None) -> bool:
    torch_module = torch_module or _torch()
    if torch_module is None:
        return False
    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None) if backends else None
    return bool(mps and mps.is_available())


def device_names(torch_module: Optional[Any] = None) -> List[str]:
    """Every visible accelerator, by name. Empty on a CPU-only machine.

    Naming is best-effort and never load-bearing: a runtime that cannot
    enumerate device names (an old driver, a stub in a test) must still be
    *selectable*. Failing to read a label is not failing to have a device.
    """
    torch_module = torch_module or _torch()
    if torch_module is None:
        return []
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return []
    count_fn = getattr(cuda, "device_count", None)
    name_fn = getattr(cuda, "get_device_name", None)
    if count_fn is None or name_fn is None:
        return []
    try:
        count = int(count_fn())
    except Exception:  # pragma: no cover - driver-dependent
        return []
    names = []
    for index in range(count):
        try:
            names.append(str(name_fn(index)))
        except Exception:  # pragma: no cover - driver-dependent
            names.append(f"device {index}")
    return names


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------
def available_devices(torch_module: Optional[Any] = None) -> List[DeviceSpec]:
    """Everything this machine could actually run on, CPU always included."""
    torch_module = torch_module if torch_module is not None else _torch()
    devices = [CPU]
    kind = gpu_kind(torch_module)
    if kind is not None:
        names = device_names(torch_module)
        runtime = (
            hip_version(torch_module) if kind == KIND_ROCM else cuda_version(torch_module)
        )
        for index, name in enumerate(names or [""]):
            devices.append(
                DeviceSpec(
                    kind=kind,
                    torch_device=f"cuda:{index}" if len(names) > 1 else "cuda",
                    name=name,
                    index=index if names else None,
                    runtime_version=runtime,
                )
            )
    if mps_available(torch_module):
        devices.append(DeviceSpec(kind=KIND_MPS, torch_device="mps", name="mps"))
    return devices


def resolve_device(
    requested: str = "auto", torch_module: Optional[Any] = None
) -> DeviceSpec:
    """Resolve a device request into a concrete :class:`DeviceSpec`.

    ``auto`` prefers an accelerator and falls back to the CPU silently, because
    that is what "auto" asks for. Every *named* device is exact: asking for
    ``rocm`` on a machine without a ROCm runtime raises
    :class:`DeviceUnavailable` rather than quietly running somewhere else.
    """
    requested = (requested or "auto").strip().lower()
    torch_module = torch_module if torch_module is not None else _torch()

    if requested == "auto":
        candidates = available_devices(torch_module)
        accelerators = [d for d in candidates if d.is_accelerator]
        return accelerators[0] if accelerators else CPU

    if requested == KIND_CPU:
        return CPU

    if requested in (KIND_ROCM, KIND_CUDA):
        return _resolve_gpu(requested, torch_module)

    if requested == KIND_MPS:
        if not mps_available(torch_module):
            raise DeviceUnavailable("Apple MPS is not available on this machine")
        return DeviceSpec(kind=KIND_MPS, torch_device="mps", name="mps")

    if requested.startswith("cuda"):
        # An explicit torch device string such as "cuda:1". Honour it, but
        # still report the correct vendor.
        kind = gpu_kind(torch_module)
        if kind is None:
            raise DeviceUnavailable(
                f"{requested!r} was requested but no GPU runtime is available"
            )
        index = _index_of(requested)
        names = device_names(torch_module)
        return DeviceSpec(
            kind=kind,
            torch_device=requested,
            name=names[index] if index is not None and index < len(names) else "",
            index=index,
            runtime_version=(
                hip_version(torch_module) if kind == KIND_ROCM else cuda_version(torch_module)
            ),
        )

    raise DeviceUnavailable(
        f"unknown device {requested!r}; expected one of {', '.join(REQUESTS)} "
        "or an explicit torch device such as 'cuda:0'"
    )


def _resolve_gpu(requested: str, torch_module: Optional[Any]) -> DeviceSpec:
    kind = gpu_kind(torch_module)
    if kind is None:
        raise DeviceUnavailable(
            f"{KIND_LABELS[requested]} was requested but no GPU is available to "
            "PyTorch on this machine"
        )
    if kind != requested:
        # The crucial case: a ROCm request on a CUDA build, or the reverse.
        raise DeviceUnavailable(
            f"{KIND_LABELS[requested]} was requested but the available GPU "
            f"runtime is {KIND_LABELS[kind]}; refusing to run somewhere other "
            "than where you asked"
        )
    names = device_names(torch_module)
    runtime = hip_version(torch_module) if kind == KIND_ROCM else cuda_version(torch_module)
    return DeviceSpec(
        kind=kind,
        torch_device="cuda",
        name=names[0] if names else "",
        index=0 if names else None,
        runtime_version=runtime,
    )


def _index_of(device_string: str) -> Optional[int]:
    _, _, suffix = device_string.partition(":")
    try:
        return int(suffix)
    except ValueError:
        return None


def verify_execution(
    spec: DeviceSpec, torch_module: Optional[Any] = None
) -> Tuple[bool, str]:
    """Prove a tensor operation really runs on ``spec``.

    Returns ``(ok, detail)``. This is what stands between "we asked for AMD"
    and "AMD did the work": a benchmark may not claim acceleration unless this
    has confirmed the device actually executed something.
    """
    torch_module = torch_module if torch_module is not None else _torch()
    if torch_module is None:
        return (spec.kind == KIND_CPU, "torch is not installed")
    if spec.kind == KIND_CPU:
        return (True, "cpu")
    try:
        tensor = torch_module.ones(8, 8, device=spec.torch_device)
        result = (tensor @ tensor).sum().item()
        placed = str(tensor.device)
    except Exception as exc:  # pragma: no cover - hardware-dependent
        return (False, f"{type(exc).__name__}: {exc}")
    expected = float(8 * 8 * 8)
    if result != expected:  # pragma: no cover - would be a broken runtime
        return (False, f"arithmetic on {placed} produced {result}, expected {expected}")
    return (True, f"tensor executed on {placed}")
