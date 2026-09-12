"""Accelerator selection and environment probing (M0.8).

Used by the perception backend and the benchmark harness — and by nothing
else. The deterministic reasoning layers are pure Python by design and stay
that way: putting risk scoring on a GPU would be theatre, since it accounts for
under 0.1% of Sentinel's runtime.
"""

from __future__ import annotations

from .device import (
    CPU,
    KIND_CPU,
    KIND_CUDA,
    KIND_LABELS,
    KIND_MPS,
    KIND_ROCM,
    KINDS,
    REQUESTS,
    DeviceSpec,
    DeviceUnavailable,
    available_devices,
    cuda_version,
    device_names,
    gpu_kind,
    hip_version,
    is_rocm_build,
    mps_available,
    resolve_device,
    verify_execution,
)
from .probe import EnvironmentInfo, file_digest, probe_environment

__all__ = [
    "CPU",
    "KINDS",
    "KIND_CPU",
    "KIND_CUDA",
    "KIND_LABELS",
    "KIND_MPS",
    "KIND_ROCM",
    "REQUESTS",
    "DeviceSpec",
    "DeviceUnavailable",
    "EnvironmentInfo",
    "available_devices",
    "cuda_version",
    "device_names",
    "file_digest",
    "gpu_kind",
    "hip_version",
    "is_rocm_build",
    "mps_available",
    "probe_environment",
    "resolve_device",
    "verify_execution",
]
