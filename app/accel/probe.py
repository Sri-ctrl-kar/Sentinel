"""Environment metadata for reproducible benchmarks.

A throughput number without the machine it was measured on is a rumour. This
module collects everything needed to reproduce or fairly compare a run: OS,
Python, PyTorch, Ultralytics, OpenCV, the ROCm/HIP or CUDA runtime, GPU and CPU
names, RAM, and the weights file's own SHA-256.

Nothing here is a secret: no hostname, no user, no path outside the weights
file's basename, no environment variables.
"""

from __future__ import annotations

import hashlib
import os
import platform
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .device import (
    KIND_LABELS,
    cuda_version,
    device_names,
    gpu_kind,
    hip_version,
    is_rocm_build,
    mps_available,
)

UNKNOWN = "unknown"


def _module_version(name: str) -> Optional[str]:
    try:
        module = __import__(name)
    except Exception:  # noqa: BLE001 - a broken optional dep is not fatal here
        return None
    return str(getattr(module, "__version__", UNKNOWN))


def cpu_name() -> str:
    """A human-readable CPU model, best effort across platforms."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or UNKNOWN


def cpu_count() -> Optional[int]:
    return os.cpu_count()


def total_ram_gb() -> Optional[float]:
    """Total RAM in GiB, or ``None`` where it cannot be read portably."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None
    return round(pages * page_size / (1024**3), 1)


def file_digest(path: Optional[str], limit_bytes: int = 256 * 1024 * 1024) -> Optional[str]:
    """SHA-256 of a weights file, so two runs can be shown to use one model."""
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    read = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            read += len(chunk)
            if read > limit_bytes:  # pragma: no cover - guards absurd inputs
                return None
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class EnvironmentInfo:
    """Everything about the machine that could move a benchmark number."""

    os: str
    python: str
    torch: Optional[str] = None
    ultralytics: Optional[str] = None
    opencv: Optional[str] = None
    numpy: Optional[str] = None
    hip: Optional[str] = None
    cuda: Optional[str] = None
    rocm_build: bool = False
    gpu_kind: Optional[str] = None
    gpus: List[str] = field(default_factory=list)
    mps: bool = False
    cpu: str = UNKNOWN
    cpu_count: Optional[int] = None
    ram_gb: Optional[float] = None

    @property
    def has_amd_gpu(self) -> bool:
        return self.gpu_kind == "rocm"

    @property
    def accelerator_label(self) -> str:
        if self.gpu_kind:
            return KIND_LABELS.get(self.gpu_kind, self.gpu_kind)
        return KIND_LABELS["mps"] if self.mps else KIND_LABELS["cpu"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "os": self.os,
            "python": self.python,
            "torch": self.torch,
            "ultralytics": self.ultralytics,
            "opencv": self.opencv,
            "numpy": self.numpy,
            "hip": self.hip,
            "cuda": self.cuda,
            "rocm_build": self.rocm_build,
            "gpu_kind": self.gpu_kind,
            "gpus": list(self.gpus),
            "mps": self.mps,
            "cpu": self.cpu,
            "cpu_count": self.cpu_count,
            "ram_gb": self.ram_gb,
        }

    def lines(self) -> List[str]:
        """The same information, laid out for a terminal."""
        rows = [
            ("OS", self.os),
            ("Python", self.python),
            ("PyTorch", self.torch or "not installed"),
            ("Ultralytics", self.ultralytics or "not installed"),
            ("OpenCV", self.opencv or "not installed"),
            ("HIP available", "yes" if self.rocm_build else "no"),
            ("HIP version", self.hip or "—"),
            ("CUDA version", self.cuda or "—"),
            ("Accelerators", ", ".join(self.gpus) if self.gpus else "none visible"),
            ("CPU", self.cpu),
            ("CPU cores", str(self.cpu_count) if self.cpu_count else UNKNOWN),
            ("RAM", f"{self.ram_gb} GiB" if self.ram_gb else UNKNOWN),
        ]
        return [f"{label:<16}: {value}" for label, value in rows]


def probe_environment(torch_module: Optional[Any] = None) -> EnvironmentInfo:
    """Collect the current machine's metadata. Never raises."""
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ImportError:
            torch_module = None

    return EnvironmentInfo(
        os=f"{platform.system()} {platform.release()} ({platform.machine()})",
        python=sys.version.split()[0],
        torch=getattr(torch_module, "__version__", None),
        ultralytics=_module_version("ultralytics"),
        opencv=_module_version("cv2"),
        numpy=_module_version("numpy"),
        hip=hip_version(torch_module),
        cuda=cuda_version(torch_module),
        rocm_build=is_rocm_build(torch_module),
        gpu_kind=gpu_kind(torch_module),
        gpus=device_names(torch_module),
        mps=mps_available(torch_module),
        cpu=cpu_name(),
        cpu_count=cpu_count(),
        ram_gb=total_ram_gb(),
    )
