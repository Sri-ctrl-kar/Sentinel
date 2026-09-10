"""Tests for the YOLO backend's device resolution.

These do not load a model — they check the AMD/NVIDIA device-selection logic,
which is the part most likely to break when moving between CPU, CUDA and ROCm.
"""

import sys
import types

import pytest

from app.perception.backends import yolo_ultralytics as backend


class FakeCuda:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


def fake_torch(monkeypatch, *, cuda=False, hip=None):
    module = types.ModuleType("torch")
    module.cuda = FakeCuda(cuda)
    module.version = types.SimpleNamespace(hip=hip, cuda="12.1" if cuda and not hip else None)
    module.backends = types.SimpleNamespace(mps=None)
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


def test_explicit_device_is_respected(monkeypatch):
    fake_torch(monkeypatch, cuda=True)
    assert backend.resolve_device("cpu") == "cpu"


def test_auto_falls_back_to_cpu_without_an_accelerator(monkeypatch):
    fake_torch(monkeypatch, cuda=False)
    assert backend.resolve_device("auto") == "cpu"
    assert backend.describe_accelerator() == "cpu"


def test_auto_selects_cuda_when_available(monkeypatch):
    fake_torch(monkeypatch, cuda=True)
    assert backend.resolve_device("auto") == "cuda"
    assert backend.describe_accelerator() == "cuda"


def test_rocm_is_reported_as_rocm_but_still_uses_the_cuda_device_string(monkeypatch):
    # ROCm builds of torch expose HIP through the CUDA API surface, so the
    # device string stays "cuda" while the accelerator is really AMD.
    fake_torch(monkeypatch, cuda=True, hip="6.2.0")
    assert backend.resolve_device("auto") == "cuda"
    assert backend.describe_accelerator() == "rocm"


def test_missing_torch_degrades_to_cpu(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(builtins, "__import__", blocked)
    assert backend.resolve_device("auto") == "cpu"
    assert backend.describe_accelerator() == "cpu"


def test_missing_ultralytics_gives_an_actionable_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("ultralytics"):
            raise ImportError("no ultralytics")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match="requires ultralytics"):
        backend.UltralyticsYOLODetector()
