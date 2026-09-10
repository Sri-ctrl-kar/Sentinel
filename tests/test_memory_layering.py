"""Mechanically enforce the M0.2 architectural boundary.

The spec requires perception, event generation, temporal memory and future
reasoning to stay clearly separated, and the temporal memory to contain no
model-specific code. A docstring promising that decays; this test does not.
"""

import ast
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Anything that ties code to a specific model, framework or media library.
MODEL_SPECIFIC = ("torch", "ultralytics", "cv2", "numpy", "onnxruntime", "tensorflow")


def module_files(package: str):
    directory = os.path.join(ROOT, "app", *package.split("."))
    for entry in sorted(os.listdir(directory)):
        if entry.endswith(".py"):
            yield os.path.join(directory, entry)


def imported_names(path: str):
    """Every module name imported by a file, absolute and relative."""
    with open(path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)

    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # Relative imports: "." -> app, ".." -> the parent package.
            prefix = "." * node.level
            names.append(f"{prefix}{node.module or ''}")
    return names


# ---------------------------------------------------------------------------
# The temporal memory must not know about any model
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", list(module_files("memory")))
def test_memory_has_no_model_specific_imports(path):
    for name in imported_names(path):
        root = name.lstrip(".").split(".")[0]
        assert root not in MODEL_SPECIFIC, (
            f"{os.path.basename(path)} imports '{name}'. The temporal memory must "
            f"stay model-agnostic — see app/memory/__init__.py."
        )


@pytest.mark.parametrize("path", list(module_files("memory")))
def test_memory_does_not_import_the_perception_layer(path):
    for name in imported_names(path):
        assert "perception" not in name, (
            f"{os.path.basename(path)} imports '{name}'. Temporal memory consumes "
            f"events, never tracks or detections."
        )


def test_importing_memory_does_not_drag_in_a_model_runtime():
    """Importing app.memory in a clean interpreter must not load torch or cv2."""
    code = (
        "import sys; import app.memory; "
        f"loaded=[m for m in {MODEL_SPECIFIC!r} if m in sys.modules]; "
        "print(','.join(loaded))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", (
        f"importing app.memory pulled in {result.stdout.strip()}"
    )


# ---------------------------------------------------------------------------
# The other boundaries
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", list(module_files("events")))
def test_event_layer_has_no_model_specific_imports(path):
    for name in imported_names(path):
        root = name.lstrip(".").split(".")[0]
        assert root not in MODEL_SPECIFIC, (
            f"{os.path.basename(path)} imports '{name}'. Event generation reads "
            f"Track objects, never model output."
        )


def test_spatial_module_is_dependency_free():
    path = os.path.join(ROOT, "app", "spatial.py")
    for name in imported_names(path):
        root = name.lstrip(".").split(".")[0]
        assert root not in MODEL_SPECIFIC
        assert "perception" not in name


def test_only_the_yolo_backend_imports_torch():
    """Model libraries stay confined to app/perception/backends/."""
    offenders = []
    for dirpath, _dirnames, filenames in os.walk(os.path.join(ROOT, "app")):
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            roots = {n.lstrip(".").split(".")[0] for n in imported_names(path)}
            if roots & {"torch", "ultralytics"}:
                offenders.append(os.path.relpath(path, ROOT))
    assert offenders == ["app/perception/backends/yolo_ultralytics.py"]


def test_reasoning_layer_is_untouched_by_m0_2():
    """M0.2 must not start building the reasoning layer."""
    for path in module_files("reasoning"):
        for name in imported_names(path):
            root = name.lstrip(".").split(".")[0]
            assert root not in MODEL_SPECIFIC
