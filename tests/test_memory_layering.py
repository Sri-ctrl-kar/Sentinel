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


def module_files(package: str, recursive: bool = False):
    directory = os.path.join(ROOT, "app", *package.split("."))
    if not recursive:
        for entry in sorted(os.listdir(directory)):
            if entry.endswith(".py"):
                yield os.path.join(directory, entry)
        return
    for dirpath, _dirnames, filenames in os.walk(directory):
        for filename in sorted(filenames):
            if filename.endswith(".py"):
                yield os.path.join(dirpath, filename)


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


# ---------------------------------------------------------------------------
# The calibration layer sits between memory and reasoning (M0.4)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", list(module_files("calibration", recursive=True)))
def test_calibration_layer_has_no_model_specific_imports(path):
    for name in imported_names(path):
        root = name.lstrip(".").split(".")[0]
        assert root not in MODEL_SPECIFIC, (
            f"{os.path.relpath(path, ROOT)} imports '{name}'. Calibration is "
            f"consumed by the reasoning layer, which must load no model runtime "
            f"— the homography solver is pure Python for this reason."
        )


@pytest.mark.parametrize("path", list(module_files("calibration", recursive=True)))
def test_calibration_layer_does_not_import_perception_or_memory(path):
    for name in imported_names(path):
        assert "perception" not in name, (
            f"{os.path.relpath(path, ROOT)} imports '{name}'. Calibration "
            f"converts coordinates; it must not know what produced them."
        )
        assert "memory" not in name, (
            f"{os.path.relpath(path, ROOT)} imports '{name}'. Calibration is a "
            f"transform, not a store."
        )


def test_calibration_logic_stays_out_of_perception_and_memory():
    """The transform must not leak into the layers that feed it."""
    for package in ("perception", "memory", "events"):
        for path in module_files(package, recursive=True):
            for name in imported_names(path):
                assert "calibration" not in name, (
                    f"{os.path.relpath(path, ROOT)} imports '{name}'. "
                    f"Calibration belongs between memory and reasoning."
                )


def test_importing_calibration_does_not_drag_in_a_model_runtime():
    code = (
        "import sys; import app.calibration; "
        f"loaded=[m for m in {MODEL_SPECIFIC!r} if m in sys.modules]; "
        "print(','.join(loaded))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


# ---------------------------------------------------------------------------
# The evaluation layer consumes the others; it must not be consumed by them
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", list(module_files("evaluation", recursive=True)))
def test_evaluation_metric_code_has_no_model_specific_imports(path):
    """Metric code is pure Python; only the clip runner touches media."""
    for name in imported_names(path):
        root = name.lstrip(".").split(".")[0]
        assert root not in MODEL_SPECIFIC, (
            f"{os.path.relpath(path, ROOT)} imports '{name}' at module level. "
            f"Evaluation metrics must stay importable without a model runtime; "
            f"the pipeline is imported lazily inside run_clip()."
        )


def test_no_layer_imports_the_evaluation_package():
    """Evaluation grades the other layers; nothing may depend on it."""
    offenders = []
    for package in ("perception", "memory", "events", "calibration", "reasoning"):
        for path in module_files(package, recursive=True):
            for name in imported_names(path):
                if "evaluation" in name:
                    offenders.append(os.path.relpath(path, ROOT))
    assert offenders == [], (
        f"{offenders} import the evaluation package. Evaluation is a consumer "
        f"of these layers, never a dependency of them."
    )


def test_evaluation_logic_stays_out_of_the_risk_engine():
    """Benchmark vocabulary must not leak into the thing being benchmarked."""
    forbidden = ("lead_time", "true_positive", "false_positive", "idf1", "mota")
    for path in module_files("reasoning", recursive=True):
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read().lower()
        for term in forbidden:
            assert term not in source, (
                f"{os.path.relpath(path, ROOT)} mentions '{term}'. Evaluation "
                f"logic belongs in app/evaluation/, not in the risk engine."
            )


def test_importing_evaluation_metrics_does_not_drag_in_a_model_runtime():
    code = (
        "import sys; import app.evaluation; "
        f"loaded=[m for m in {MODEL_SPECIFIC!r} if m in sys.modules]; "
        "print(','.join(loaded))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", (
        f"importing app.evaluation pulled in {result.stdout.strip()}"
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


# ---------------------------------------------------------------------------
# The reasoning layer must not know about any model or the perception layer
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", list(module_files("reasoning", recursive=True)))
def test_reasoning_layer_has_no_model_specific_imports(path):
    for name in imported_names(path):
        root = name.lstrip(".").split(".")[0]
        assert root not in MODEL_SPECIFIC, (
            f"{os.path.relpath(path, ROOT)} imports '{name}'. The risk engine must "
            f"never know about YOLO, torch, Ultralytics or OpenCV — it reasons "
            f"about Event objects."
        )


@pytest.mark.parametrize("path", list(module_files("reasoning", recursive=True)))
def test_reasoning_layer_does_not_import_the_perception_layer(path):
    for name in imported_names(path):
        assert "perception" not in name, (
            f"{os.path.relpath(path, ROOT)} imports '{name}'. The risk engine "
            f"consumes events and temporal memory, never tracks or detections."
        )


def test_importing_the_risk_engine_does_not_drag_in_a_model_runtime():
    """A clean interpreter must be able to reason without torch or OpenCV."""
    code = (
        "import sys; import app.reasoning; "
        f"loaded=[m for m in {MODEL_SPECIFIC!r} if m in sys.modules]; "
        "print(','.join(loaded))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", (
        f"importing app.reasoning pulled in {result.stdout.strip()}"
    )


def test_reasoning_reuses_the_shared_geometry_module():
    """Geometry lives in app/spatial.py; the engine must not re-implement it."""
    engine_sources = list(module_files("reasoning", recursive=True))
    assert any(
        any("spatial" in name for name in imported_names(path))
        for path in engine_sources
    ), "the risk engine should reuse app.spatial rather than duplicating geometry"


# ---------------------------------------------------------------------------
# The AI layer sits downstream of everything (M0.7)
# ---------------------------------------------------------------------------
#: Packages that talk to a language model. The deterministic pipeline must not
#: import any of them, anywhere, at any depth.
LLM_SDKS = (
    "anthropic",
    "openai",
    "google",
    "cohere",
    "mistralai",
    "litellm",
    "langchain",
    "transformers",
    "vllm",
    "ollama",
    "llama_cpp",
    "huggingface_hub",
)

#: The single file allowed to import a model SDK.
ONLY_SDK_FILE = "app/intelligence/providers/anthropic_claude.py"


def test_no_deterministic_layer_imports_a_model_sdk():
    """Detection through risk scoring must never touch an LLM SDK."""
    offenders = []
    for package in (
        "perception",
        "memory",
        "events",
        "calibration",
        "reasoning",
        "evaluation",
        "storage",
    ):
        for path in module_files(package, recursive=True):
            for name in imported_names(path):
                root = name.lstrip(".").split(".")[0]
                if root in LLM_SDKS:
                    offenders.append((os.path.relpath(path, ROOT), name))
    assert offenders == [], (
        f"{offenders} import a language-model SDK. The deterministic pipeline "
        f"decides whether an incident exists; the AI layer only describes it."
    )


def test_no_deterministic_layer_imports_the_intelligence_package():
    """The AI layer is a consumer of the pipeline, never a dependency of it."""
    offenders = []
    for package in (
        "perception",
        "memory",
        "events",
        "calibration",
        "reasoning",
        "evaluation",
        "storage",
    ):
        for path in module_files(package, recursive=True):
            for name in imported_names(path):
                if "intelligence" in name:
                    offenders.append(os.path.relpath(path, ROOT))
    assert offenders == [], (
        f"{offenders} import app.intelligence. Risk scores must be identical "
        f"whether or not a reasoner is ever constructed."
    )


def test_only_the_anthropic_provider_imports_a_model_sdk():
    """Model SDKs stay confined to one provider file."""
    offenders = []
    for dirpath, _dirnames, filenames in os.walk(os.path.join(ROOT, "app")):
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            roots = {n.lstrip(".").split(".")[0] for n in imported_names(path)}
            if roots & set(LLM_SDKS):
                offenders.append(os.path.relpath(path, ROOT))
    assert offenders == [ONLY_SDK_FILE], (
        f"expected only {ONLY_SDK_FILE} to import a model SDK, found {offenders}"
    )


@pytest.mark.parametrize("path", list(module_files("intelligence")))
def test_intelligence_core_does_not_import_a_provider(path):
    """Evidence, schema, prompt and grounding stay provider-agnostic."""
    for name in imported_names(path):
        assert "providers" not in name, (
            f"{os.path.relpath(path, ROOT)} imports '{name}'. Providers are "
            f"resolved by name at build time so the core never depends on one."
        )


@pytest.mark.parametrize("path", list(module_files("intelligence", recursive=True)))
def test_intelligence_layer_does_not_import_perception(path):
    for name in imported_names(path):
        assert "perception" not in name, (
            f"{os.path.relpath(path, ROOT)} imports '{name}'. The AI layer reads "
            f"incident evidence, never detections."
        )


def test_importing_the_intelligence_layer_loads_no_model_runtime_or_sdk():
    """A clean interpreter must reach the AI boundary with nothing loaded."""
    watched = MODEL_SPECIFIC + LLM_SDKS
    code = (
        "import sys; import app.intelligence; "
        f"loaded=[m for m in {watched!r} if m in sys.modules]; "
        "print(','.join(loaded))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", (
        f"importing app.intelligence pulled in {result.stdout.strip()}"
    )


def test_building_the_mock_reasoner_loads_no_model_sdk():
    """The whole demo path must work with no vendor package installed."""
    code = (
        "import sys; from app.intelligence import build_reasoner; "
        "build_reasoner('mock'); "
        f"loaded=[m for m in {LLM_SDKS!r} if m in sys.modules]; "
        "print(','.join(loaded))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""
