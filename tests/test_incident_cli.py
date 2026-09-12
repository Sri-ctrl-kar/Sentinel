"""The incident demo command and its environment handling (sections J, F).

The CLI is where a reviewer meets M0.7, so the boundary between what Sentinel
measured and what a model said about it has to be impossible to miss — these
tests assert the fence, not just the content.
"""

from __future__ import annotations

import json

import pytest

from app.incident import (
    AI_BANNER,
    DETERMINISTIC_BANNER,
    build_evidence,
    main,
)
from app.intelligence.settings import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_REASONER,
    ReasonerSettings,
    load_dotenv,
)


def run(argv, capsys):
    code = main(argv)
    return code, capsys.readouterr().out


# ---------------------------------------------------------------------------
# The demo command
# ---------------------------------------------------------------------------
def test_the_documented_command_runs(capsys):
    code, out = run(["--scenario", "D", "--reasoner", "mock"], capsys)
    assert code == 0
    assert DETERMINISTIC_BANNER in out
    assert AI_BANNER in out


def test_the_deterministic_half_comes_first(capsys):
    _code, out = run(["--scenario", "D", "--reasoner", "mock"], capsys)
    assert out.index(DETERMINISTIC_BANNER) < out.index(AI_BANNER)


def test_the_boundary_is_visually_obvious(capsys):
    _code, out = run(["--scenario", "D", "--reasoner", "mock"], capsys)
    assert "=" * 72 in out
    assert "source of truth" in out
    assert "it does not decide, score or override it" in out


def test_the_deterministic_half_shows_the_evidence_and_the_risk_state(capsys):
    _code, out = run(["--scenario", "D", "--reasoner", "mock"], capsys)
    head = out.split(AI_BANNER)[0]
    assert "PERSON_VEHICLE_COLLISION_RISK" in head
    assert "risk score    : 100.0/100" in head
    assert "severity critical" in head
    assert "lifecycle     : imminent" in head
    assert "current separation: 3.90 m" in head
    assert "PREDICTED_TRAJECTORY_CONFLICT" in head
    assert "time to risk: predicted" in head


def test_the_ai_half_is_labelled_with_its_provider(capsys):
    _code, out = run(["--scenario", "D", "--reasoner", "mock"], capsys)
    tail = out.split(AI_BANNER)[1]
    assert "provider: mock / deterministic-template-v0.7" in tail
    assert "deterministic template" in tail


def test_the_risk_score_is_labelled_as_not_a_probability(capsys):
    _code, out = run(["--scenario", "D", "--reasoner", "mock"], capsys)
    assert "NOT a probability" in out


def test_the_grounding_report_is_shown(capsys):
    _code, out = run(["--scenario", "D", "--reasoner", "mock"], capsys)
    assert "GROUNDING CHECK" in out
    assert "PASS" in out


def test_the_uncalibrated_run_reports_pixels(capsys):
    _code, out = run(
        ["--scenario", "D", "--reasoner", "mock", "--no-calibration"], capsys
    )
    assert "image_pixels" in out
    assert "distances in px" in out
    assert "m/s" not in out


@pytest.mark.parametrize("scenario", ["A", "B", "C", "D", "E", "F", "G", "H"])
def test_every_scenario_can_be_explained(scenario, capsys):
    code, out = run(["--scenario", scenario, "--reasoner", "mock"], capsys)
    assert code == 0
    assert AI_BANNER in out


def test_strict_mode_passes_on_grounded_output(capsys):
    code, _out = run(["--scenario", "D", "--reasoner", "mock", "--strict"], capsys)
    assert code == 0


def test_an_unknown_reasoner_fails_cleanly(capsys):
    code = main(["--scenario", "D", "--reasoner", "nope"])
    captured = capsys.readouterr()
    assert code == 2
    assert "unknown reasoner" in captured.err


def test_json_output_separates_evidence_explanation_and_grounding(capsys):
    _code, out = run(["--scenario", "D", "--reasoner", "mock", "--json"], capsys)
    payload = json.loads(out)[0]
    assert set(payload) == {
        "deterministic_evidence",
        "ai_explanation",
        "grounding",
    }
    assert payload["deterministic_evidence"]["risk_score"] == 100.0
    assert payload["ai_explanation"]["is_language_model"] is False
    assert payload["grounding"]["ok"] is True


def test_the_cli_is_deterministic(capsys):
    _c1, first = run(["--scenario", "D", "--reasoner", "mock", "--json"], capsys)
    _c2, second = run(["--scenario", "D", "--reasoner", "mock", "--json"], capsys)
    assert first == second


def test_the_limit_flag_explains_more_than_one_assessment(capsys):
    _code, out = run(
        ["--scenario", "G", "--reasoner", "mock", "--limit", "2", "--json"], capsys
    )
    assert len(json.loads(out)) == 2


def test_build_evidence_uses_the_deterministic_engine():
    evidence = build_evidence("D")[0]
    assert evidence.risk_score == 100.0
    assert evidence.incident_state == "imminent"


# ---------------------------------------------------------------------------
# Settings and credentials
# ---------------------------------------------------------------------------
def test_the_default_provider_needs_no_credentials():
    settings = ReasonerSettings.from_env(environ={})
    assert settings.provider == DEFAULT_REASONER == "mock"
    assert settings.has_credentials is False
    assert settings.model == DEFAULT_ANTHROPIC_MODEL


def test_settings_come_from_the_environment():
    settings = ReasonerSettings.from_env(
        environ={
            "SENTINEL_REASONER": "anthropic",
            "SENTINEL_REASONER_MODEL": "claude-opus-5",
            "SENTINEL_REASONER_MAX_TOKENS": "1234",
            "ANTHROPIC_API_KEY": "sk-test",
        }
    )
    assert settings.provider == "anthropic"
    assert settings.model == "claude-opus-5"
    assert settings.max_tokens == 1234
    assert settings.has_credentials is True


def test_a_malformed_numeric_setting_falls_back_to_the_default():
    settings = ReasonerSettings.from_env(
        environ={"SENTINEL_REASONER_MAX_TOKENS": "lots"}
    )
    assert settings.max_tokens == 2000


def test_settings_never_expose_the_key():
    settings = ReasonerSettings.from_env(environ={"ANTHROPIC_API_KEY": "sk-secret"})
    assert "sk-secret" not in repr(settings)
    assert "sk-secret" not in settings.describe()
    assert settings.describe().endswith("credentials=present")


def test_dotenv_does_not_override_the_process_environment(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=from-file\nSENTINEL_REASONER=anthropic\n")
    environ = {"ANTHROPIC_API_KEY": "from-shell"}
    applied = load_dotenv(str(env_file), environ=environ)
    assert environ["ANTHROPIC_API_KEY"] == "from-shell"
    assert environ["SENTINEL_REASONER"] == "anthropic"
    assert applied == {"SENTINEL_REASONER": "anthropic"}


def test_a_missing_dotenv_is_not_an_error(tmp_path):
    assert load_dotenv(str(tmp_path / "nothing-here"), environ={}) == {}


def test_the_example_env_file_documents_the_variables():
    with open(".env.example", "r", encoding="utf-8") as handle:
        text = handle.read()
    for variable in (
        "ANTHROPIC_API_KEY",
        "SENTINEL_REASONER",
        "SENTINEL_REASONER_MODEL",
    ):
        assert variable in text
