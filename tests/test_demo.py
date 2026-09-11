"""The synthetic warehouse demo command."""

import json

import pytest

from app.demo import (
    build_warehouse_memory,
    main,
    run_demo,
    warehouse_config,
)


def test_demo_scene_produces_two_entities():
    memory = build_warehouse_memory()
    assert sorted(memory.entities()) == ["forklift_2", "person_1"]


def test_demo_scene_records_the_zone_entry():
    memory = build_warehouse_memory()
    entries = memory.stream(actions=["entered_zone"], entity_id="person_1")
    assert entries
    assert entries[0].attributes["zone"] == "forklift_bay"


def test_demo_reaches_a_critical_assessment():
    report = run_demo()
    assert report.top is not None
    assert report.top.severity in ("high", "critical")
    assert report.top.incident_type == "PERSON_VEHICLE_COLLISION_RISK"


def test_demo_defaults_to_the_first_alertable_moment():
    """Not the final frame: the useful question is when Sentinel would warn."""
    memory = build_warehouse_memory()
    _first, last = memory.span
    report = run_demo()
    assert report.timestamp < last


def test_demo_predicts_a_useful_amount_of_warning():
    report = run_demo()
    tti = report.top.predicted_time_to_incident_seconds
    assert tti is not None
    assert tti > 0.3, "an alert with no lead time is not a prediction"


def test_demo_is_deterministic():
    assert run_demo().to_dict() == run_demo().to_dict()


def test_demo_config_marks_the_bay_as_an_operating_zone():
    config = warehouse_config()
    assert config.operating_zones == ["forklift_bay"]
    assert config.zones.names == ["forklift_bay"]


def test_demo_cli_prints_an_incident_analysis(capsys):
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "SENTINEL INCIDENT ANALYSIS" in output
    assert "Severity:" in output
    assert "person_1" in output
    assert "forklift_2" in output
    assert "Recommended intervention:" in output
    assert "image_pixels" in output


def test_demo_cli_json_mode(capsys):
    assert main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["coordinate_space"] == "image_pixels"
    assert payload["assessments"]


def test_demo_cli_timeline_shows_risk_building(capsys):
    assert main(["--timeline"]) == 0
    output = capsys.readouterr().out
    assert "RISK DEVELOPMENT OVER TIME" in output
    assert "critical" in output


def test_demo_cli_accepts_an_explicit_timestamp(capsys):
    assert main(["--at", "0.5"]) == 0
    assert "t=0.50s" in capsys.readouterr().out


def test_demo_cli_all_shows_every_assessment(capsys):
    main(["--all"])
    everything = capsys.readouterr().out
    main([])
    just_top = capsys.readouterr().out
    assert len(everything) >= len(just_top)


# ---------------------------------------------------------------------------
# M0.4: calibrated demo mode
# ---------------------------------------------------------------------------
def test_calibration_report_maps_both_entities_to_the_ground_plane():
    from app.demo import calibration_report

    text = "\n".join(calibration_report())
    assert "GROUND-PLANE CALIBRATION" in text
    assert "person_1" in text and "forklift_2" in text
    assert "ground_plane_meters" in text
    assert "separation" in text and "closing speed" in text


def test_calibration_report_declares_itself_synthetic():
    from app.demo import calibration_report

    text = "\n".join(calibration_report())
    assert "SYNTHETIC EXAMPLE" in text
    assert "no accuracy claim" in text
    assert "Calibration limitations" in text


def test_calibration_report_is_deterministic():
    from app.demo import calibration_report

    assert calibration_report() == calibration_report()


def test_demo_cli_calibrated_mode(capsys):
    assert main(["--calibrated"]) == 0
    output = capsys.readouterr().out
    assert "SENTINEL INCIDENT ANALYSIS" in output
    assert "GROUND-PLANE CALIBRATION" in output
    assert "m/s" in output


def test_demo_without_calibration_shows_no_metric_block(capsys):
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "GROUND-PLANE CALIBRATION" not in output
    assert "image_pixels" in output


def test_calibrated_demo_does_not_change_the_risk_score():
    from app.calibration.examples import warehouse_calibration

    plain = run_demo()
    calibrated = run_demo(calibration=warehouse_calibration())
    assert plain.max_score == calibrated.max_score
    assert plain.timestamp == calibrated.timestamp


# ---------------------------------------------------------------------------
# M0.5: world-space prediction demo
# ---------------------------------------------------------------------------
def test_world_prediction_report_shows_the_escalation():
    from app.demo import world_prediction_report

    lines = world_prediction_report()
    text = "\n".join(lines)
    assert "WORLD-SPACE PREDICTIVE RISK" in text
    assert "ground_plane_meters" in text
    assert "PREDICTED_TRAJECTORY_CONFLICT" in text
    assert "critical" in text
    assert "NOT a probability" in text


def test_world_prediction_report_shows_risk_rising_over_time():
    """current state -> predicted conflict -> time-to-risk -> escalation."""
    from app.demo import world_prediction_report

    rows = [l for l in world_prediction_report() if l.startswith("   0.") or l.startswith("   1.")]
    scores = []
    for row in rows:
        parts = row.split()
        if len(parts) >= 6:
            try:
                scores.append(float(parts[5]))
            except ValueError:
                continue
    assert scores
    assert max(scores) > min(scores), "risk should escalate across the scene"


def test_world_prediction_report_is_deterministic():
    from app.demo import world_prediction_report

    assert world_prediction_report() == world_prediction_report()


def test_demo_cli_world_mode(capsys):
    assert main(["--world"]) == 0
    output = capsys.readouterr().out
    assert "WORLD-SPACE PREDICTIVE RISK" in output


def test_demo_cli_evaluate_mode(capsys):
    assert main(["--evaluate"]) == 0
    output = capsys.readouterr().out
    assert "SENTINEL PREDICTIVE EVALUATION" in output
    assert "prediction lead time" in output


def test_evaluation_module_cli(capsys):
    from app.evaluation import main as evaluate_main

    assert evaluate_main([]) == 0
    assert "SENTINEL PREDICTIVE EVALUATION" in capsys.readouterr().out


def test_evaluation_module_cli_json(capsys):
    import json

    from app.evaluation import main as evaluate_main

    assert evaluate_main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scenario_count"] == 8
    assert "metric_caveat" in payload
