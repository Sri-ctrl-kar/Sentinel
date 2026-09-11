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
