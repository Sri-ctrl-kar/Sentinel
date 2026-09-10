"""Risk engine behaviour: aggregation, attribution, determinism and output."""

import json

import pytest

from app.events.schema import Event
from app.memory import TemporalEventMemory
from app.reasoning import Explainer, RiskConfig, RiskEngine
from app.reasoning.models.risk import (
    SEVERITIES,
    FactorScore,
    RiskAssessment,
    RiskReport,
    severity_for,
)
from app.spatial import Zone, ZoneSet

from scenarios import load, warehouse_config


# ---------------------------------------------------------------------------
# Severity bands
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "score,expected",
    [
        (0, "normal"), (19.9, "normal"),
        (20, "low"), (39.9, "low"),
        (40, "medium"), (64.9, "medium"),
        (65, "high"), (89.9, "high"),
        (90, "critical"), (100, "critical"),
    ],
)
def test_severity_thresholds(score, expected):
    assert severity_for(score) == expected


def test_severity_is_monotonic():
    previous = -1
    for score in range(0, 101):
        rank = SEVERITIES.index(severity_for(score))
        assert rank >= previous
        previous = rank


# ---------------------------------------------------------------------------
# Score aggregation
# ---------------------------------------------------------------------------
def test_risk_score_equals_subtotal_times_multiplier():
    """The published formula must actually be the formula used."""
    scenario = load("C")
    scenario.config.report_threshold = 0.0
    assessment = RiskEngine(scenario.config).assess(scenario.memory, at=scenario.at).top

    subtotal = sum(
        f.contribution for f in assessment.contributing_factors if f.name != "escalation"
    )
    expected = min(100.0, subtotal * assessment.escalation_multiplier)
    assert assessment.risk_score == pytest.approx(expected, abs=0.01)
    assert assessment.details["subtotal_points"] == pytest.approx(subtotal, abs=0.01)


def test_each_factor_contribution_is_score_times_weight():
    scenario = load("D")
    assessment = RiskEngine(scenario.config).assess(scenario.memory, at=scenario.at).top
    for factor in assessment.contributing_factors:
        assert factor.contribution == pytest.approx(factor.score * factor.weight, abs=1e-4)


def test_base_factor_weights_sum_to_one_hundred():
    weights = RiskConfig().weights
    assert weights.base_total == 100.0


def test_risk_score_is_clamped_to_one_hundred():
    scenario = load("D")
    assessment = RiskEngine(scenario.config).assess(scenario.memory, at=scenario.at).top
    assert assessment.risk_score == 100.0
    # ...even though the raw arithmetic exceeded it.
    assert assessment.details["subtotal_points"] * assessment.escalation_multiplier > 100


def test_escalation_cannot_manufacture_risk_from_nothing():
    """0 x 1.35 is still 0: corroboration amplifies, it does not invent."""
    scenario = load("A")
    scenario.config.report_threshold = 0.0
    assessment = RiskEngine(scenario.config).assess(scenario.memory, at=scenario.at).top
    assert assessment.risk_score == 0.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_repeated_assessment_is_byte_identical():
    scenario = load("D")
    engine = RiskEngine(scenario.config)
    first = engine.assess(scenario.memory, at=scenario.at).to_dict()
    second = engine.assess(scenario.memory, at=scenario.at).to_dict()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_two_engine_instances_agree():
    scenario = load("C")
    a = RiskEngine(warehouse_config()).assess(scenario.memory, at=scenario.at)
    b = RiskEngine(warehouse_config()).assess(scenario.memory, at=scenario.at)
    assert a.to_dict() == b.to_dict()


def test_assessment_does_not_depend_on_wall_clock_or_ordering():
    """Re-ingesting the same events in reverse must not change the verdict."""
    scenario = load("C")
    shuffled = TemporalEventMemory.from_events(reversed(scenario.memory.events))
    original = RiskEngine(warehouse_config()).assess(scenario.memory, at=scenario.at)
    rebuilt = RiskEngine(warehouse_config()).assess(shuffled, at=scenario.at)
    assert rebuilt.max_score == pytest.approx(original.max_score)


# ---------------------------------------------------------------------------
# Entity and evidence attribution
# ---------------------------------------------------------------------------
def test_assessment_names_the_entities_involved():
    scenario = load("D")
    assessment = RiskEngine(scenario.config).assess(scenario.memory, at=scenario.at).top
    assert set(assessment.involved_entity_ids) == {"person_1", "forklift_2"}


def test_evidence_event_ids_exist_in_the_memory():
    """Evidence must be checkable — a dangling reference is not evidence."""
    scenario = load("D")
    assessment = RiskEngine(scenario.config).assess(scenario.memory, at=scenario.at).top
    known = {e.event_id for e in scenario.memory}
    assert assessment.evidence_event_ids
    for event_id in assessment.evidence_event_ids:
        assert event_id in known


def test_evidence_never_cites_the_future():
    scenario = load("D")
    at = 1.0
    assessment = RiskEngine(scenario.config).assess(scenario.memory, at=at).top
    by_id = {e.event_id: e for e in scenario.memory}
    for event_id in assessment.evidence_event_ids:
        assert by_id[event_id].timestamp <= at


def test_evidence_is_deduplicated():
    scenario = load("D")
    assessment = RiskEngine(scenario.config).assess(scenario.memory, at=scenario.at).top
    assert len(assessment.evidence_event_ids) == len(set(assessment.evidence_event_ids))


def test_report_can_be_filtered_by_entity():
    scenario = load("D")
    report = RiskEngine(scenario.config).assess(scenario.memory, at=scenario.at)
    assert report.for_entity("person_1")
    assert report.for_entity("nobody_99") == []


# ---------------------------------------------------------------------------
# Multiple simultaneous entities
# ---------------------------------------------------------------------------
def build_crowd_memory():
    """Two people and two forklifts, all observed simultaneously."""
    events = []
    actors = [
        ("person_1", "person", 100, 12, 300),
        ("person_2", "person", 700, -12, 100),
        ("forklift_3", "forklift", 500, -10, 300),
        ("forklift_4", "forklift", 200, 10, 100),
    ]
    for entity_id, class_name, start, step, y in actors:
        for i in range(12):
            x = start + step * i
            events.append(
                Event(
                    timestamp=round(i / 10.0, 4),
                    entity_id=entity_id,
                    action="appeared" if i == 0 else "moved",
                    attributes={"class_name": class_name, "confidence": 0.9},
                    position=[x, y],
                    bbox=[x - 20, y - 45, x + 20, y + 45],
                    event_id=f"evt_{entity_id}_{i:02d}",
                )
            )
    return TemporalEventMemory.from_events(events)


def test_every_person_vehicle_pair_is_evaluated():
    memory = build_crowd_memory()
    config = RiskConfig(vehicle_classes=["forklift"], report_threshold=0.0)
    report = RiskEngine(config).assess(memory, at=1.1)

    pairs = {
        tuple(sorted(a.involved_entity_ids))
        for a in report.assessments
        if len(a.involved_entity_ids) == 2
    }
    assert pairs == {
        ("forklift_3", "person_1"),
        ("forklift_4", "person_1"),
        ("forklift_3", "person_2"),
        ("forklift_4", "person_2"),
    }


def test_assessments_are_sorted_by_risk_descending():
    memory = build_crowd_memory()
    config = RiskConfig(vehicle_classes=["forklift"], report_threshold=0.0)
    report = RiskEngine(config).assess(memory, at=1.1)
    scores = [a.risk_score for a in report.assessments]
    assert scores == sorted(scores, reverse=True)
    assert report.top.risk_score == max(scores)


def test_each_pair_is_scored_on_its_own_geometry():
    """A dangerous pair must not contaminate an unrelated one."""
    memory = build_crowd_memory()
    config = RiskConfig(vehicle_classes=["forklift"], report_threshold=0.0)
    report = RiskEngine(config).assess(memory, at=1.1)
    scores = {
        tuple(sorted(a.involved_entity_ids)): a.risk_score for a in report.assessments
        if len(a.involved_entity_ids) == 2
    }
    # person_1 and forklift_3 converge head-on; person_1 and forklift_4 do not.
    assert scores[("forklift_3", "person_1")] > scores[("forklift_4", "person_1")]


def test_vehicle_vehicle_pairs_are_not_assessed():
    memory = build_crowd_memory()
    config = RiskConfig(vehicle_classes=["forklift"], report_threshold=0.0)
    report = RiskEngine(config).assess(memory, at=1.1)
    for assessment in report.assessments:
        classes = {
            memory.latest_state(eid).class_name for eid in assessment.involved_entity_ids
        }
        assert "person" in classes


# ---------------------------------------------------------------------------
# Reporting behaviour
# ---------------------------------------------------------------------------
def test_empty_memory_produces_an_empty_report():
    report = RiskEngine().assess(TemporalEventMemory())
    assert len(report) == 0
    assert report.max_score == 0.0
    assert report.severity == "normal"
    assert report.top is None


def test_report_threshold_suppresses_quiet_situations():
    scenario = load("F")
    loud = RiskEngine(warehouse_config(report_threshold=0.0)).assess(
        scenario.memory, at=scenario.at
    )
    quiet = RiskEngine(warehouse_config(report_threshold=50.0)).assess(
        scenario.memory, at=scenario.at
    )
    assert len(loud) > len(quiet)


def test_assess_defaults_to_the_last_moment_in_memory():
    scenario = load("D")
    report = RiskEngine(scenario.config).assess(scenario.memory)
    _first, last = scenario.memory.span
    assert report.timestamp == last


def test_assess_timeline_walks_the_whole_clip():
    scenario = load("D")
    reports = RiskEngine(scenario.config).assess_timeline(scenario.memory, step=0.5)
    assert len(reports) > 1
    assert [r.timestamp for r in reports] == sorted(r.timestamp for r in reports)


def test_assess_timeline_shows_risk_developing():
    """The engine must see the situation build, not just its end state."""
    scenario = load("D")
    reports = RiskEngine(scenario.config).assess_timeline(scenario.memory, step=0.2)
    scores = [r.max_score for r in reports]
    assert scores[0] < scores[-1]
    assert max(scores) >= 90.0


def test_assess_timeline_rejects_a_non_positive_step():
    scenario = load("D")
    with pytest.raises(ValueError, match="step must be"):
        RiskEngine(scenario.config).assess_timeline(scenario.memory, step=0)


def test_score_accepts_a_plain_event_list():
    scenario = load("C")
    report = RiskEngine(warehouse_config()).score(scenario.memory.events)
    assert isinstance(report, RiskReport)


def test_at_or_above_filters_by_severity():
    scenario = load("D")
    report = RiskEngine(scenario.config).assess(scenario.memory, at=scenario.at)
    assert report.at_or_above("critical")
    assert len(report.at_or_above("normal")) >= len(report.at_or_above("critical"))


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------
def test_confidence_is_zero_when_nothing_contributes():
    scenario = load("A")
    scenario.config.report_threshold = 0.0
    assessment = RiskEngine(scenario.config).assess(scenario.memory, at=scenario.at).top
    assert assessment.confidence == 0.0


def test_confidence_reflects_the_evidence_behind_the_score():
    early = RiskEngine(warehouse_config()).assess(load("D").memory, at=0.4)
    late = RiskEngine(warehouse_config()).assess(load("D").memory, at=1.5)
    assert early.top.confidence <= late.top.confidence


# ---------------------------------------------------------------------------
# Serialisation and configuration validation
# ---------------------------------------------------------------------------
def test_report_serialises_to_json():
    scenario = load("D")
    report = RiskEngine(scenario.config).assess(scenario.memory, at=scenario.at)
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["coordinate_space"] == "image_pixels"
    assert payload["assessments"][0]["risk_score"] > 0
    assert payload["assessments"][0]["contributing_factors"]


def test_assessment_exposes_factors_by_name():
    scenario = load("D")
    assessment = RiskEngine(scenario.config).assess(scenario.memory, at=scenario.at).top
    assert assessment.factor("proximity") is not None
    assert assessment.factor("no_such_factor") is None
    assert "trajectory" in assessment.factor_names


def test_config_rejects_a_non_pixel_coordinate_space():
    with pytest.raises(ValueError, match="only 'image_pixels'"):
        RiskConfig(coordinate_space="world_metres")


def test_config_rejects_inverted_radii():
    with pytest.raises(ValueError, match="critical_radius_px must be"):
        RiskConfig(critical_radius_px=500.0, interaction_radius_px=100.0)
    with pytest.raises(ValueError, match="conflict_radius_px must be"):
        RiskConfig(conflict_radius_px=900.0, trajectory_miss_radius_px=100.0)


def test_config_is_serialisable_and_states_its_units():
    payload = RiskConfig().to_dict()
    assert payload["coordinate_space"] == "image_pixels"
    assert "interaction_radius_px" in payload
    assert "closing_speed_reference_px_per_s" in payload


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------
def test_explainer_renders_the_expected_sections():
    scenario = load("D")
    assessment = RiskEngine(scenario.config).assess(scenario.memory, at=scenario.at).top
    text = Explainer().explain(assessment)

    assert "SENTINEL INCIDENT ANALYSIS" in text
    assert "Risk: 100/100" in text
    assert "Severity: CRITICAL" in text
    assert "Incident: PERSON_VEHICLE_COLLISION_RISK" in text
    assert "person_1" in text and "forklift_2" in text
    assert "Evidence:" in text
    assert "Recommended intervention:" in text
    assert "Predicted time-to-risk:" in text


def test_explainer_states_the_coordinate_space():
    scenario = load("D")
    assessment = RiskEngine(scenario.config).assess(scenario.memory, at=scenario.at).top
    text = Explainer().explain(assessment)
    assert "image_pixels" in text
    assert "IMAGE PIXELS" in text
    assert "not physical distances" in text


def test_explainer_shows_the_arithmetic():
    """The score must be checkable by hand from the printed breakdown."""
    scenario = load("C")
    assessment = RiskEngine(scenario.config).assess(scenario.memory, at=scenario.at).top
    text = Explainer().explain(assessment)
    assert "Scoring breakdown" in text
    assert "subtotal" in text
    assert "RISK SCORE" in text


def test_explainer_says_when_it_cannot_predict():
    scenario = load("G")
    scenario.config.report_threshold = 0.0
    assessment = RiskEngine(scenario.config).assess(scenario.memory, at=scenario.at).top
    text = Explainer().explain(assessment)
    assert "not predictable" in text


def test_explainer_handles_an_empty_report():
    text = Explainer().explain_report(RiskEngine().assess(TemporalEventMemory()))
    assert "No developing situations detected" in text
