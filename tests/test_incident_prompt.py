"""Evidence-grounded prompt construction (M0.7 section D).

Each of the eight prompt rules is asserted here, because a prompt that
silently loses a rule is indistinguishable from one that keeps it until a
model gets it wrong in front of a user.
"""

from __future__ import annotations

from app.intelligence import build_prompt
from app.intelligence.prompt import SYSTEM_PROMPT, evidence_block, user_message
from app.intelligence.evidence import PredictionEvidence, TimeToRiskEvidence
from app.spatial import GROUND_PLANE_METERS, IMAGE_PIXELS
from incident_helpers import make_evidence, scenario_evidence


def test_the_prompt_has_a_system_and_a_user_half():
    prompt = build_prompt(scenario_evidence("D"))
    assert prompt.system == SYSTEM_PROMPT
    assert prompt.to_messages() == [{"role": "user", "content": prompt.user}]


# ---------------------------------------------------------------------------
# The eight rules
# ---------------------------------------------------------------------------
def test_rule_use_only_the_supplied_evidence():
    assert "USE ONLY THE SUPPLIED EVIDENCE" in SYSTEM_PROMPT


def test_rule_invent_nothing_names_what_must_not_be_invented():
    lowered = SYSTEM_PROMPT.lower()
    for forbidden in ("locations", "people", "vehicles", "measurements", "names"):
        assert forbidden in lowered


def test_rule_never_convert_units():
    assert "NEVER CONVERT UNITS" in SYSTEM_PROMPT
    assert "are NOT metres" in SYSTEM_PROMPT


def test_rule_the_risk_score_is_not_a_probability():
    assert "THE RISK SCORE IS NOT A PROBABILITY" in SYSTEM_PROMPT
    assert "ordinal" in SYSTEM_PROMPT


def test_rule_distinguish_present_from_predicted():
    assert "DISTINGUISH PRESENT FROM PREDICTED" in SYSTEM_PROMPT


def test_rule_sentinel_does_not_act():
    assert "SENTINEL DOES NOT ACT" in SYSTEM_PROMPT


def test_rule_say_so_when_the_evidence_is_thin():
    assert "SAY SO WHEN THE EVIDENCE IS THIN" in SYSTEM_PROMPT


def test_rule_explain_why_the_system_raised_this():
    assert "EXPLAIN WHY THE SYSTEM RAISED THIS" in SYSTEM_PROMPT


def test_the_prompt_states_that_the_deterministic_engine_is_authoritative():
    lowered = SYSTEM_PROMPT.lower()
    assert "authoritative" in lowered
    assert "already made every judgement" in lowered


# ---------------------------------------------------------------------------
# The evidence block
# ---------------------------------------------------------------------------
def test_the_evidence_block_contains_the_measurements_with_units():
    evidence = scenario_evidence("D")
    block = evidence_block(evidence)
    assert "current separation: 3.90 m" in block
    assert "closing speed: 5.50 m/s" in block


def test_the_evidence_block_lists_every_entity_and_claims_completeness():
    evidence = scenario_evidence("D")
    block = evidence_block(evidence)
    assert "this is the complete list" in block
    for entity_id in evidence.entity_ids:
        assert entity_id in block


def test_the_evidence_block_carries_the_exact_json():
    evidence = scenario_evidence("D")
    assert evidence.to_json() in evidence_block(evidence)


def test_the_evidence_block_states_the_factors_that_raised_the_incident():
    block = evidence_block(scenario_evidence("D"))
    for factor in ("proximity", "closing_speed", "trajectory", "zone"):
        assert factor in block


def test_the_prompt_states_the_prediction_and_its_horizon():
    block = evidence_block(scenario_evidence("D"))
    assert "PREDICTED_TRAJECTORY_CONFLICT" in block
    assert "prediction horizon: 6.0 s" in block


def test_a_predicted_crossing_is_labelled_a_prediction():
    block = evidence_block(scenario_evidence("D"))
    assert "a prediction, not an event" in block


def test_a_present_unsafe_condition_is_labelled_present_tense():
    evidence = make_evidence(
        time_to_risk=TimeToRiskEvidence(status="already_unsafe", seconds=0.0),
        prediction=PredictionEvidence(outcome="CURRENTLY_UNSAFE_PROXIMITY"),
        incident_state="current",
    )
    block = evidence_block(evidence)
    assert "crossed" in block and "NOW" in block
    assert "present condition, not a prediction" in block


# ---------------------------------------------------------------------------
# Units in the prompt
# ---------------------------------------------------------------------------
def test_pixel_evidence_gets_an_explicit_pixel_warning():
    message = user_message(scenario_evidence("D", calibrated=False))
    assert "IMAGE PIXEL space" in message
    assert "not 40 metres" in message
    assert "no camera calibration" in message.lower()


def test_calibrated_evidence_says_the_distances_are_real_metres():
    message = user_message(scenario_evidence("D", calibrated=True))
    assert "calibrated to the ground plane" in message
    assert "Do not describe them as pixels" in message


def test_the_task_block_pins_the_coordinate_space_to_echo():
    for calibrated, space in ((True, GROUND_PLANE_METERS), (False, IMAGE_PIXELS)):
        message = user_message(scenario_evidence("D", calibrated=calibrated))
        assert f"coordinate_space: exactly {space!r}" in message


def test_insufficient_history_is_called_out_in_the_task_block():
    evidence = make_evidence(
        prediction=PredictionEvidence(
            outcome="PREDICTION_UNAVAILABLE",
            unavailable_reason="insufficient_history",
        ),
        time_to_risk=None,
    )
    message = user_message(evidence)
    assert "insufficient history" in message
    assert "do not supply a prediction of your own" in message


def test_the_prompt_never_contains_a_risk_verdict_for_the_model_to_make():
    """The model is asked to explain a decision, never to make one."""
    message = user_message(scenario_evidence("D")).lower()
    assert "decide whether" not in message
    assert "you are interpreting it, not deciding it" in message
