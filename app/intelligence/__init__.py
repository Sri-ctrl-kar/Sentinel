"""Multimodal incident intelligence (M0.7).

Strictly downstream of :mod:`app.reasoning`. The deterministic pipeline has
already decided whether an incident exists, how severe it is and what is
predicted to happen; this package turns that decision into something a human
can read, and checks the result against the evidence it came from.

::

    evidence = evidence_from_assessment(assessment)      # deterministic
    reasoner = build_reasoner("mock")                    # provider-agnostic
    explanation = reasoner.reason(evidence)              # interpretation
    report = check_grounding(explanation, evidence)      # enforcement

The AI layer explains Sentinel's structured evidence; it does not replace the
deterministic risk engine.
"""

from __future__ import annotations

from .evidence import (
    EVIDENCE_SCHEMA_VERSION,
    EntityEvidence,
    FactorEvidence,
    IncidentEvidence,
    PredictionEvidence,
    Quantity,
    TimeToRiskEvidence,
    evidence_from_assessment,
)
from .grounding import (
    GroundingError,
    GroundingReport,
    GroundingViolation,
    assert_grounded,
    check_grounding,
)
from .lifecycle import (
    ACTIVE_STATES,
    INCIDENT_STATES,
    LifecycleTracker,
    derive_incident_state,
    describe_state,
)
from .prompt import Prompt, build_prompt
from .reasoner import (
    GroundedReasoner,
    IncidentReasoner,
    ReasonerError,
    ReasonerUnavailable,
    available_reasoners,
    build_reasoner,
    register_reasoner,
)
from .schema import (
    EXPLANATION_SCHEMA_VERSION,
    URGENCIES,
    ExplanationSchemaError,
    IncidentExplanation,
    explanation_json_schema,
)

__all__ = [
    "ACTIVE_STATES",
    "EVIDENCE_SCHEMA_VERSION",
    "EXPLANATION_SCHEMA_VERSION",
    "INCIDENT_STATES",
    "URGENCIES",
    "EntityEvidence",
    "ExplanationSchemaError",
    "FactorEvidence",
    "GroundedReasoner",
    "GroundingError",
    "GroundingReport",
    "GroundingViolation",
    "IncidentEvidence",
    "IncidentExplanation",
    "IncidentReasoner",
    "LifecycleTracker",
    "PredictionEvidence",
    "Prompt",
    "Quantity",
    "ReasonerError",
    "ReasonerUnavailable",
    "TimeToRiskEvidence",
    "assert_grounded",
    "available_reasoners",
    "build_prompt",
    "build_reasoner",
    "check_grounding",
    "derive_incident_state",
    "describe_state",
    "evidence_from_assessment",
    "explanation_json_schema",
    "register_reasoner",
]
