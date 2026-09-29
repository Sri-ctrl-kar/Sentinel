"""The incident route: the M0.7 intelligence chain, over a video analysis.

This is the only place in the API that touches ``app.intelligence``, and it
does so by calling the existing functions in the existing order:

    RiskAssessment            (app.reasoning.risk_engine.RiskEngine)
      -> IncidentEvidence     (app.intelligence.evidence.evidence_from_assessment)
      -> IncidentExplanation  (app.intelligence.reasoner.build_reasoner(...).reason)
      -> GroundingReport      (app.intelligence.grounding.check_grounding)

There is no second intelligence implementation here — no prompt, no schema, no
grounding rule. The route picks the worst assessment in the clip, hands it to
the validated chain, and serialises what comes back.

The default reasoner is ``mock``: deterministic, no credentials, no network.
"""

from __future__ import annotations

from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from ...intelligence import (
    check_grounding,
    describe_state,
    evidence_from_assessment,
)
from ...intelligence.lifecycle import LifecycleTracker, situation_key
from ...intelligence.reasoner import ReasonerError, ReasonerUnavailable, build_reasoner
from ...intelligence.schema import ExplanationSchemaError
from ..schemas import IncidentResponse
from .analysis import require_complete, require_record, risk_reports_for, worst_report

router = APIRouter(prefix="/api", tags=["incident"])


def lifecycle_history(reports: List[Any], assessment: Any) -> List[dict]:
    """Every lifecycle state *this* situation passed through, in order.

    ``derive_incident_state`` sees a single assessment, and the assessment this
    route picks is the highest-scoring one — which is almost always ``current``,
    because the score peaks once the situation is already unsafe. The states the
    situation passed through on the way there, ``developing`` and ``imminent``,
    are present in the risk timeline and were simply never read.

    The walk is scoped to one situation by :func:`situation_key` (the entity
    pair). Blending unrelated pairs into a single trail would report a
    progression that never happened. :class:`LifecycleTracker` supplies the
    ``previous`` state that ``resolved`` needs and that a single-assessment
    derivation can never have.

    No state is derived here: this calls the M0.7 tracker and nothing else.
    """
    key = situation_key(assessment)
    tracker = LifecycleTracker()
    history: List[dict] = []
    for report in reports:
        for candidate in report.assessments:
            if situation_key(candidate) != key:
                continue
            history.append(
                {
                    "timestamp": round(candidate.timestamp, 4),
                    "state": tracker.update(candidate),
                    "risk_score": round(candidate.risk_score, 2),
                }
            )
            break
    return history


@router.get("/analyze/{analysis_id}/incident", response_model=IncidentResponse)
def get_incident(
    request: Request,
    analysis_id: str,
    reasoner: str = Query("mock", description="mock | anthropic"),
    explain: bool = Query(True, description="Also generate an AI explanation"),
) -> IncidentResponse:
    """Evidence for the clip's worst moment, optionally explained and checked.

    ``incident_found: false`` with a ``note`` is a legitimate answer: plenty of
    footage contains no assessable situation, and inventing one would be worse
    than saying so.
    """
    record = require_complete(require_record(request, analysis_id))
    reports = risk_reports_for(record)
    worst = worst_report(reports)

    if worst is None or worst.top is None:
        return IncidentResponse(
            analysis_id=analysis_id,
            incident_found=False,
            note=(
                "the risk engine produced no assessment for this clip; no "
                "incident evidence exists to explain"
            ),
        )

    assessment = worst.top
    temporal = record.temporal
    evidence = evidence_from_assessment(
        assessment,
        events=temporal.events if temporal is not None else (),
    )

    base = dict(
        analysis_id=analysis_id,
        incident_found=True,
        lifecycle_state=evidence.incident_state,
        lifecycle_history=lifecycle_history(reports, assessment),
        quantities=[q.to_dict() for q in evidence.quantities()],
        evidence=evidence.to_dict(),
        note=describe_state(evidence.incident_state),
    )
    if not explain:
        return IncidentResponse(**base)

    try:
        engine = build_reasoner(reasoner)
    except ReasonerUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ReasonerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        explanation = engine.reason(evidence)
    except ReasonerUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ExplanationSchemaError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"the reasoner returned output that failed schema validation: {exc}",
        ) from exc
    except ReasonerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    grounding = check_grounding(explanation, evidence)
    return IncidentResponse(
        **base,
        reasoner=engine.describe(),
        explanation=explanation.to_dict(),
        grounding=grounding.to_dict(),
    )
