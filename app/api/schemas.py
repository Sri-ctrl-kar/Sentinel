"""Pydantic response schemas for the M0.9 API.

These mirror the ``to_dict()`` contracts the validated core already publishes.
They add no semantics: every field here exists because a core object produces
it, and where the core carries an open-ended mapping (``Event.attributes``,
``RiskAssessment.details``, factor ``details``) so does the schema.

``tests/test_api_contract.py`` includes drift guards: for each mirrored payload
it asserts that every key the core's ``to_dict()`` emits is declared here. If
M0.10 adds a field to ``IncidentEvidence``, a test says so rather than the API
silently dropping it.

Field names match the core's keys exactly, so a frontend reading the API and a
script reading a CLI-produced JSON file see the same names.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..intelligence.evidence import EVIDENCE_SCHEMA_VERSION
from ..intelligence.schema import EXPLANATION_SCHEMA_VERSION
from .store import STATUSES


class ApiModel(BaseModel):
    """Base: reject unknown fields so a response shape cannot drift silently."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
class DeviceInfo(ApiModel):
    """Sentinel's semantic device, never PyTorch's internal string.

    ``kind`` / ``semantic_device`` are the answer a caller should read:
    ``cpu``, ``rocm``, ``cuda`` or ``mps``. ``torch_device`` is included under
    its own name for completeness and is the *runtime* string — on ROCm it is
    ``"cuda"``, which is correct for torch and wrong as a device identity.
    """

    kind: str = Field(description="Semantic device: cpu | rocm | cuda | mps")
    semantic_device: str = Field(description="Same as kind; explicit for clients")
    label: str = Field(description='Human label, e.g. "AMD ROCm / HIP"')
    torch_device: str = Field(
        description="PyTorch runtime string. On ROCm this is 'cuda'; it is not "
        "the semantic device."
    )
    name: str = ""
    index: Optional[int] = None
    runtime_version: Optional[str] = None
    requested: Optional[str] = None


class EnvironmentPayload(ApiModel):
    os: str
    python: str
    torch: Optional[str] = None
    ultralytics: Optional[str] = None
    opencv: Optional[str] = None
    numpy: Optional[str] = None
    hip: Optional[str] = None
    cuda: Optional[str] = None
    rocm_build: bool = False
    gpu_kind: Optional[str] = None
    gpus: List[str] = Field(default_factory=list)
    mps: bool = False
    cpu: str = "unknown"
    cpu_count: Optional[int] = None
    ram_gb: Optional[float] = None


class DeviceResponse(ApiModel):
    selected: DeviceInfo
    available: List[DeviceInfo]
    environment: EnvironmentPayload


# ---------------------------------------------------------------------------
# Analysis lifecycle
# ---------------------------------------------------------------------------
class AnalysisOptions(ApiModel):
    """Everything a caller may set, all of it an existing PipelineConfig knob."""

    detector: str = "yolo"
    weights: Optional[str] = None
    device: str = "auto"
    imgsz: Optional[int] = None
    confidence: Optional[float] = None
    classes: Optional[List[str]] = None
    stride: Optional[int] = None
    max_frames: Optional[int] = None
    sample_interval: Optional[float] = Field(
        default=None,
        description="Seconds between sampled 'moved'/'stationary' events. An "
        "existing PipelineConfig knob; the validated default (1.0) applies when "
        "omitted. Frame-level overlay data does not depend on this.",
    )
    zones: List[str] = Field(
        default_factory=list,
        description="Inline image-space zones, 'NAME=X1,Y1,X2,Y2'",
    )
    operating_zones: List[str] = Field(default_factory=list)
    risk_step: float = Field(default=0.5, gt=0.0)


class ProgressPayload(ApiModel):
    frames_processed: int = 0
    total_frames: Optional[int] = None
    percent: Optional[float] = None


class VideoPayload(ApiModel):
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


class ErrorPayload(ApiModel):
    type: str
    message: str


class AnalysisCreated(ApiModel):
    analysis_id: str
    status: str = Field(description=" | ".join(STATUSES))
    original_filename: str
    device: DeviceInfo


class PipelineSummary(ApiModel):
    """The run's own account of itself, from PipelineResult."""

    frames_processed: int
    processing_seconds: float
    processing_fps: float
    detections_total: int
    track_reports_total: int
    track_ids_created: int
    track_frames: Dict[str, int] = Field(default_factory=dict)
    event_count: int
    entities: List[str] = Field(default_factory=list)
    detector: Dict[str, Any] = Field(default_factory=dict)
    tracker: Optional[str] = None
    coordinate_space: str
    zones: List[str] = Field(default_factory=list)


class AnalysisStatusResponse(ApiModel):
    analysis_id: str
    status: str = Field(description=" | ".join(STATUSES))
    original_filename: str
    created_at: float
    elapsed_seconds: Optional[float] = None
    progress: ProgressPayload
    device: Optional[DeviceInfo] = None
    video: Optional[VideoPayload] = None
    pipeline: Optional[PipelineSummary] = None
    error: Optional[ErrorPayload] = None
    video_url: Optional[str] = None


class AnalysisListResponse(ApiModel):
    count: int
    analyses: List[AnalysisStatusResponse]


# ---------------------------------------------------------------------------
# Events (mirrors app/events/schema.py:Event.to_dict)
# ---------------------------------------------------------------------------
class EventPayload(ApiModel):
    event_id: Optional[str] = None
    timestamp: float
    frame_index: Optional[int] = None
    entity_id: str
    track_id: Optional[int] = None
    action: str
    position: Optional[List[float]] = None
    bbox: Optional[List[float]] = None
    coordinate_space: str
    attributes: Dict[str, Any] = Field(default_factory=dict)
    zones: List[str] = Field(default_factory=list)
    source: Optional[str] = None


class EntityStatePayload(ApiModel):
    entity_id: str
    class_name: Optional[str] = None
    class_id: Optional[int] = None
    track_id: Optional[int] = None
    confidence: Optional[float] = None
    position: Optional[List[float]] = None
    bbox: Optional[List[float]] = None
    coordinate_space: str
    zones: List[str] = Field(default_factory=list)
    first_seen: Optional[float] = None
    last_seen: Optional[float] = None
    duration_seconds: float = 0.0
    last_action: Optional[str] = None
    present: bool = False
    event_count: int = 0


class EventsResponse(ApiModel):
    analysis_id: str
    count: int
    coordinate_space: str
    action_counts: Dict[str, int] = Field(default_factory=dict)
    events: List[EventPayload]
    entities: List[EntityStatePayload] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Risk (mirrors app/reasoning/models/risk.py)
# ---------------------------------------------------------------------------
class FactorPayload(ApiModel):
    name: str
    score: float
    weight: float
    contribution: float
    confidence: float
    rationale: str = ""
    evidence_event_ids: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)
    coordinate_space: str


class TimeToRiskPayload(ApiModel):
    model_config = ConfigDict(extra="allow")  # threshold key is unit-suffixed

    status: str
    seconds: Optional[float] = None
    coordinate_space: str
    units: str
    reason: Optional[str] = None


class RiskAssessmentPayload(ApiModel):
    risk_score: float
    severity: str
    incident_type: str
    involved_entity_ids: List[str]
    timestamp: float
    confidence: float
    predicted_time_to_incident_seconds: Optional[float] = None
    prediction_outcome: Optional[str] = None
    time_to_risk: Optional[TimeToRiskPayload] = None
    space_fallback_reason: Optional[str] = None
    contributing_factors: List[FactorPayload] = Field(default_factory=list)
    evidence_event_ids: List[str] = Field(default_factory=list)
    recommended_intervention: str = ""
    escalation_multiplier: float = 1.0
    coordinate_space: str
    details: Dict[str, Any] = Field(default_factory=dict)


class RiskReportPayload(ApiModel):
    timestamp: float
    coordinate_space: str
    score_interpretation: str
    max_risk_score: float
    severity: str
    assessment_count: int
    assessments: List[RiskAssessmentPayload] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RiskResponse(ApiModel):
    analysis_id: str
    risk_step: float
    coordinate_space: str
    score_interpretation: str = Field(
        description="The risk score is an ordinal engineering signal, not a "
        "calibrated probability."
    )
    max_risk_score: float
    max_severity: str
    report_count: int
    worst: Optional[RiskReportPayload] = None
    timeline: List[RiskReportPayload] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Frame-level timeline (the API's own addition — see app/api/runner.py)
# ---------------------------------------------------------------------------
class TrackPayload(ApiModel):
    track_id: int
    entity_id: str
    bbox: List[float]
    position: List[float]
    confidence: float
    class_id: int
    class_name: str
    age: int
    hits: int


class FrameRiskPayload(ApiModel):
    timestamp: float
    risk_score: float
    severity: str
    incident_type: str
    involved_entity_ids: List[str] = Field(default_factory=list)
    coordinate_space: str
    time_to_risk: Optional[TimeToRiskPayload] = None


class TimelineFramePayload(ApiModel):
    frame_index: int
    timestamp: float
    tracks: List[TrackPayload] = Field(default_factory=list)
    events: List[EventPayload] = Field(default_factory=list)
    risk: Optional[FrameRiskPayload] = None


class TimelineResponse(ApiModel):
    analysis_id: str
    coordinate_space: str
    frame_count: int
    returned: int
    offset: int
    video: Optional[VideoPayload] = None
    frames: List[TimelineFramePayload] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Incident intelligence (mirrors app/intelligence/*)
# ---------------------------------------------------------------------------
class QuantityPayload(ApiModel):
    name: str
    value: float
    unit: str
    coordinate_space: str


class EntityEvidencePayload(ApiModel):
    entity_id: str
    class_name: Optional[str] = None
    position: Optional[List[float]] = None
    velocity: Optional[List[float]] = None
    speed: Optional[float] = None
    zones: List[str] = Field(default_factory=list)
    coordinate_space: str


class FactorEvidencePayload(ApiModel):
    name: str
    score: float
    weight: float
    contribution: float
    rationale: str = ""
    confidence: float = 0.0


class PredictionEvidencePayload(ApiModel):
    outcome: str
    horizon_seconds: Optional[float] = None
    minimum_separation: Optional[float] = None
    seconds_to_minimum_separation: Optional[float] = None
    unsafe_separation_threshold: Optional[float] = None
    unavailable_reason: Optional[str] = None
    is_forward_looking: bool = False


class TimeToRiskEvidencePayload(ApiModel):
    status: str
    seconds: Optional[float] = None
    threshold: Optional[float] = None
    reason: Optional[str] = None


class IncidentEvidencePayload(ApiModel):
    schema_version: str = EVIDENCE_SCHEMA_VERSION
    incident_id: str
    timestamp: float
    coordinate_space: str
    distance_unit: str
    speed_unit: str
    calibration_active: bool = False
    space_fallback_reason: Optional[str] = None
    entities: List[EntityEvidencePayload] = Field(default_factory=list)
    risk_score: float
    severity: str
    incident_type: str
    incident_state: str
    confidence: float
    current_separation: Optional[float] = None
    closing_speed: Optional[float] = None
    factors: List[FactorEvidencePayload] = Field(default_factory=list)
    prediction: Optional[PredictionEvidencePayload] = None
    time_to_risk: Optional[TimeToRiskEvidencePayload] = None
    triggered_event_ids: List[str] = Field(default_factory=list)
    triggered_event_actions: List[str] = Field(default_factory=list)
    recommended_intervention: str = ""
    score_interpretation: str


class ExplanationPayload(ApiModel):
    schema_version: str = EXPLANATION_SCHEMA_VERSION
    incident_id: str = ""
    summary: str
    severity_explanation: str
    evidence_points: List[str]
    predicted_outcome: str
    recommended_action: str
    urgency: str
    uncertainty: str
    coordinate_space: str
    provider: str = ""
    model: str = ""
    is_language_model: bool = False


class GroundingViolationPayload(ApiModel):
    code: str
    detail: str
    excerpt: str = ""


class GroundingPayload(ApiModel):
    incident_id: str
    ok: bool
    violations: List[GroundingViolationPayload] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class IncidentResponse(ApiModel):
    """The M0.7 triple, over a video analysis instead of a scenario."""

    analysis_id: str
    incident_found: bool
    lifecycle_state: Optional[str] = None
    quantities: List[QuantityPayload] = Field(default_factory=list)
    evidence: Optional[IncidentEvidencePayload] = None
    explanation: Optional[ExplanationPayload] = None
    grounding: Optional[GroundingPayload] = None
    reasoner: Optional[str] = None
    note: Optional[str] = None


class HealthResponse(ApiModel):
    status: str
    service: str
    api_version: str
    analyses: int
