/**
 * TypeScript mirrors of the Sentinel M0.9 API schemas.
 *
 * Every field here exists in `app/api/schemas.py`. Nothing is added, renamed or
 * inferred: if the API does not return it, it is not in this file, and the UI
 * has nothing to render for it. Optional fields are optional here too, because
 * "absent" is information the interface must show rather than paper over.
 */

export type AnalysisStatus = 'queued' | 'running' | 'complete' | 'failed';

/** app/reasoning/models/risk.py:SEVERITIES */
export type Severity = 'normal' | 'low' | 'medium' | 'high' | 'critical';

/** app/intelligence/lifecycle.py:INCIDENT_STATES */
export type LifecycleState =
  | 'observed'
  | 'developing'
  | 'imminent'
  | 'current'
  | 'resolved';

/** app/accel/device.py — the semantic device, never torch's runtime string. */
export type DeviceKind = 'cpu' | 'rocm' | 'cuda' | 'mps';

export interface DeviceInfo {
  kind: DeviceKind;
  semantic_device: DeviceKind;
  label: string;
  /** PyTorch's runtime string. On ROCm this is "cuda"; it is not the device. */
  torch_device: string;
  name: string;
  index: number | null;
  runtime_version: string | null;
  requested: string | null;
}

export interface EnvironmentPayload {
  os: string;
  python: string;
  torch: string | null;
  ultralytics: string | null;
  opencv: string | null;
  numpy: string | null;
  hip: string | null;
  cuda: string | null;
  rocm_build: boolean;
  gpu_kind: string | null;
  gpus: string[];
  mps: boolean;
  cpu: string;
  cpu_count: number | null;
  ram_gb: number | null;
}

export interface DeviceResponse {
  selected: DeviceInfo;
  available: DeviceInfo[];
  environment: EnvironmentPayload;
}

export interface ProgressPayload {
  frames_processed: number;
  total_frames: number | null;
  percent: number | null;
}

export interface VideoPayload {
  path: string;
  width: number;
  height: number;
  fps: number;
  frame_count: number;
  duration_seconds: number;
}

export interface ErrorPayload {
  type: string;
  message: string;
}

export interface DetectorInfo {
  backend?: string;
  model?: string;
  device?: string;
  accelerator?: string;
  classes?: string[] | null;
}

export interface PipelineSummary {
  frames_processed: number;
  processing_seconds: number;
  processing_fps: number;
  detections_total: number;
  track_reports_total: number;
  track_ids_created: number;
  track_frames: Record<string, number>;
  event_count: number;
  entities: string[];
  detector: DetectorInfo;
  tracker: string | null;
  coordinate_space: string;
  zones: string[];
}

export interface AnalysisCreated {
  analysis_id: string;
  status: AnalysisStatus;
  original_filename: string;
  device: DeviceInfo;
}

export interface AnalysisStatusResponse {
  analysis_id: string;
  status: AnalysisStatus;
  original_filename: string;
  created_at: number;
  elapsed_seconds: number | null;
  progress: ProgressPayload;
  device: DeviceInfo | null;
  video: VideoPayload | null;
  pipeline: PipelineSummary | null;
  error: ErrorPayload | null;
  video_url: string | null;
}

export type EventAction =
  | 'appeared'
  | 'detected'
  | 'moved'
  | 'stationary'
  | 'entered_zone'
  | 'exited_zone'
  | 'disappeared';

export interface EventPayload {
  event_id?: string | null;
  timestamp: number;
  frame_index?: number | null;
  entity_id: string;
  track_id?: number | null;
  action: EventAction | string;
  position?: number[] | null;
  bbox?: number[] | null;
  coordinate_space: string;
  attributes: Record<string, unknown>;
  zones: string[];
  source?: string | null;
}

export interface EntityStatePayload {
  entity_id: string;
  class_name: string | null;
  class_id: number | null;
  track_id: number | null;
  confidence: number | null;
  position: number[] | null;
  bbox: number[] | null;
  coordinate_space: string;
  zones: string[];
  first_seen: number | null;
  last_seen: number | null;
  duration_seconds: number;
  last_action: string | null;
  present: boolean;
  event_count: number;
}

export interface EventsResponse {
  analysis_id: string;
  count: number;
  coordinate_space: string;
  action_counts: Record<string, number>;
  events: EventPayload[];
  entities: EntityStatePayload[];
}

export interface TimeToRiskPayload {
  /** already_unsafe | predicted | not_predicted */
  status: string;
  seconds: number | null;
  coordinate_space: string;
  units: string;
  reason: string | null;
  /** Unit-suffixed threshold key, e.g. unsafe_separation_threshold_px. */
  [key: string]: unknown;
}

export interface FactorPayload {
  name: string;
  score: number;
  weight: number;
  contribution: number;
  confidence: number;
  rationale: string;
  evidence_event_ids: string[];
  details: Record<string, unknown>;
  coordinate_space: string;
}

export interface RiskAssessmentPayload {
  risk_score: number;
  severity: Severity;
  incident_type: string;
  involved_entity_ids: string[];
  timestamp: number;
  confidence: number;
  predicted_time_to_incident_seconds: number | null;
  prediction_outcome: string | null;
  time_to_risk: TimeToRiskPayload | null;
  space_fallback_reason: string | null;
  contributing_factors: FactorPayload[];
  evidence_event_ids: string[];
  recommended_intervention: string;
  escalation_multiplier: number;
  coordinate_space: string;
  details: Record<string, unknown>;
}

export interface RiskReportPayload {
  timestamp: number;
  coordinate_space: string;
  score_interpretation: string;
  max_risk_score: number;
  severity: Severity;
  assessment_count: number;
  assessments: RiskAssessmentPayload[];
  metadata: Record<string, unknown>;
}

export interface RiskResponse {
  analysis_id: string;
  risk_step: number;
  coordinate_space: string;
  score_interpretation: string;
  max_risk_score: number;
  max_severity: Severity;
  report_count: number;
  worst: RiskReportPayload | null;
  timeline: RiskReportPayload[];
}

export interface TrackPayload {
  track_id: number;
  entity_id: string;
  /** [x1, y1, x2, y2] in image pixels. */
  bbox: number[];
  position: number[];
  confidence: number;
  class_id: number;
  class_name: string;
  age: number;
  hits: number;
}

export interface FrameRiskPayload {
  timestamp: number;
  risk_score: number;
  severity: Severity;
  incident_type: string;
  involved_entity_ids: string[];
  coordinate_space: string;
  time_to_risk: TimeToRiskPayload | null;
}

export interface TimelineFramePayload {
  frame_index: number;
  timestamp: number;
  tracks: TrackPayload[];
  events: EventPayload[];
  risk: FrameRiskPayload | null;
}

export interface TimelineResponse {
  analysis_id: string;
  coordinate_space: string;
  frame_count: number;
  returned: number;
  offset: number;
  video: VideoPayload | null;
  frames: TimelineFramePayload[];
}

export interface QuantityPayload {
  name: string;
  value: number;
  unit: string;
  coordinate_space: string;
}

export interface EntityEvidencePayload {
  entity_id: string;
  class_name: string | null;
  position: number[] | null;
  velocity: number[] | null;
  speed: number | null;
  zones: string[];
  coordinate_space: string;
}

export interface FactorEvidencePayload {
  name: string;
  score: number;
  weight: number;
  contribution: number;
  rationale: string;
  confidence: number;
}

export interface PredictionEvidencePayload {
  outcome: string;
  horizon_seconds: number | null;
  minimum_separation: number | null;
  seconds_to_minimum_separation: number | null;
  unsafe_separation_threshold: number | null;
  unavailable_reason: string | null;
  is_forward_looking: boolean;
}

export interface TimeToRiskEvidencePayload {
  status: string;
  seconds: number | null;
  threshold: number | null;
  reason: string | null;
}

export interface IncidentEvidencePayload {
  schema_version: string;
  incident_id: string;
  timestamp: number;
  coordinate_space: string;
  distance_unit: string;
  speed_unit: string;
  calibration_active: boolean;
  space_fallback_reason: string | null;
  entities: EntityEvidencePayload[];
  risk_score: number;
  severity: Severity;
  incident_type: string;
  incident_state: LifecycleState;
  confidence: number;
  current_separation: number | null;
  closing_speed: number | null;
  factors: FactorEvidencePayload[];
  prediction: PredictionEvidencePayload | null;
  time_to_risk: TimeToRiskEvidencePayload | null;
  triggered_event_ids: string[];
  triggered_event_actions: string[];
  recommended_intervention: string;
  score_interpretation: string;
}

export interface ExplanationPayload {
  schema_version: string;
  incident_id: string;
  summary: string;
  severity_explanation: string;
  evidence_points: string[];
  predicted_outcome: string;
  recommended_action: string;
  urgency: string;
  uncertainty: string;
  coordinate_space: string;
  provider: string;
  model: string;
  is_language_model: boolean;
}

export interface GroundingViolationPayload {
  code: string;
  detail: string;
  excerpt: string;
}

export interface GroundingPayload {
  incident_id: string;
  ok: boolean;
  violations: GroundingViolationPayload[];
  notes: string[];
}

/** One time step of the incident's lifecycle, derived by the backend tracker. */
export interface LifecyclePoint {
  timestamp: number;
  state: LifecycleState;
  risk_score: number;
}

export interface IncidentResponse {
  analysis_id: string;
  incident_found: boolean;
  /** The state at the clip's worst moment — what this whole response describes. */
  lifecycle_state: LifecycleState | null;
  /** The same situation at every risk time step, in order. May be empty. */
  lifecycle_history: LifecyclePoint[];
  quantities: QuantityPayload[];
  evidence: IncidentEvidencePayload | null;
  explanation: ExplanationPayload | null;
  grounding: GroundingPayload | null;
  reasoner: string | null;
  note: string | null;
}

export interface HealthResponse {
  status: string;
  service: string;
  api_version: string;
  analyses: number;
}
