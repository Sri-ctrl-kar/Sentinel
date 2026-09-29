/**
 * Lookups that bind the video clock to the analysis.
 *
 * Everything here is a *selection* from data the API returned. Nothing
 * interpolates: a frame is the frame nearest the current time, and a risk
 * report is the last one the engine actually produced at or before it. The UI
 * never shows a value for a moment Sentinel did not assess.
 */

import type {
  EventPayload,
  LifecyclePoint,
  RiskAssessmentPayload,
  RiskReportPayload,
  TimelineFramePayload,
} from '../api/types';

/** The frame whose timestamp is nearest `time`, or null when there are none. */
export function frameAt(
  frames: TimelineFramePayload[],
  time: number,
): TimelineFramePayload | null {
  if (frames.length === 0) return null;
  let best = frames[0];
  let bestDelta = Math.abs(best.timestamp - time);
  for (const frame of frames) {
    const delta = Math.abs(frame.timestamp - time);
    if (delta < bestDelta) {
      best = frame;
      bestDelta = delta;
    }
  }
  return best;
}

/**
 * The most recent risk report at or before `time` — a step function, not a
 * curve. Before the first assessment there is nothing to show, and `null` says
 * so.
 */
export function riskAt(
  reports: RiskReportPayload[],
  time: number,
): RiskReportPayload | null {
  let current: RiskReportPayload | null = null;
  for (const report of reports) {
    if (report.timestamp <= time + 1e-9) current = report;
    else break;
  }
  return current;
}

/**
 * The state the backend derived for the incident at or before `time`.
 *
 * A lookup, never a derivation: every state in `lifecycle_history` was produced
 * by `app/intelligence/lifecycle.py`, and this only picks which one the playhead
 * is standing in. Null before the first one, which the caller must say out loud
 * rather than falling back to the worst moment.
 */
export function lifecycleAt(
  history: LifecyclePoint[],
  time: number,
): LifecyclePoint | null {
  let current: LifecyclePoint | null = null;
  for (const point of history) {
    if (point.timestamp <= time + 1e-9) current = point;
    else break;
  }
  return current;
}

/** Stable identity for "the same situation", mirroring `situation_key`. */
function situationKey(entityIds: string[]): string {
  return [...entityIds].sort().join('\u0000');
}

/**
 * This incident's own assessment at or before `time`.
 *
 * Scoped to one entity pair, so the panel keeps following the situation it is
 * about instead of jumping to whichever pair happens to score highest at the
 * playhead — that is what `riskAt` is for. Same step-function rule as the rest
 * of this module: the last assessment the engine actually produced, or null.
 */
export function assessmentAt(
  reports: RiskReportPayload[],
  time: number,
  entityIds: string[],
): RiskAssessmentPayload | null {
  if (entityIds.length === 0) return null;
  const key = situationKey(entityIds);
  let current: RiskAssessmentPayload | null = null;
  for (const report of reports) {
    if (report.timestamp > time + 1e-9) break;
    for (const assessment of report.assessments) {
      if (situationKey(assessment.involved_entity_ids) === key) {
        current = assessment;
        break;
      }
    }
  }
  return current;
}

/** The highest-scoring report in the run, or null when none was produced. */
export function peakRisk(
  reports: RiskReportPayload[],
): RiskReportPayload | null {
  let peak: RiskReportPayload | null = null;
  for (const report of reports) {
    if (report.assessment_count === 0) continue;
    if (!peak || report.max_risk_score > peak.max_risk_score) peak = report;
  }
  return peak;
}

export interface TimelineMarker {
  key: string;
  timestamp: number;
  kind: 'event' | 'risk';
  label: string;
  detail: string;
  entityId?: string;
  severity?: string;
}

/**
 * Markers for the scrub track: every recorded event, plus each moment the risk
 * severity changed. Severity transitions are derived by comparing consecutive
 * reports the engine produced — no new assessment is invented.
 */
export function buildMarkers(
  events: EventPayload[],
  reports: RiskReportPayload[],
): TimelineMarker[] {
  const markers: TimelineMarker[] = events.map((event, index) => ({
    key: event.event_id ?? `event-${index}-${event.timestamp}`,
    timestamp: event.timestamp,
    kind: 'event',
    label: event.action,
    detail: event.zones.length
      ? `${event.entity_id} · ${event.zones.join(', ')}`
      : event.entity_id,
    entityId: event.entity_id,
  }));

  let previous: string | null = null;
  for (const report of reports) {
    const severity = report.assessment_count > 0 ? report.severity : 'normal';
    if (severity !== previous) {
      if (previous !== null) {
        markers.push({
          key: `risk-${report.timestamp}`,
          timestamp: report.timestamp,
          kind: 'risk',
          label: `risk → ${severity}`,
          detail: `score ${report.max_risk_score.toFixed(1)}`,
          severity,
        });
      }
      previous = severity;
    }
  }

  return markers.sort((a, b) => a.timestamp - b.timestamp);
}

/** Total clip length, preferring the video's own duration. */
export function clipDuration(
  videoDuration: number | null | undefined,
  frames: TimelineFramePayload[],
): number {
  if (videoDuration && videoDuration > 0) return videoDuration;
  if (frames.length > 0) return frames[frames.length - 1].timestamp;
  return 0;
}
