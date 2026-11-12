/**
 * Display helpers.
 *
 * Two rules run through this file. Missing data is shown as missing — every
 * formatter has a defined answer for `null` and none of them is `0`. And a risk
 * score is never turned into a probability: it is an ordinal signal, formatted
 * as a number out of 100 and nothing else.
 */

import type { LifecycleState, Severity } from '../api/types';

/** The four operator-facing states the dashboard distinguishes. */
export type RiskState = 'NORMAL' | 'WATCH' | 'HIGH' | 'CRITICAL';

/**
 * Map the engine's five severity bands onto four display states.
 *
 * `low` and `medium` both mean "worth watching, not yet acute", so they share a
 * state. The engine's own band is always shown alongside, so nothing is lost.
 */
export function riskState(severity: Severity | null | undefined): RiskState {
  switch (severity) {
    case 'critical':
      return 'CRITICAL';
    case 'high':
      return 'HIGH';
    case 'medium':
    case 'low':
      return 'WATCH';
    default:
      return 'NORMAL';
  }
}

export const MISSING = '—';

export function formatSeconds(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return MISSING;
  return `${value.toFixed(digits)} s`;
}

export function formatClock(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return MISSING;
  const minutes = Math.floor(value / 60);
  const seconds = value - minutes * 60;
  return `${minutes}:${seconds.toFixed(2).padStart(5, '0')}`;
}

export function formatScore(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return MISSING;
  return value.toFixed(1);
}

/** A measurement with its unit. The unit is never dropped. */
export function formatQuantity(
  value: number | null | undefined,
  unit: string | null | undefined,
  digits = 2,
): string {
  if (value === null || value === undefined || Number.isNaN(value)) return MISSING;
  return `${value.toFixed(digits)}${unit ? ` ${unit}` : ''}`;
}

export function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return MISSING;
  return `${value.toFixed(0)}%`;
}

/** snake_case → readable words, used for factor and action names. */
export function humanise(value: string | null | undefined): string {
  if (!value) return MISSING;
  return value.replace(/_/g, ' ').toLowerCase();
}

export function titleise(value: string | null | undefined): string {
  if (!value) return MISSING;
  return value.replace(/_/g, ' ').toUpperCase();
}

/** app/intelligence/lifecycle.py:STATE_DESCRIPTIONS, verbatim. */
export const LIFECYCLE_DESCRIPTIONS: Record<LifecycleState, string> = {
  observed: 'seen and scored; nothing developing',
  developing: 'a future unsafe approach is predicted',
  imminent: 'predicted to become unsafe within 2s',
  current: 'the unsafe condition exists now',
  resolved: 'was active; no longer',
};

export const LIFECYCLE_ORDER: LifecycleState[] = [
  'observed',
  'developing',
  'imminent',
  'current',
  'resolved',
];

/**
 * Is this situation in the future or the present?
 *
 * The distinction the whole product rests on. It is read from the engine's own
 * `time_to_risk.status` and lifecycle state — never guessed from a score.
 */
export function tenseOf(
  timeToRiskStatus: string | null | undefined,
  lifecycle: LifecycleState | null | undefined,
): 'PREDICTED' | 'CURRENT' | 'UNKNOWN' {
  if (timeToRiskStatus === 'already_unsafe' || lifecycle === 'current') {
    return 'CURRENT';
  }
  if (timeToRiskStatus === 'predicted' || lifecycle === 'developing' || lifecycle === 'imminent') {
    return 'PREDICTED';
  }
  return 'UNKNOWN';
}

/**
 * The outcomes in which the engine says the pair is unsafe now, or predicts it
 * will be. Mirrors `UNSAFE_OUTCOMES` in `app/reasoning/prediction.py`.
 */
export const UNSAFE_PREDICTION_OUTCOMES: readonly string[] = [
  'PREDICTED_UNSAFE_PROXIMITY',
  'PREDICTED_TRAJECTORY_CONFLICT',
  'CURRENTLY_UNSAFE_PROXIMITY',
];

/**
 * What to put in front of an operator for one assessment.
 *
 * `recommended_intervention` is a fixed string the engine looks up from the
 * incident type alone (`INTERVENTIONS` in `app/reasoning/risk_engine.py`). It
 * therefore reads the same whether a conflict is predicted or not — which put
 * "alert both parties; hold the vehicle until paths diverge" on screen beside
 * "no predicted conflict" for a CONVERGING_TRAJECTORIES assessment whose
 * closest approach stays outside the unsafe separation threshold. Prescribing
 * a vehicle hold for a predicted near miss overstates what the engine found.
 *
 * So the intervention is shown only where the engine's own prediction outcome
 * says the pair is unsafe now or is predicted to be. Where it says the paths
 * stay clear, the panel says to monitor and nothing more. Where there is no
 * prediction at all, it says that instead of either prescribing an action or
 * claiming the paths are clear — a refusal, not a verdict.
 *
 * This is presentation only: the score, severity, lifecycle, prediction and
 * every backend value are untouched, and the full outcome and its reason stay
 * on screen above this line.
 */
export function recommendationFor(
  outcome: string | null | undefined,
  intervention: string | null | undefined,
): string | null {
  if (UNSAFE_PREDICTION_OUTCOMES.includes(outcome ?? '')) {
    return intervention ? intervention : null;
  }
  if (outcome === 'NO_PREDICTED_CONFLICT') {
    return 'Monitor — no conflict predicted.';
  }
  return 'Monitor — no prediction available for this situation.';
}

/**
 * A unit-suffixed number out of a payload bag, with the unit it was named in.
 *
 * The backend writes distances under a key that carries the unit —
 * `unsafe_separation_threshold_px` or `..._m`, `current_separation_px` or
 * `..._m` (`app/reasoning/models/risk.py:to_dict`,
 * `app/reasoning/prediction.py`). Which one exists depends on whether the clip
 * was calibrated, so the reader finds the key rather than assuming a unit. A
 * pixel figure is never relabelled as metres, and a missing key is missing,
 * not zero.
 */
export function unitSuffixed(
  bag: Record<string, unknown> | null | undefined,
  prefix: string,
): { value: number; unit: string } | null {
  if (!bag) return null;
  for (const [key, value] of Object.entries(bag)) {
    if (!key.startsWith(`${prefix}_`)) continue;
    const unit = key.slice(prefix.length + 1);
    if (typeof value === 'number' && Number.isFinite(value) && (unit === 'px' || unit === 'm')) {
      return { value, unit };
    }
  }
  return null;
}

/**
 * The separation at or below which this pair counts as unsafely close.
 *
 * `unsafe_separation_px` (90 by default), not `critical_radius_px` (70, the
 * radius at which the proximity *factor* saturates). The two are different
 * numbers for different jobs, and only this one decides `already_unsafe` —
 * see `app/reasoning/risk_engine.py:_time_to_risk`.
 */
export function unsafeThreshold(
  timeToRisk: Record<string, unknown> | null | undefined,
): { value: number; unit: string } | null {
  return unitSuffixed(timeToRisk, 'unsafe_separation_threshold');
}

/** How far apart this pair is at the moment the assessment describes. */
export function currentSeparation(
  details: Record<string, unknown> | null | undefined,
): { value: number; unit: string } | null {
  const prediction = details?.prediction;
  if (!prediction || typeof prediction !== 'object') return null;
  return unitSuffixed(prediction as Record<string, unknown>, 'current_separation');
}

/** Human label for a detector model string such as "yolov8n.pt". */
export function modelLabel(model: string | null | undefined): string {
  if (!model) return MISSING;
  const base = model.split('/').pop() ?? model;
  const stem = base.replace(/\.(pt|onnx|engine)$/i, '');
  return /^yolov\d/i.test(stem) ? stem.toUpperCase().replace('YOLOV', 'YOLOv') : stem;
}
