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

/** Human label for a detector model string such as "yolov8n.pt". */
export function modelLabel(model: string | null | undefined): string {
  if (!model) return MISSING;
  const base = model.split('/').pop() ?? model;
  const stem = base.replace(/\.(pt|onnx|engine)$/i, '');
  return /^yolov\d/i.test(stem) ? stem.toUpperCase().replace('YOLOV', 'YOLOv') : stem;
}
