/**
 * Test fixtures captured from the real API.
 *
 * These JSON files were produced by running the actual FastAPI service against
 * the repository's synthetic clip with the mock detector — not hand-written.
 * If a response shape changes, the fixtures stop matching the types and the
 * tests notice.
 */

import deviceFixture from './fixtures/device.json';
import eventsFixture from './fixtures/events.json';
import incidentFixture from './fixtures/incident.json';
import riskFixture from './fixtures/risk.json';
import statusFixture from './fixtures/status.json';
import timelineFixture from './fixtures/timeline.json';

import type {
  AnalysisStatusResponse,
  DeviceResponse,
  EventsResponse,
  IncidentResponse,
  RiskResponse,
  TimelineResponse,
} from '../api/types';

export const status = statusFixture as unknown as AnalysisStatusResponse;
export const events = eventsFixture as unknown as EventsResponse;
export const risk = riskFixture as unknown as RiskResponse;
export const timeline = timelineFixture as unknown as TimelineResponse;
export const incident = incidentFixture as unknown as IncidentResponse;
export const device = deviceFixture as unknown as DeviceResponse;

export const ANALYSIS_ID = 'demo-analysis';

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

export interface RouteOverrides {
  health?: unknown;
  device?: unknown;
  create?: unknown;
  /** Successive responses for GET /api/analyze/{id}, consumed in order. */
  statuses?: unknown[];
  events?: unknown;
  risk?: unknown;
  timeline?: unknown;
  incident?: unknown;
  /** Routes that should fail, e.g. {'/api/health': 0} for an unreachable API. */
  fail?: Record<string, number>;
}

/**
 * A fetch stand-in that answers the real routes with the real payloads.
 *
 * `statuses` models polling: each GET of the status endpoint takes the next
 * entry and the last one repeats, so a test can walk queued → running →
 * complete without timers.
 */
export function makeFetch(overrides: RouteOverrides = {}) {
  const statuses = overrides.statuses ?? [clone(status)];
  let statusIndex = 0;

  return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = typeof input === 'string' ? input : input.toString();
    const path = url.replace(/^https?:\/\/[^/]+/, '').split('?')[0];

    for (const [failing, code] of Object.entries(overrides.fail ?? {})) {
      if (path === failing) {
        if (code === 0) throw new TypeError('network error');
        return json({ detail: `failed: ${failing}` }, code);
      }
    }

    if (path === '/api/health') return json(overrides.health ?? { status: 'ok', service: 'sentinel-api', api_version: '0.9', analyses: 0 });
    if (path === '/api/device') return json(overrides.device ?? clone(device));

    if (path === '/api/analyze' && init?.method === 'POST') {
      return json(
        overrides.create ?? {
          analysis_id: ANALYSIS_ID,
          status: 'queued',
          original_filename: 'bay.mp4',
          device: clone(device).selected,
        },
        202,
      );
    }

    if (path === `/api/analyze/${ANALYSIS_ID}`) {
      const body = statuses[Math.min(statusIndex, statuses.length - 1)];
      statusIndex += 1;
      return json(body);
    }
    if (path === `/api/analyze/${ANALYSIS_ID}/events`) return json(overrides.events ?? clone(events));
    if (path === `/api/analyze/${ANALYSIS_ID}/risk`) return json(overrides.risk ?? clone(risk));
    if (path === `/api/analyze/${ANALYSIS_ID}/timeline`) return json(overrides.timeline ?? clone(timeline));
    if (path === `/api/analyze/${ANALYSIS_ID}/incident`) return json(overrides.incident ?? clone(incident));

    return json({ detail: `unexpected route ${path}` }, 404);
  };
}

function json(body: unknown, statusCode = 200): Response {
  return new Response(JSON.stringify(body), {
    status: statusCode,
    headers: { 'content-type': 'application/json' },
  });
}

export function videoFile(name = 'bay.mp4'): File {
  return new File([new Uint8Array([0, 1, 2, 3])], name, { type: 'video/mp4' });
}

export { clone };
