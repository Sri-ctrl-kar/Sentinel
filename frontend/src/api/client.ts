/**
 * The Sentinel API client.
 *
 * One place knows the base URL, one place knows how a failure is reported. The
 * backend returns errors as `{detail: "..."}`; `ApiError` carries that through
 * so the UI can show what the server actually said rather than "request
 * failed".
 */

import type {
  AnalysisCreated,
  AnalysisStatusResponse,
  DeviceResponse,
  EventsResponse,
  HealthResponse,
  IncidentResponse,
  RiskResponse,
  TimelineResponse,
} from './types';

/** Configurable per environment; see frontend/.env.example. */
export const API_BASE_URL: string = (
  import.meta.env?.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'
)
  .toString()
  .replace(/\/$/, '');

export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

/** Absolute URL for a path, for `<video src>` and links. */
export function apiUrl(path: string): string {
  return `${API_BASE_URL}${path.startsWith('/') ? path : `/${path}`}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(apiUrl(path), init);
  } catch (cause) {
    // A network-level failure: the API is not running, or CORS blocked it.
    throw new ApiError(
      0,
      `cannot reach the Sentinel API at ${API_BASE_URL}. Is it running?`,
    );
  }

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body && typeof body.detail === 'string') detail = body.detail;
      else if (Array.isArray(body?.detail)) {
        detail = body.detail
          .map((d: { msg?: string }) => d?.msg ?? 'invalid request')
          .join('; ');
      }
    } catch {
      /* a non-JSON error body: keep the status line */
    }
    throw new ApiError(response.status, detail);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/api/health');
}

export function getDevice(): Promise<DeviceResponse> {
  return request<DeviceResponse>('/api/device');
}

export interface UploadOptions {
  detector?: string;
  sampleInterval?: number;
  riskStep?: number;
  maxFrames?: number;
  zones?: string;
  operatingZones?: string;
}

export function uploadVideo(
  file: File,
  options: UploadOptions = {},
): Promise<AnalysisCreated> {
  const form = new FormData();
  form.append('video', file);
  if (options.detector) form.append('detector', options.detector);
  if (options.sampleInterval !== undefined) {
    form.append('sample_interval', String(options.sampleInterval));
  }
  if (options.riskStep !== undefined) {
    form.append('risk_step', String(options.riskStep));
  }
  if (options.maxFrames !== undefined) {
    form.append('max_frames', String(options.maxFrames));
  }
  if (options.zones) form.append('zones', options.zones);
  if (options.operatingZones) {
    form.append('operating_zones', options.operatingZones);
  }
  return request<AnalysisCreated>('/api/analyze', {
    method: 'POST',
    body: form,
  });
}

export function getAnalysis(id: string): Promise<AnalysisStatusResponse> {
  return request<AnalysisStatusResponse>(`/api/analyze/${id}`);
}

export function getEvents(id: string): Promise<EventsResponse> {
  return request<EventsResponse>(`/api/analyze/${id}/events`);
}

export function getRisk(id: string): Promise<RiskResponse> {
  return request<RiskResponse>(`/api/analyze/${id}/risk`);
}

export function getTimeline(id: string): Promise<TimelineResponse> {
  return request<TimelineResponse>(`/api/analyze/${id}/timeline`);
}

export function getIncident(
  id: string,
  reasoner = 'mock',
): Promise<IncidentResponse> {
  return request<IncidentResponse>(
    `/api/analyze/${id}/incident?reasoner=${encodeURIComponent(reasoner)}`,
  );
}

export function videoUrl(id: string): string {
  return apiUrl(`/api/analyze/${id}/video`);
}
