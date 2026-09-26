/**
 * The upload → poll → load state machine.
 *
 * One hook owns the whole lifecycle so no component has to reason about it:
 * idle, uploading, queued, running, complete, failed. Polling stops the moment
 * the API reports a terminal state, and every result fetched afterwards comes
 * from the endpoints the backend already exposes.
 *
 * Progress is whatever the API reports and nothing more. There is no simulated
 * bar: if `percent` is null, the UI says frames processed and leaves it there.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

import {
  ApiError,
  getAnalysis,
  getEvents,
  getIncident,
  getRisk,
  getTimeline,
  uploadVideo,
  type UploadOptions,
} from '../api/client';
import type {
  AnalysisStatusResponse,
  EventsResponse,
  IncidentResponse,
  RiskResponse,
  TimelineResponse,
} from '../api/types';

export type Phase =
  | 'idle'
  | 'uploading'
  | 'queued'
  | 'running'
  | 'loading'
  | 'complete'
  | 'failed';

export interface AnalysisState {
  phase: Phase;
  analysisId: string | null;
  status: AnalysisStatusResponse | null;
  events: EventsResponse | null;
  risk: RiskResponse | null;
  timeline: TimelineResponse | null;
  incident: IncidentResponse | null;
  error: string | null;
}

const EMPTY: AnalysisState = {
  phase: 'idle',
  analysisId: null,
  status: null,
  events: null,
  risk: null,
  timeline: null,
  incident: null,
  error: null,
};

export const POLL_INTERVAL_MS = 700;

export function useAnalysis(pollInterval: number = POLL_INTERVAL_MS) {
  const [state, setState] = useState<AnalysisState>(EMPTY);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const cancelled = useRef(false);

  const stopPolling = useCallback(() => {
    if (timer.current !== null) {
      clearTimeout(timer.current);
      timer.current = null;
    }
  }, []);

  useEffect(() => {
    cancelled.current = false;
    return () => {
      cancelled.current = true;
      stopPolling();
    };
  }, [stopPolling]);

  const loadResults = useCallback(async (id: string, status: AnalysisStatusResponse) => {
    setState((prev) => ({ ...prev, phase: 'loading', status }));
    try {
      // Fetched together: the dashboard is only meaningful with all four.
      const [events, risk, timeline, incident] = await Promise.all([
        getEvents(id),
        getRisk(id),
        getTimeline(id),
        getIncident(id),
      ]);
      if (cancelled.current) return;
      setState({
        phase: 'complete',
        analysisId: id,
        status,
        events,
        risk,
        timeline,
        incident,
        error: null,
      });
    } catch (error) {
      if (cancelled.current) return;
      setState((prev) => ({
        ...prev,
        phase: 'failed',
        error: describe(error),
      }));
    }
  }, []);

  const poll = useCallback(
    async (id: string) => {
      try {
        const status = await getAnalysis(id);
        if (cancelled.current) return;

        if (status.status === 'complete') {
          stopPolling();
          await loadResults(id, status);
          return;
        }
        if (status.status === 'failed') {
          stopPolling();
          setState((prev) => ({
            ...prev,
            phase: 'failed',
            status,
            error: status.error
              ? `${status.error.type}: ${status.error.message}`
              : 'the analysis failed without reporting a reason',
          }));
          return;
        }

        setState((prev) => ({ ...prev, phase: status.status, status }));
        timer.current = setTimeout(() => void poll(id), pollInterval);
      } catch (error) {
        if (cancelled.current) return;
        stopPolling();
        setState((prev) => ({ ...prev, phase: 'failed', error: describe(error) }));
      }
    },
    [loadResults, pollInterval, stopPolling],
  );

  const start = useCallback(
    async (file: File, options?: UploadOptions) => {
      stopPolling();
      setState({ ...EMPTY, phase: 'uploading' });
      try {
        const created = await uploadVideo(file, options);
        if (cancelled.current) return;
        setState((prev) => ({
          ...prev,
          phase: created.status === 'complete' ? 'loading' : created.status,
          analysisId: created.analysis_id,
        }));
        await poll(created.analysis_id);
      } catch (error) {
        if (cancelled.current) return;
        setState({ ...EMPTY, phase: 'failed', error: describe(error) });
      }
    },
    [poll, stopPolling],
  );

  const reset = useCallback(() => {
    stopPolling();
    setState(EMPTY);
  }, [stopPolling]);

  return { ...state, start, reset };
}

function describe(error: unknown): string {
  if (error instanceof ApiError) return error.detail;
  if (error instanceof Error) return error.message;
  return 'unexpected error';
}
