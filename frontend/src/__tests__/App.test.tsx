/**
 * End-to-end flows through the dashboard, against stubbed API responses that
 * are byte-for-byte what the real service returned.
 *
 * Polling is exercised for real: the status endpoint answers queued, then
 * running, then complete, and the UI is asserted at each step.
 */

import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import App from '../App';
import {
  ANALYSIS_ID,
  clone,
  incident as incidentFixture,
  makeFetch,
  risk as riskFixture,
  status as statusFixture,
  videoFile,
  type RouteOverrides,
} from '../test/fixtures';

function mount(overrides: RouteOverrides = {}) {
  vi.stubGlobal('fetch', vi.fn(makeFetch(overrides)));
  return render(<App />);
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

async function uploadAndFinish(overrides: RouteOverrides = {}) {
  const user = userEvent.setup();
  mount(overrides);
  await screen.findByRole('button', { name: /upload surveillance video/i });
  await user.upload(screen.getByTestId('video-input'), videoFile());
  await screen.findByLabelText('Video intelligence', undefined, { timeout: 4000 });
  return user;
}

// ---------------------------------------------------------------------------
describe('landing and upload', () => {
  it('opens on the Sentinel landing state', async () => {
    mount();
    expect(screen.getByRole('heading', { name: 'SENTINEL' })).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /upload surveillance video/i }),
    ).toBeInTheDocument();
    expect(screen.getByText(/\.mp4 · \.mov/)).toBeInTheDocument();
  });

  it('reports the API connection and device in the header', async () => {
    mount();
    await waitFor(() =>
      expect(screen.getByTestId('connection-chip')).toHaveTextContent('API online'),
    );
    expect(screen.getByTestId('device-chip')).toHaveTextContent('CPU');
  });

  it('shows an AMD accelerator when the API reports one', async () => {
    const amd = clone(statusFixture);
    mount({
      device: {
        selected: {
          kind: 'rocm',
          semantic_device: 'rocm',
          label: 'AMD ROCm / HIP',
          torch_device: 'cuda',
          name: 'AMD Instinct MI300X',
          index: 0,
          runtime_version: '7.1.52802',
          requested: 'auto',
        },
        available: [],
        environment: { ...(amd as unknown as { environment?: unknown }) },
      },
    });
    await waitFor(() =>
      expect(screen.getByTestId('device-chip')).toHaveTextContent('AMD ROCm / HIP'),
    );
    expect(screen.getByTestId('device-chip')).not.toHaveTextContent('CUDA');
  });

  it('rejects an unsupported file before contacting the API', async () => {
    mount();
    const input = screen.getByTestId('video-input') as HTMLInputElement;
    const notVideo = new File(['x'], 'notes.txt', { type: 'text/plain' });
    // userEvent.upload applies the input's `accept` filter, so the file would
    // never reach the handler. Fire the change directly to exercise the guard.
    fireEvent.change(input, { target: { files: [notVideo] } });
    expect(await screen.findByTestId('upload-error')).toHaveTextContent(
      /not a supported video format/i,
    );
  });

  it('shows the API as unavailable when it cannot be reached', async () => {
    mount({ fail: { '/api/health': 0 } });
    expect(await screen.findByTestId('error-state')).toHaveTextContent(
      /Cannot reach the Sentinel API/i,
    );
    expect(screen.getByText(/uvicorn app.api.server:app/)).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
describe('analysis progress', () => {
  it('walks queued → running → complete, showing the API\'s own progress', async () => {
    const queued = clone(statusFixture);
    queued.status = 'queued';
    queued.pipeline = null;
    queued.progress = { frames_processed: 0, total_frames: 60, percent: 0 };

    const running = clone(statusFixture);
    running.status = 'running';
    running.pipeline = null;
    running.progress = { frames_processed: 30, total_frames: 60, percent: 50 };

    const user = userEvent.setup();
    mount({ statuses: [queued, running, clone(statusFixture)] });
    await screen.findByRole('button', { name: /upload surveillance video/i });
    await user.upload(screen.getByTestId('video-input'), videoFile());

    await waitFor(() =>
      expect(screen.getByTestId('progress-stage')).toHaveTextContent(/queued/i),
    );
    await waitFor(() =>
      expect(screen.getByTestId('progress-frames')).toHaveTextContent('30 / 60 frames'),
    );
    await screen.findByLabelText('Video intelligence', undefined, { timeout: 4000 });
  });

  it('does not fake a percentage the API did not report', async () => {
    const running = clone(statusFixture);
    running.status = 'running';
    running.pipeline = null;
    running.progress = { frames_processed: 12, total_frames: null, percent: null };

    const user = userEvent.setup();
    mount({ statuses: [running, clone(statusFixture)] });
    await screen.findByRole('button', { name: /upload surveillance video/i });
    await user.upload(screen.getByTestId('video-input'), videoFile());

    await waitFor(() =>
      expect(screen.getByText('progress not reported')).toBeInTheDocument(),
    );
    expect(screen.getByTestId('progress-frames')).toHaveTextContent('12 frames');
  });
});

// ---------------------------------------------------------------------------
describe('completed analysis', () => {
  it('renders the full command center', async () => {
    await uploadAndFinish();
    expect(screen.getByLabelText('Video intelligence')).toBeInTheDocument();
    expect(screen.getByLabelText('Predictive risk')).toBeInTheDocument();
    expect(screen.getByLabelText('Incident prediction')).toBeInTheDocument();
    expect(screen.getByLabelText('Why this alert')).toBeInTheDocument();
    expect(screen.getByLabelText('Event timeline')).toBeInTheDocument();
    expect(screen.getByLabelText('Incident lifecycle')).toBeInTheDocument();
    expect(screen.getByLabelText('Technical provenance')).toBeInTheDocument();
  });

  it('plays the original uploaded video from the API', async () => {
    await uploadAndFinish();
    const video = screen.getByTestId('video') as HTMLVideoElement;
    expect(video.getAttribute('src')).toContain(`/api/analyze/${ANALYSIS_ID}/video`);
  });

  it('draws the track overlay for the current frame', async () => {
    await uploadAndFinish();
    const overlay = await screen.findByTestId('track-overlay');
    expect(overlay).toHaveAttribute('viewBox', '0 0 640 384');
  });

  it('shows the score, state and disclaimer together', async () => {
    await uploadAndFinish();
    expect(screen.getByTestId('score-disclaimer')).toHaveTextContent(
      'Risk score is an engineering risk signal, not a probability.',
    );
  });

  it('seeks the video when a timeline event is clicked', async () => {
    const user = await uploadAndFinish();
    const video = screen.getByTestId('video') as HTMLVideoElement;
    const rows = within(screen.getByLabelText('Event timeline')).getAllByRole('button');
    const listRow = rows.find((row) => row.className.includes('timeline__row'))!;
    const stamp = Number(listRow.textContent?.match(/([\d.]+)s/)?.[1]);

    await user.click(listRow);
    await waitFor(() => expect(video.currentTime).toBeCloseTo(stamp, 2));
  });

  it('updates the risk panel as the playhead moves', async () => {
    await uploadAndFinish();
    const video = screen.getByTestId('video') as HTMLVideoElement;
    const peak = riskFixture.timeline.reduce((a, b) =>
      b.max_risk_score > a.max_risk_score ? b : a,
    );

    video.currentTime = peak.timestamp;
    fireEvent.timeUpdate(video);

    await waitFor(() =>
      expect(screen.getByTestId('risk-score')).toHaveTextContent(
        peak.assessments[0].risk_score.toFixed(1),
      ),
    );
    expect(screen.getByTestId('risk-state')).toHaveTextContent('CRITICAL');
  });

  it('starts over with a new video', async () => {
    const user = await uploadAndFinish();
    await user.click(screen.getByRole('button', { name: /new video/i }));
    expect(
      await screen.findByRole('button', { name: /upload surveillance video/i }),
    ).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
describe('failure and missing data', () => {
  it('shows a failed analysis with the reason the API gave', async () => {
    const failed = clone(statusFixture);
    failed.status = 'failed';
    failed.pipeline = null;
    failed.error = { type: 'IOError', message: 'Could not open video file' };

    const user = userEvent.setup();
    mount({ statuses: [failed] });
    await screen.findByRole('button', { name: /upload surveillance video/i });
    await user.upload(screen.getByTestId('video-input'), videoFile());

    const error = await screen.findByTestId('error-state');
    expect(error).toHaveTextContent('IOError');
    expect(error).toHaveTextContent('Could not open video file');
  });

  it('surfaces an upload rejection from the server', async () => {
    const user = userEvent.setup();
    mount({ fail: { '/api/analyze': 415 } });
    await screen.findByRole('button', { name: /upload surveillance video/i });
    await user.upload(screen.getByTestId('video-input'), videoFile());
    expect(await screen.findByTestId('error-state')).toHaveTextContent(/failed/i);
  });

  it('renders the dashboard when no incident was found', async () => {
    await uploadAndFinish({
      incident: {
        analysis_id: ANALYSIS_ID,
        incident_found: false,
        lifecycle_state: null,
        quantities: [],
        evidence: null,
        explanation: null,
        grounding: null,
        reasoner: null,
        note: 'the risk engine produced no assessment for this clip',
      },
    });
    expect(screen.getByTestId('no-incident')).toHaveTextContent(/no assessment/i);
    expect(screen.getByLabelText('Video intelligence')).toBeInTheDocument();
  });

  it('renders without track data', async () => {
    const empty = { ...clone(statusFixture) };
    await uploadAndFinish({
      timeline: {
        analysis_id: ANALYSIS_ID,
        coordinate_space: 'image_pixels',
        frame_count: 0,
        returned: 0,
        offset: 0,
        video: empty.video,
        frames: [],
      },
    });
    expect(screen.getByText('No tracks in this frame')).toBeInTheDocument();
  });

  it('renders when the risk timeline is empty', async () => {
    await uploadAndFinish({
      risk: {
        analysis_id: ANALYSIS_ID,
        risk_step: 0.5,
        coordinate_space: 'image_pixels',
        score_interpretation: '0-100 ordinal risk score; NOT a calibrated probability',
        max_risk_score: 0,
        max_severity: 'normal',
        report_count: 0,
        worst: null,
        timeline: [],
      },
    });
    expect(screen.getByTestId('risk-none')).toBeInTheDocument();
  });

  it('renders when the incident has no time to risk', async () => {
    const noTtr = clone(incidentFixture);
    noTtr.evidence!.time_to_risk = null;
    await uploadAndFinish({ incident: noTtr });
    expect(screen.getByTestId('prediction-ttr')).toHaveTextContent('unavailable');
  });

  it('never shows a predicted conflict as something that happened', async () => {
    const predicted = clone(incidentFixture);
    predicted.lifecycle_state = 'imminent';
    predicted.evidence!.incident_state = 'imminent';
    predicted.evidence!.time_to_risk = {
      status: 'predicted',
      seconds: 1.4,
      threshold: 90,
      reason: null,
    };
    await uploadAndFinish({ incident: predicted });
    const badge = screen.getByTestId('tense-badge');
    expect(badge).toHaveTextContent('PREDICTED');
    expect(badge).toHaveTextContent(/has not happened/i);
  });
});
