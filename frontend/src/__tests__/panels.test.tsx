/**
 * Panel rendering against real API payloads.
 *
 * The fixtures were captured from the running service, so these tests assert
 * what an operator will actually see — including the cases that matter most:
 * a predicted conflict shown as predicted, a missing measurement shown as
 * missing, and a risk score never restated as a probability.
 */

import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { IncidentResponse, RiskResponse } from '../api/types';
import { EvidencePanel } from '../components/EvidencePanel';
import { LifecycleTrail } from '../components/LifecycleTrail';
import { NarrativePanel } from '../components/NarrativePanel';
import { PredictionPanel } from '../components/PredictionPanel';
import { RiskPanel } from '../components/RiskPanel';
import { TrackOverlay } from '../components/TrackOverlay';
import { TrustPanel } from '../components/TrustPanel';
import { VideoIntelligence } from '../components/VideoIntelligence';
import {
  clone,
  incident as incidentFixture,
  risk as riskFixture,
  status as statusFixture,
  timeline as timelineFixture,
} from '../test/fixtures';
import { peakRisk } from '../lib/timeline';

const peak = peakRisk(riskFixture.timeline)!;

// ---------------------------------------------------------------------------
describe('track overlay', () => {
  const frame = timelineFixture.frames.find((f) => f.tracks.length > 0)!;

  it('draws a box per track, in the video pixel space', () => {
    render(<TrackOverlay tracks={frame.tracks} width={640} height={384} />);
    const overlay = screen.getByTestId('track-overlay');
    expect(overlay).toHaveAttribute('viewBox', '0 0 640 384');
    for (const track of frame.tracks) {
      expect(screen.getByTestId(`track-${track.track_id}`)).toBeInTheDocument();
    }
  });

  it('labels each box with its track id, class and confidence', () => {
    render(<TrackOverlay tracks={frame.tracks} width={640} height={384} />);
    const track = frame.tracks[0];
    const group = screen.getByTestId(`track-${track.track_id}`);
    const label = within(group).getByText(new RegExp(`#${track.track_id}`));
    expect(label.textContent).toContain(track.class_name);
    expect(label.textContent).toContain(`${(track.confidence * 100).toFixed(0)}%`);
  });

  it('uses the exact coordinates the API returned', () => {
    render(<TrackOverlay tracks={frame.tracks} width={640} height={384} />);
    const [x1, y1, x2, y2] = frame.tracks[0].bbox;
    const rect = screen
      .getByTestId(`track-${frame.tracks[0].track_id}`)
      .querySelector('rect')!;
    expect(rect.getAttribute('x')).toBe(String(x1));
    expect(rect.getAttribute('y')).toBe(String(y1));
    expect(rect.getAttribute('width')).toBe(String(x2 - x1));
    expect(rect.getAttribute('height')).toBe(String(y2 - y1));
  });

  it('emphasises the entities the risk engine named', () => {
    const involved = [frame.tracks[0].entity_id];
    render(
      <TrackOverlay tracks={frame.tracks} width={640} height={384} involved={involved} />,
    );
    const emphasised = screen
      .getByTestId(`track-${frame.tracks[0].track_id}`)
      .querySelector('rect')!;
    expect(emphasised.getAttribute('stroke-width')).toBe('3');
  });

  it('renders an empty, labelled overlay when a frame has no tracks', () => {
    render(<TrackOverlay tracks={[]} width={640} height={384} />);
    expect(screen.getByTestId('track-overlay')).toHaveAttribute(
      'aria-label',
      'no tracked objects in this frame',
    );
  });

  it('renders nothing when the video dimensions are unknown', () => {
    const { container } = render(<TrackOverlay tracks={[]} width={0} height={0} />);
    expect(container).toBeEmptyDOMElement();
  });
});

// ---------------------------------------------------------------------------
describe('risk panel', () => {
  it('shows the score, the display state and the engine band', () => {
    render(<RiskPanel risk={riskFixture} current={peak} currentTime={peak.timestamp} />);
    const expected = peak.assessments[0].risk_score.toFixed(1);
    expect(screen.getByTestId('risk-score')).toHaveTextContent(expected);
    expect(screen.getByTestId('risk-state')).toHaveTextContent('CRITICAL');
    expect(screen.getByText(/engine severity band: CRITICAL/i)).toBeInTheDocument();
  });

  it('names the moment the assessment belongs to', () => {
    render(<RiskPanel risk={riskFixture} current={peak} currentTime={peak.timestamp} />);
    expect(screen.getByTestId('risk-asof')).toHaveTextContent(
      `${peak.timestamp.toFixed(2)} s`,
    );
  });

  it('never presents the score as a percentage or a probability', () => {
    const { container } = render(
      <RiskPanel risk={riskFixture} current={peak} currentTime={peak.timestamp} />,
    );
    const text = container.textContent ?? '';
    expect(text).toContain('/100');
    expect(text).not.toMatch(/probability/i);
    expect(text).not.toMatch(/\d+% (chance|likely|probability)/i);
  });

  it('says there is no assessment rather than showing zero', () => {
    render(<RiskPanel risk={riskFixture} current={null} currentTime={0} />);
    expect(screen.getByTestId('risk-none')).toBeInTheDocument();
    expect(screen.queryByTestId('risk-score')).not.toBeInTheDocument();
  });

  it('shows a present breach as happening now, not as a countdown', () => {
    render(<RiskPanel risk={riskFixture} current={peak} currentTime={peak.timestamp} />);
    expect(screen.getByTestId('ttr-value')).toHaveTextContent('UNSAFE NOW');
    expect(screen.getByText(/present condition, not a prediction/i)).toBeInTheDocument();
  });

  it('shows a predicted crossing as a countdown', () => {
    const predicted = clone(peak);
    predicted.assessments[0].time_to_risk = {
      status: 'predicted',
      seconds: 1.4,
      coordinate_space: 'image_pixels',
      units: 'pixels',
      reason: null,
    };
    render(<RiskPanel risk={riskFixture} current={predicted} currentTime={0.5} />);
    expect(screen.getByTestId('ttr-value')).toHaveTextContent('1.40 s');
    expect(screen.getByText(/Predicted time until/i)).toBeInTheDocument();
  });

  it('says time to risk is unavailable, with the reason, when there is none', () => {
    const notPredicted = clone(peak);
    notPredicted.assessments[0].time_to_risk = {
      status: 'not_predicted',
      seconds: null,
      coordinate_space: 'image_pixels',
      units: 'pixels',
      reason: 'insufficient_history',
    };
    render(<RiskPanel risk={riskFixture} current={notPredicted} currentTime={0.5} />);
    expect(screen.getByTestId('ttr-value')).toHaveTextContent('unavailable');
    expect(screen.getByText(/insufficient history/i)).toBeInTheDocument();
  });

  it('handles a missing time_to_risk object entirely', () => {
    const missing = clone(peak);
    missing.assessments[0].time_to_risk = null;
    render(<RiskPanel risk={riskFixture} current={missing} currentTime={0.5} />);
    expect(screen.getByTestId('ttr-value')).toHaveTextContent('unavailable');
  });
});

// ---------------------------------------------------------------------------
describe('prediction panel', () => {
  it('marks a present breach as CURRENT, never as predicted', () => {
    render(<PredictionPanel incident={incidentFixture} />);
    const badge = screen.getByTestId('tense-badge');
    expect(badge).toHaveTextContent('CURRENT');
    expect(badge).toHaveTextContent(/happening now/i);
    expect(badge).not.toHaveTextContent('PREDICTED');
  });

  it('marks a forward-looking conflict as PREDICTED and says it has not happened', () => {
    const predicted = clone(incidentFixture) as IncidentResponse;
    predicted.lifecycle_state = 'imminent';
    predicted.evidence!.incident_state = 'imminent';
    predicted.evidence!.time_to_risk = {
      status: 'predicted',
      seconds: 1.4,
      threshold: 90,
      reason: null,
    };
    render(<PredictionPanel incident={predicted} />);
    const badge = screen.getByTestId('tense-badge');
    expect(badge).toHaveTextContent('PREDICTED');
    expect(badge).toHaveTextContent(/has not happened/i);
    expect(screen.getByTestId('prediction-ttr')).toHaveTextContent('1.40 s');
  });

  it('shows the incident type, lifecycle and involved entities from the API', () => {
    render(<PredictionPanel incident={incidentFixture} />);
    expect(screen.getByTestId('incident-type')).toHaveTextContent(
      incidentFixture.evidence!.incident_type.replace(/_/g, ' '),
    );
    expect(screen.getByTestId('lifecycle-value')).toHaveTextContent(
      incidentFixture.evidence!.incident_state.toUpperCase(),
    );
    const entities = screen.getByTestId('involved-entities');
    for (const entity of incidentFixture.evidence!.entities) {
      expect(entities).toHaveTextContent(entity.entity_id);
    }
  });

  it('reports honestly when no incident was raised', () => {
    const none: IncidentResponse = {
      analysis_id: 'demo-analysis',
      incident_found: false,
      lifecycle_state: null,
      lifecycle_history: [],
      quantities: [],
      evidence: null,
      explanation: null,
      grounding: null,
      reasoner: null,
      note: 'the risk engine produced no assessment for this clip',
    };
    render(<PredictionPanel incident={none} />);
    expect(screen.getByTestId('no-incident')).toHaveTextContent(/no assessment/i);
  });

  it('survives a missing prediction block', () => {
    const noPrediction = clone(incidentFixture) as IncidentResponse;
    noPrediction.evidence!.prediction = null;
    render(<PredictionPanel incident={noPrediction} />);
    expect(screen.queryByTestId('prediction-outcome')).not.toBeInTheDocument();
    expect(screen.getByTestId('incident-type')).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
describe('evidence panel', () => {
  it('lists only the factors that actually scored', () => {
    render(<EvidencePanel incident={incidentFixture} />);
    const list = screen.getByTestId('evidence-list');
    const active = incidentFixture.evidence!.factors.filter((f) => f.contribution > 0);
    const inactive = incidentFixture.evidence!.factors.filter((f) => f.contribution === 0);
    expect(active.length).toBeGreaterThan(0);
    for (const factor of active) {
      expect(list).toHaveTextContent(factor.name.replace(/_/g, ' '));
    }
    for (const factor of inactive) {
      expect(list).not.toHaveTextContent(factor.rationale);
    }
  });

  it('shows each factor rationale exactly as the engine wrote it', () => {
    render(<EvidencePanel incident={incidentFixture} />);
    const first = incidentFixture.evidence!.factors.find((f) => f.contribution > 0)!;
    expect(screen.getByText(first.rationale)).toBeInTheDocument();
  });

  it('shows every measurement with its unit', () => {
    render(<EvidencePanel incident={incidentFixture} />);
    const measures = screen.getByTestId('evidence-measures');
    for (const quantity of incidentFixture.quantities) {
      expect(measures).toHaveTextContent(quantity.name);
      expect(measures).toHaveTextContent(quantity.unit);
    }
  });

  it('warns that pixel distances are not physical distances', () => {
    render(<EvidencePanel incident={incidentFixture} />);
    expect(screen.getByText(/not physical distance/i)).toBeInTheDocument();
  });

  it('renders nothing invented when there is no evidence', () => {
    render(<EvidencePanel incident={null} />);
    expect(screen.getByText(/No incident evidence/i)).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
describe('lifecycle trail', () => {
  it("shows the engine's five states and marks the current one", () => {
    render(<LifecycleTrail state={incidentFixture.lifecycle_state} />);
    const trail = screen.getByTestId('lifecycle-trail');
    for (const step of ['OBSERVED', 'DEVELOPING', 'IMMINENT', 'CURRENT', 'RESOLVED']) {
      expect(trail).toHaveTextContent(step);
    }
    expect(screen.getByTestId('lifecycle-current')).toHaveTextContent(
      incidentFixture.lifecycle_state!.toUpperCase(),
    );
  });

  it('marks the current state with a word, not only a colour', () => {
    render(<LifecycleTrail state="imminent" />);
    expect(screen.getByTestId('lifecycle-current')).toHaveTextContent('NOW');
  });

  it('lights the states the clip passed through before its worst moment', () => {
    // The captured clip is imminent for seven steps before it becomes unsafe.
    // Showing only `lifecycle_state` would report CURRENT and nothing else.
    render(
      <LifecycleTrail
        state={incidentFixture.lifecycle_state}
        history={incidentFixture.lifecycle_history}
      />,
    );
    expect(screen.getByTestId('lifecycle-at-imminent')).toBeInTheDocument();
    expect(screen.getByTestId('lifecycle-at-current')).toBeInTheDocument();
    expect(screen.getByTestId('lifecycle-at-observed')).toBeInTheDocument();
  });

  it('leaves a state dark when the situation never entered it', () => {
    // This clip never developed: every prediction landed inside the 2s horizon,
    // so DEVELOPING must stay unlit rather than be filled in as a ladder rung.
    expect(
      incidentFixture.lifecycle_history.some((p) => p.state === 'developing'),
    ).toBe(false);

    render(
      <LifecycleTrail
        state={incidentFixture.lifecycle_state}
        history={incidentFixture.lifecycle_history}
      />,
    );
    expect(screen.queryByTestId('lifecycle-at-developing')).toBeNull();
    expect(
      screen.getByTestId('lifecycle-trail').querySelectorAll('.lifecycle__dot--reached'),
    ).toHaveLength(3);
  });

  it('says when each state was first entered, using the engine timestamps', () => {
    const first = (state: string) =>
      incidentFixture.lifecycle_history.find((p) => p.state === state)!.timestamp;

    render(
      <LifecycleTrail
        state={incidentFixture.lifecycle_state}
        history={incidentFixture.lifecycle_history}
      />,
    );
    expect(screen.getByTestId('lifecycle-at-imminent')).toHaveTextContent(
      first('imminent').toFixed(2),
    );
    expect(screen.getByTestId('lifecycle-at-current')).toHaveTextContent(
      first('current').toFixed(2),
    );
  });

  it('falls back to the ladder when no history is available', () => {
    render(<LifecycleTrail state="current" history={[]} />);
    expect(screen.queryByTestId('lifecycle-at-current')).toBeNull();
    expect(
      screen.getByTestId('lifecycle-trail').querySelectorAll('.lifecycle__dot--reached'),
    ).toHaveLength(4);
  });

  it('says there is no lifecycle when none was returned', () => {
    render(<LifecycleTrail state={null} />);
    expect(screen.getByText(/No incident lifecycle/i)).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
describe('trust panel', () => {
  const grounding = incidentFixture.grounding!;

  it('reports the device label the API returned', () => {
    render(
      <TrustPanel
        device={statusFixture.device}
        pipeline={statusFixture.pipeline}
        grounding={grounding}
        coordinateSpace="image_pixels"
      />,
    );
    expect(screen.getByTestId('trust-device')).toHaveTextContent(
      statusFixture.device!.label,
    );
  });

  it('never labels an AMD device as CUDA', () => {
    const amd = {
      ...statusFixture.device!,
      kind: 'rocm' as const,
      semantic_device: 'rocm' as const,
      label: 'AMD ROCm / HIP',
      torch_device: 'cuda',
      name: 'AMD Instinct MI300X',
      runtime_version: '7.1.52802',
    };
    render(
      <TrustPanel
        device={amd}
        pipeline={statusFixture.pipeline}
        grounding={grounding}
        coordinateSpace="image_pixels"
      />,
    );
    expect(screen.getByTestId('trust-device')).toHaveTextContent('AMD ROCm / HIP');
    expect(screen.getByTestId('trust-device')).not.toHaveTextContent('CUDA');
  });

  it('shows the grounding verdict from the backend', () => {
    render(
      <TrustPanel
        device={statusFixture.device}
        pipeline={statusFixture.pipeline}
        grounding={grounding}
        coordinateSpace="image_pixels"
      />,
    );
    expect(screen.getByTestId('trust-grounding')).toHaveTextContent(
      grounding.ok ? 'PASS' : 'FAIL',
    );
  });

  it('says precision is not reported rather than asserting one', () => {
    render(
      <TrustPanel
        device={statusFixture.device}
        pipeline={statusFixture.pipeline}
        grounding={grounding}
        coordinateSpace="image_pixels"
      />,
    );
    expect(screen.getByTestId('trust-precision')).toHaveTextContent('not reported');
  });

  it('always carries the ordinal-score disclaimer', () => {
    render(
      <TrustPanel device={null} pipeline={null} grounding={null} coordinateSpace={null} />,
    );
    expect(screen.getByTestId('score-disclaimer')).toHaveTextContent(
      'not a probability',
    );
  });

  it('degrades to em dashes when metadata is missing', () => {
    render(
      <TrustPanel device={null} pipeline={null} grounding={null} coordinateSpace={null} />,
    );
    expect(screen.getByTestId('trust-device')).toHaveTextContent('—');
    expect(screen.getByTestId('trust-model')).toHaveTextContent('—');
    expect(screen.getByTestId('trust-grounding')).toHaveTextContent('not run');
  });
});

// ---------------------------------------------------------------------------
describe('AI narrative', () => {
  it('shows the explanation and names what produced it', () => {
    render(<NarrativePanel incident={incidentFixture} />);
    expect(screen.getByTestId('ai-summary')).toHaveTextContent(
      incidentFixture.explanation!.summary.slice(0, 40),
    );
    const provenance = screen.getByTestId('ai-provenance');
    expect(provenance).toHaveTextContent(incidentFixture.explanation!.provider);
    expect(provenance).toHaveTextContent('not a language model');
    expect(provenance).toHaveTextContent('PASS');
  });

  it('shows grounding violations when the checker found some', () => {
    const failing = clone(incidentFixture) as IncidentResponse;
    failing.grounding = {
      incident_id: 'INC-1',
      ok: false,
      violations: [
        { code: 'invented_location', detail: 'names a location', excerpt: 'warehouse' },
      ],
      notes: [],
    };
    render(<NarrativePanel incident={failing} />);
    expect(screen.getByTestId('grounding-violations')).toHaveTextContent(
      'invented_location',
    );
  });

  it('renders nothing when no explanation was requested', () => {
    const noExplanation = clone(incidentFixture) as IncidentResponse;
    noExplanation.explanation = null;
    const { container } = render(<NarrativePanel incident={noExplanation} />);
    expect(container).toBeEmptyDOMElement();
  });
});

// ---------------------------------------------------------------------------
describe('risk fixture sanity', () => {
  it('is the real API payload, carrying its own disclaimer', () => {
    const response: RiskResponse = riskFixture;
    expect(response.score_interpretation).toMatch(/NOT a calibrated probability/);
    expect(response.report_count).toBe(response.timeline.length);
  });
});

// ---------------------------------------------------------------------------
describe('video stage degradation', () => {
  it('keeps the dashboard usable when the browser cannot decode the file', async () => {
    const frames = timelineFixture.frames;
    render(
      <VideoIntelligence
        videoSrc="http://example.test/video.mp4"
        video={statusFixture.video}
        frames={frames}
        currentTime={frames[0].timestamp}
        onTimeChange={() => {}}
        seekTo={null}
        involved={[]}
      />,
    );

    const video = screen.getByTestId('video');
    fireEvent.error(video);

    expect(await screen.findByTestId('video-unplayable')).toHaveTextContent(
      /cannot decode this video file/i,
    );
    // The analysis is still there: the overlay and the transport remain.
    expect(screen.getByTestId('track-overlay')).toBeInTheDocument();
    expect(screen.getByLabelText('Scrub analysis timeline')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /play video/i })).toBeDisabled();
  });

  it('shows the frame index and track tags for the current frame', () => {
    const frames = timelineFixture.frames;
    const withTracks = frames.find((f) => f.tracks.length > 0)!;
    render(
      <VideoIntelligence
        videoSrc="http://example.test/video.mp4"
        video={statusFixture.video}
        frames={frames}
        currentTime={withTracks.timestamp}
        onTimeChange={() => {}}
        seekTo={null}
        involved={[]}
      />,
    );
    expect(
      screen.getByText(`frame ${withTracks.frame_index}`, { exact: false }),
    ).toBeInTheDocument();
    // The label appears twice by design: on the box and in the track-tag list
    // under the transport. Assert on the list, which is the readable one.
    const tags = document.querySelector('.tracklist')!;
    for (const track of withTracks.tracks) {
      expect(tags).toHaveTextContent(`#${track.track_id} ${track.class_name}`);
    }
  });
});
