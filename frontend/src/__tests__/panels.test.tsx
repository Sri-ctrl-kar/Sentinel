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

import type { IncidentResponse, LifecyclePoint, RiskResponse } from '../api/types';
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

  // -------------------------------------------------------------------------
  // Which threshold the panel quotes, and how it squares an unsafe present
  // with a low band.
  describe('unsafe separation wording', () => {
    /** An assessment unsafe now at 22.06px, against a 90px threshold. */
    function unsafeNow(score = 30.0) {
      const report = clone(peak);
      const assessment = report.assessments[0];
      assessment.risk_score = score;
      assessment.severity = 'low';
      assessment.time_to_risk = {
        status: 'already_unsafe',
        seconds: 0,
        coordinate_space: 'image_pixels',
        units: 'pixels',
        reason: null,
        unsafe_separation_threshold_px: 90.0,
      };
      assessment.details = {
        ...assessment.details,
        // critical_radius_px is the proximity saturation radius, NOT the
        // unsafe threshold. It is present so the test can prove it is not read.
        thresholds: { critical_radius_px: 70.0, unsafe_separation_px: 90.0 },
        prediction: { current_separation_px: 22.06, unsafe_separation_threshold_px: 90.0 },
      };
      return report;
    }

    it('quotes the unsafe separation threshold, not the proximity radius', () => {
      render(<RiskPanel risk={riskFixture} current={unsafeNow()} currentTime={19.36} />);
      const note = screen.getByTestId('ttr-note');
      expect(note).toHaveTextContent('90 px');
      expect(note).not.toHaveTextContent('70 px');
      expect(note.textContent).not.toMatch(/\b70\b/);
    });

    it('states the current separation against that threshold', () => {
      render(<RiskPanel risk={riskFixture} current={unsafeNow()} currentTime={19.36} />);
      const note = screen.getByTestId('ttr-note');
      expect(note).toHaveTextContent('22 px');
      expect(note).toHaveTextContent(
        /at or inside the unsafe separation threshold \(90 px\)/i,
      );
      expect(note).toHaveTextContent(/present condition, not a prediction/i);
    });

    it('keeps score, band and present-unsafe status as separate statements', () => {
      // The reconciliation: a LOW band beside UNSAFE NOW is the engine's
      // design, and the panel says which question each answers.
      render(<RiskPanel risk={riskFixture} current={unsafeNow()} currentTime={19.36} />);
      expect(screen.getByTestId('risk-score')).toHaveTextContent('30.0');
      expect(screen.getByText(/engine severity band: LOW/i)).toBeInTheDocument();
      expect(screen.getByTestId('ttr-value')).toHaveTextContent('UNSAFE NOW');
      expect(screen.getByTestId('ttr-note')).toHaveTextContent(
        /separate question from how close the pair is right now/i,
      );
    });

    it('names the same threshold on a predicted crossing', () => {
      const report = unsafeNow();
      report.assessments[0].time_to_risk = {
        status: 'predicted',
        seconds: 1.4,
        coordinate_space: 'image_pixels',
        units: 'pixels',
        reason: null,
        unsafe_separation_threshold_px: 90.0,
      };
      render(<RiskPanel risk={riskFixture} current={report} currentTime={5} />);
      expect(screen.getByTestId('ttr-note')).toHaveTextContent('90 px');
      expect(screen.getByTestId('ttr-note')).toHaveTextContent('22 px');
    });

    it('invents no threshold when the payload carries none', () => {
      const report = unsafeNow();
      report.assessments[0].time_to_risk = {
        status: 'already_unsafe',
        seconds: 0,
        coordinate_space: 'image_pixels',
        units: 'pixels',
        reason: null,
      };
      const note = render(
        <RiskPanel risk={riskFixture} current={report} currentTime={19.36} />,
      ).getByTestId('ttr-note');
      expect(note).toHaveTextContent(/unsafe separation threshold is crossed/i);
      expect(note.textContent).not.toMatch(/\d+\s?px/);
    });

    it('reads a calibrated threshold in metres', () => {
      const report = unsafeNow();
      report.assessments[0].time_to_risk = {
        status: 'already_unsafe',
        seconds: 0,
        coordinate_space: 'ground_plane_meters',
        units: 'meters',
        reason: null,
        unsafe_separation_threshold_m: 2.0,
      };
      report.assessments[0].details = {
        prediction: { current_separation_m: 1.42 },
      };
      render(<RiskPanel risk={riskFixture} current={report} currentTime={19.36} />);
      const note = screen.getByTestId('ttr-note');
      expect(note).toHaveTextContent('2.00 m');
      expect(note).toHaveTextContent('1.42 m');
    });
  });
});

// ---------------------------------------------------------------------------
describe('prediction panel', () => {
  // The assessment the engine produced at the clip's worst moment, and one from
  // earlier in the same clip — both straight out of the captured risk timeline.
  const worstAssessment = peak.assessments[0];
  const earlier = riskFixture.timeline.find(
    (report) =>
      report.assessments.length > 0 &&
      report.assessments[0].time_to_risk?.status === 'predicted',
  )!.assessments[0];

  it('marks a present breach as CURRENT, never as predicted', () => {
    render(
      <PredictionPanel
        incident={incidentFixture}
        live={worstAssessment}
        liveState="current"
      />,
    );
    const badge = screen.getByTestId('tense-badge');
    expect(badge).toHaveTextContent('CURRENT');
    expect(badge).toHaveTextContent(/happening now/i);
    expect(badge).not.toHaveTextContent('PREDICTED');
  });

  it('marks a forward-looking conflict as PREDICTED and says it has not happened', () => {
    const predicted = clone(earlier);
    predicted.time_to_risk = {
      status: 'predicted',
      seconds: 1.4,
      coordinate_space: 'image_pixels',
      units: 'pixels',
      reason: null,
    };
    render(
      <PredictionPanel
        incident={incidentFixture}
        live={predicted}
        liveState="imminent"
      />,
    );
    const badge = screen.getByTestId('tense-badge');
    expect(badge).toHaveTextContent('PREDICTED');
    expect(badge).toHaveTextContent(/has not happened/i);
    expect(screen.getByTestId('prediction-ttr')).toHaveTextContent('1.40 s');
  });

  it('follows the playhead instead of freezing on the worst moment', () => {
    // The defect this replaced: every seek showed the worst moment's score and
    // CURRENT, however early in the clip the operator was looking.
    const { rerender } = render(
      <PredictionPanel
        incident={incidentFixture}
        live={earlier}
        liveState="imminent"
      />,
    );
    expect(screen.getByTestId('prediction-score')).toHaveTextContent(
      earlier.risk_score.toFixed(1),
    );
    expect(screen.getByTestId('lifecycle-value')).toHaveTextContent('IMMINENT');
    expect(screen.getByTestId('tense-badge')).not.toHaveTextContent('happening now');

    rerender(
      <PredictionPanel
        incident={incidentFixture}
        live={worstAssessment}
        liveState="current"
      />,
    );
    expect(screen.getByTestId('prediction-score')).toHaveTextContent(
      worstAssessment.risk_score.toFixed(1),
    );
    expect(screen.getByTestId('tense-badge')).toHaveTextContent(/happening now/i);
  });

  it('names the situation on screen and says how it relates to the worst moment', () => {
    // A long clip holds many situations. The panel must say which one it is
    // showing, and whether that is the one the material below describes.
    const other = clone(earlier);
    other.involved_entity_ids = ['person_9', 'bus_4'];
    other.risk_score = 21.5;

    const { rerender } = render(
      <PredictionPanel
        incident={incidentFixture}
        live={other}
        liveState="developing"
        isWorstSituation={false}
      />,
    );
    expect(screen.getByTestId('live-situation')).toHaveTextContent('person_9 + bus_4');
    expect(screen.getByTestId('situation-scope')).toHaveTextContent(/different situation/i);
    expect(screen.getByTestId('prediction-score')).toHaveTextContent('21.5');
    // The worst moment is still on screen, still labelled, still its own.
    const divider = screen.getByTestId('worst-moment-divider');
    expect(divider).toHaveTextContent(incidentFixture.evidence!.incident_id);
    expect(divider).toHaveTextContent(
      incidentFixture.evidence!.risk_score.toFixed(1),
    );

    rerender(
      <PredictionPanel
        incident={incidentFixture}
        live={worstAssessment}
        liveState="current"
        isWorstSituation
      />,
    );
    expect(screen.getByTestId('situation-scope')).toHaveTextContent(
      /this is the clip's worst situation/i,
    );
  });

  it('says so when the situation was not assessed at this moment', () => {
    render(<PredictionPanel incident={incidentFixture} live={null} liveState={null} />);
    expect(screen.getByTestId('tense-badge')).toHaveTextContent(/not assessed/i);
    // Never the worst moment's numbers standing in for an unassessed moment.
    expect(screen.getByTestId('prediction-score')).toHaveTextContent('—');
    expect(screen.getByTestId('lifecycle-value')).toHaveTextContent('—');
    expect(screen.getByTestId('live-situation')).toHaveTextContent(/no situation/i);
    expect(screen.queryByTestId('situation-scope')).toBeNull();
  });

  it('names the moment the worst-moment material belongs to', () => {
    render(
      <PredictionPanel
        incident={incidentFixture}
        live={earlier}
        liveState="imminent"
      />,
    );
    const divider = screen.getByTestId('worst-moment-divider');
    expect(divider).toHaveTextContent(
      incidentFixture.evidence!.timestamp.toFixed(2),
    );
    expect(divider).toHaveTextContent(
      incidentFixture.evidence!.risk_score.toFixed(1),
    );
  });

  it('shows the incident type and the involved entities from the API', () => {
    render(
      <PredictionPanel
        incident={incidentFixture}
        live={worstAssessment}
        liveState="current"
      />,
    );
    expect(screen.getByTestId('incident-type')).toHaveTextContent(
      incidentFixture.evidence!.incident_type.replace(/_/g, ' '),
    );
    expect(screen.getByTestId('lifecycle-value')).toHaveTextContent('CURRENT');
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
      situations: [],
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

  it('survives an assessment with no prediction outcome', () => {
    const noOutcome = clone(peak.assessments[0]);
    noOutcome.prediction_outcome = null;
    noOutcome.time_to_risk = null;
    render(
      <PredictionPanel incident={incidentFixture} live={noOutcome} liveState="observed" />,
    );
    expect(screen.getByTestId('incident-type')).toBeInTheDocument();
    expect(screen.getByTestId('prediction-ttr')).toHaveTextContent('unavailable');
  });

  // -------------------------------------------------------------------------
  // The operator recommendation, gated on the engine's own prediction outcome.
  // `recommended_intervention` is looked up from the incident type alone, so it
  // reads the same whether a conflict is predicted or not. The panel must not
  // prescribe an action the prediction does not support.
  describe('operator recommendation', () => {
    const HOLD = 'alert both parties; hold the vehicle until paths diverge';

    /** The assessment an operator met at 5.50 s of the 62.88 s clip. */
    function nearMiss() {
      const assessment = clone(peak.assessments[0]);
      assessment.incident_type = 'CONVERGING_TRAJECTORIES';
      assessment.risk_score = 40.1;
      assessment.severity = 'medium';
      assessment.prediction_outcome = 'NO_PREDICTED_CONFLICT';
      assessment.recommended_intervention = HOLD;
      assessment.time_to_risk = {
        status: 'not_predicted',
        seconds: null,
        coordinate_space: 'image_pixels',
        units: 'pixels',
        reason: 'threshold not reached within the prediction horizon',
      };
      return assessment;
    }

    it.each([
      'PREDICTED_UNSAFE_PROXIMITY',
      'PREDICTED_TRAJECTORY_CONFLICT',
      'CURRENTLY_UNSAFE_PROXIMITY',
    ])('shows the engine intervention for %s', (outcome) => {
      const unsafe = nearMiss();
      unsafe.prediction_outcome = outcome;
      render(
        <PredictionPanel incident={incidentFixture} live={unsafe} liveState="current" />,
      );
      expect(screen.getByTestId('prediction-recommendation')).toHaveTextContent(HOLD);
    });

    it('prescribes monitoring, not a vehicle hold, for NO_PREDICTED_CONFLICT', () => {
      render(
        <PredictionPanel
          incident={incidentFixture}
          live={nearMiss()}
          liveState="observed"
        />,
      );
      const recommendation = screen.getByTestId('prediction-recommendation');
      expect(recommendation).toHaveTextContent('Monitor — no conflict predicted.');
      expect(recommendation).not.toHaveTextContent(/hold the vehicle/i);
      expect(screen.queryByText(HOLD)).toBeNull();
    });

    it('says no prediction was made rather than claiming the paths are clear', () => {
      const unavailable = nearMiss();
      unavailable.prediction_outcome = 'PREDICTION_UNAVAILABLE';
      render(
        <PredictionPanel
          incident={incidentFixture}
          live={unavailable}
          liveState="observed"
        />,
      );
      const recommendation = screen.getByTestId('prediction-recommendation');
      expect(recommendation).toHaveTextContent(/no prediction available/i);
      expect(recommendation).not.toHaveTextContent(/hold the vehicle/i);
      expect(recommendation).not.toHaveTextContent(/no conflict predicted/i);
    });

    it('leaves the score, severity, lifecycle and outcome exactly as reported', () => {
      // Gating the recommendation changes nothing else on the panel.
      render(
        <PredictionPanel
          incident={incidentFixture}
          live={nearMiss()}
          liveState="observed"
        />,
      );
      expect(screen.getByTestId('prediction-score')).toHaveTextContent('40.1 /100');
      expect(screen.getByTestId('lifecycle-value')).toHaveTextContent('OBSERVED');
      expect(screen.getByTestId('prediction-ttr')).toHaveTextContent('unavailable');
      const outcome = screen.getByTestId('prediction-outcome');
      expect(outcome).toHaveTextContent(/no predicted conflict/i);
      expect(outcome).toHaveTextContent(/threshold not reached/i);
    });
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

  // -------------------------------------------------------------------------
  // Where these numbers are from. The panel is worst-moment scoped and does not
  // follow the playhead, so a reader must not take its measurements for the
  // selected second.
  describe('measurement provenance', () => {
    const worstTime = incidentFixture.evidence!.timestamp;

    it('timestamps the measurements with their own source moment', () => {
      render(<EvidencePanel incident={incidentFixture} />);
      expect(screen.getByTestId('measures-caption')).toHaveTextContent(
        worstTime.toFixed(2),
      );
    });

    it('says the playhead is elsewhere when it is', () => {
      const elsewhere = clone(peak.assessments[0]);
      elsewhere.timestamp = worstTime - 12.15;
      render(<EvidencePanel incident={incidentFixture} live={elsewhere} />);
      const note = screen.getByTestId('evidence-provenance');
      expect(note).toHaveTextContent(worstTime.toFixed(2));
      expect(note).toHaveTextContent(elsewhere.timestamp.toFixed(2));
      expect(note).toHaveTextContent(/not that moment's figures/i);
      expect(screen.getByTestId('measures-caption')).toHaveTextContent(
        elsewhere.timestamp.toFixed(2),
      );
    });

    it('does not warn when the playhead is standing in the worst moment', () => {
      const here = clone(peak.assessments[0]);
      here.timestamp = worstTime;
      render(<EvidencePanel incident={incidentFixture} live={here} />);
      const note = screen.getByTestId('evidence-provenance');
      expect(note).toHaveTextContent(worstTime.toFixed(2));
      expect(note).not.toHaveTextContent(/not that moment's figures/i);
    });

    it('still names the source moment when nothing is live', () => {
      render(<EvidencePanel incident={incidentFixture} live={null} />);
      expect(screen.getByTestId('evidence-provenance')).toHaveTextContent(
        worstTime.toFixed(2),
      );
      expect(screen.getByTestId('evidence-provenance')).not.toHaveTextContent(
        /selected above/i,
      );
    });

    it('reads the timestamp from the evidence, never from the playhead', () => {
      // The numbers belong to `evidence.timestamp`; a different `live` must not
      // relabel them as that moment's.
      const elsewhere = clone(peak.assessments[0]);
      elsewhere.timestamp = 4.9;
      render(<EvidencePanel incident={incidentFixture} live={elsewhere} />);
      const measures = screen.getByTestId('evidence-measures');
      for (const quantity of incidentFixture.quantities) {
        expect(measures).toHaveTextContent(quantity.name);
      }
      expect(screen.getByTestId('measures-caption')).toHaveTextContent(
        `Measured at ${worstTime.toFixed(2)} s`,
      );
    });
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

  it('shows the states the clip passed through before its worst moment', () => {
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

  it('omits a state the situation never entered', () => {
    // This clip never developed: every prediction landed inside the 2s horizon,
    // so DEVELOPING has no entry. The trail is the transitions that happened,
    // so an unvisited state is absent rather than drawn as an unlit rung.
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

  // -------------------------------------------------------------------------
  // A situation that resolves and escalates again.
  describe('a re-escalating situation', () => {
    // The sequence app/api/routes/incident.py produces for such a pair.
    const history = [
      { timestamp: 10.0, state: 'imminent', risk_score: 40.0 },
      { timestamp: 10.5, state: 'imminent', risk_score: 41.0 },
      { timestamp: 12.0, state: 'resolved', risk_score: 2.0 },
      { timestamp: 12.5, state: 'observed', risk_score: 2.0 },
      { timestamp: 14.0, state: 'imminent', risk_score: 44.0 },
      { timestamp: 16.0, state: 'current', risk_score: 70.0 },
    ] as LifecyclePoint[];

    function segments() {
      return screen.getAllByTestId('lifecycle-segment').map((el) => ({
        state: el.getAttribute('data-state'),
        text: el.textContent ?? '',
      }));
    }

    it('shows every transition in the order it happened', () => {
      render(<LifecycleTrail state="current" history={history} />);
      const rendered = segments();
      expect(rendered.map((s) => s.state)).toEqual([
        'imminent', 'resolved', 'observed', 'imminent', 'current',
      ]);
      expect(rendered[0].text).toContain('10.00 s');
      expect(rendered[1].text).toContain('12.00 s');
      expect(rendered[2].text).toContain('12.50 s');
      expect(rendered[3].text).toContain('14.00 s');
      expect(rendered[4].text).toContain('16.00 s');
    });

    it('never prints a later state above an earlier one', () => {
      // The defect: one rung per state, in state order, left RESOLVED (first
      // seen 12.00 s) sitting above IMMINENT (first seen 10.00 s) — and the
      // second IMMINENT, at 14.00 s, had nowhere to appear at all.
      render(<LifecycleTrail state="current" history={history} />);
      const times = segments().map(
        (s) => Number(/from (\d+\.\d+) s/.exec(s.text)![1]),
      );
      expect(times).toEqual([...times].sort((a, b) => a - b));
    });

    it('does not repeat a state that simply held', () => {
      render(<LifecycleTrail state="current" history={history} />);
      // 10.5 s and 10.0 s are both imminent; only the entry is a transition.
      expect(segments()).toHaveLength(5);
      expect(screen.getByTestId('lifecycle-trail').textContent).not.toContain('10.50 s');
    });

    it('puts NOW on the second imminent, not the first', () => {
      render(
        <LifecycleTrail state="current" history={history} now="imminent" nowAt={14.0} />,
      );
      const marked = screen.getByTestId('lifecycle-current');
      expect(marked).toHaveTextContent('IMMINENT');
      expect(marked.textContent).toContain('14.00 s');
      expect(marked.textContent).not.toContain('10.00 s');
      expect(screen.getAllByTestId('lifecycle-current')).toHaveLength(1);
    });

    it('puts NOW on the first imminent while the playhead is in it', () => {
      render(
        <LifecycleTrail state="current" history={history} now="imminent" nowAt={10.5} />,
      );
      const marked = screen.getByTestId('lifecycle-current');
      expect(marked.textContent).toContain('10.00 s');
      expect(marked.textContent).not.toContain('14.00 s');
    });

    it('names the first entry of a repeated state, once', () => {
      render(<LifecycleTrail state="current" history={history} />);
      expect(screen.getAllByTestId('lifecycle-at-imminent')).toHaveLength(1);
      expect(screen.getByTestId('lifecycle-at-imminent')).toHaveTextContent('10.00 s');
    });

    it('marks nothing as now where the playhead has no state', () => {
      render(
        <LifecycleTrail state="current" history={history} now={null} nowAt={null} />,
      );
      expect(screen.queryByTestId('lifecycle-current')).toBeNull();
      expect(screen.getByTestId('lifecycle-unassessed')).toBeInTheDocument();
      expect(segments()).toHaveLength(5);
    });

    it('uses the backend timestamps unchanged', () => {
      render(<LifecycleTrail state="current" history={history} />);
      const text = screen.getByTestId('lifecycle-trail').textContent ?? '';
      for (const stamp of ['10.00 s', '12.00 s', '12.50 s', '14.00 s', '16.00 s']) {
        expect(text).toContain(stamp);
      }
    });
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
