import { describe, expect, it } from 'vitest';

import type {
  EventPayload,
  LifecyclePoint,
  RiskReportPayload,
  TimelineFramePayload,
} from '../api/types';
import {
  assessmentAt,
  buildMarkers,
  historyFor,
  liveAssessmentAt,
  sameSituation,
  clipDuration,
  frameAt,
  lifecycleAt,
  lifecycleTransitions,
  peakRisk,
  riskAt,
} from '../lib/timeline';
import {
  incident as incidentFixture,
  risk as riskFixture,
  timeline as timelineFixture,
} from '../test/fixtures';

function report(timestamp: number, score: number, severity: string): RiskReportPayload {
  return {
    timestamp,
    coordinate_space: 'image_pixels',
    score_interpretation: '0-100 ordinal risk score; NOT a calibrated probability',
    max_risk_score: score,
    severity: severity as RiskReportPayload['severity'],
    assessment_count: score > 0 ? 1 : 0,
    assessments: [],
    metadata: {},
  };
}

describe('frameAt', () => {
  it('finds the frame nearest the playhead', () => {
    const frames = timelineFixture.frames;
    const target = frames[5];
    expect(frameAt(frames, target.timestamp)?.frame_index).toBe(target.frame_index);
    expect(frameAt(frames, target.timestamp + 0.01)?.frame_index).toBe(target.frame_index);
  });

  it('returns null when there is no frame data at all', () => {
    expect(frameAt([], 1.0)).toBeNull();
  });

  it('never extrapolates past the clip', () => {
    const frames = timelineFixture.frames as TimelineFramePayload[];
    const last = frames[frames.length - 1];
    expect(frameAt(frames, 9999)?.frame_index).toBe(last.frame_index);
  });
});

describe('riskAt', () => {
  const reports = [report(0, 0, 'normal'), report(0.4, 40, 'medium'), report(0.8, 90, 'critical')];

  it('is a step function over what the engine actually assessed', () => {
    expect(riskAt(reports, 0.4)?.max_risk_score).toBe(40);
    expect(riskAt(reports, 0.79)?.max_risk_score).toBe(40);
    expect(riskAt(reports, 0.8)?.max_risk_score).toBe(90);
  });

  it('shows nothing before the first assessment', () => {
    expect(riskAt([report(1.0, 50, 'medium')], 0.5)).toBeNull();
  });

  it('never interpolates a score between assessments', () => {
    const between = riskAt(reports, 0.6);
    expect(between?.max_risk_score).toBe(40);
  });
});

describe('peakRisk', () => {
  it('finds the worst moment in the real fixture', () => {
    const peak = peakRisk(riskFixture.timeline);
    expect(peak?.max_risk_score).toBe(riskFixture.max_risk_score);
  });

  it('ignores reports with no assessment', () => {
    expect(peakRisk([report(0, 0, 'normal')])).toBeNull();
  });
});

describe('buildMarkers', () => {
  const events: EventPayload[] = [
    {
      timestamp: 0.5,
      entity_id: 'person_1',
      action: 'appeared',
      coordinate_space: 'image_pixels',
      attributes: {},
      zones: [],
      event_id: 'e1',
    },
    {
      timestamp: 0.1,
      entity_id: 'truck_2',
      action: 'entered_zone',
      coordinate_space: 'image_pixels',
      attributes: {},
      zones: ['bay'],
      event_id: 'e2',
    },
  ];

  it('sorts events and severity changes into one chronological track', () => {
    const markers = buildMarkers(events, [
      report(0, 0, 'normal'),
      report(0.3, 70, 'high'),
    ]);
    expect(markers.map((m) => m.timestamp)).toEqual([0.1, 0.3, 0.5]);
    expect(markers[1].kind).toBe('risk');
    expect(markers[1].severity).toBe('high');
  });

  it('marks a severity change only when it actually changes', () => {
    const markers = buildMarkers([], [
      report(0, 0, 'normal'),
      report(0.2, 0, 'normal'),
      report(0.4, 70, 'high'),
      report(0.6, 70, 'high'),
    ]);
    expect(markers.filter((m) => m.kind === 'risk')).toHaveLength(1);
  });

  it('carries the zone into the marker detail when there is one', () => {
    const markers = buildMarkers(events, []);
    const zoned = markers.find((m) => m.entityId === 'truck_2');
    expect(zoned?.detail).toContain('bay');
  });
});

describe('clipDuration', () => {
  it('prefers the video duration the API reported', () => {
    expect(clipDuration(3.0, timelineFixture.frames)).toBe(3.0);
  });

  it('falls back to the last frame when the duration is unknown', () => {
    const frames = timelineFixture.frames;
    expect(clipDuration(null, frames)).toBe(frames[frames.length - 1].timestamp);
  });

  it('is zero when there is nothing at all', () => {
    expect(clipDuration(null, [])).toBe(0);
  });
});

describe('lifecycleAt', () => {
  const history = incidentFixture.lifecycle_history;

  it('is null before the situation was first assessed', () => {
    expect(lifecycleAt(history, history[0].timestamp - 0.01)).toBeNull();
  });

  it('holds the state the backend derived, without interpolating', () => {
    const first = history[0];
    expect(lifecycleAt(history, first.timestamp)?.state).toBe(first.state);
    expect(lifecycleAt(history, first.timestamp + 0.01)?.state).toBe(first.state);
  });

  it('reaches current only once the clip actually does', () => {
    const current = history.find((point) => point.state === 'current')!;
    const before = history.filter((point) => point.timestamp < current.timestamp);
    for (const point of before) {
      expect(lifecycleAt(history, point.timestamp)?.state).not.toBe('current');
    }
    expect(lifecycleAt(history, current.timestamp)?.state).toBe('current');
  });

  it('is null for an empty history', () => {
    expect(lifecycleAt([], 5)).toBeNull();
  });
});

describe('assessmentAt', () => {
  const reports = riskFixture.timeline;
  const pair = incidentFixture.evidence!.entities.map((e) => e.entity_id);

  it('returns this situation own assessment at the playhead', () => {
    const worst = incidentFixture.evidence!.timestamp;
    const found = assessmentAt(reports, worst, pair);
    expect(found?.timestamp).toBe(worst);
    expect(found?.risk_score).toBe(incidentFixture.evidence!.risk_score);
  });

  it('stays inside the report the playhead is standing in', () => {
    // Within one step the selected report has not changed yet, so the same
    // assessment is still the right answer.
    const worst = incidentFixture.evidence!.timestamp;
    expect(assessmentAt(reports, worst + 0.05, pair)?.timestamp).toBe(worst);
  });

  it('never holds an assessment forward into a report that has none', () => {
    // The defect this replaced: 47 seconds after the last assessment the
    // incident panel still showed one, next to a risk panel saying there was
    // none. A report with no assessments is an answer, not a gap to fill.
    const empty = reports.filter((report) => report.assessments.length === 0);
    expect(empty.length).toBeGreaterThan(0);
    for (const report of empty) {
      expect(assessmentAt(reports, report.timestamp, pair)).toBeNull();
    }
  });

  it('agrees with riskAt about whether this moment was assessed at all', () => {
    // Both panels read the same report, so they cannot disagree about silence.
    for (const report of reports) {
      const selected = riskAt(reports, report.timestamp);
      const riskPanelHasNothing = (selected?.assessments.length ?? 0) === 0;
      if (riskPanelHasNothing) {
        expect(assessmentAt(reports, report.timestamp, pair)).toBeNull();
      }
    }
  });

  it('is null once the playhead passes the clip last assessment', () => {
    const last = reports[reports.length - 1];
    if (last.assessments.length === 0) {
      expect(assessmentAt(reports, last.timestamp + 30, pair)).toBeNull();
    }
  });

  it('does not care about entity order', () => {
    const t = incidentFixture.evidence!.timestamp;
    expect(assessmentAt(reports, t, [...pair].reverse())?.risk_score).toBe(
      assessmentAt(reports, t, pair)?.risk_score,
    );
  });

  it('is null for a situation the clip never contained', () => {
    const t = incidentFixture.evidence!.timestamp;
    expect(assessmentAt(reports, t, ['person_99', 'ghost_1'])).toBeNull();
    expect(assessmentAt(reports, t, [])).toBeNull();
  });

  it('is null before the first assessment of that situation', () => {
    const first = reports.find((r) => r.assessments.length > 0)!;
    expect(assessmentAt(reports, first.timestamp - 0.01, pair)).toBeNull();
  });
});

describe('liveAssessmentAt', () => {
  const reports = riskFixture.timeline;

  it('is exactly what the risk panel reads, at every step of the clip', () => {
    // The agreement property. The risk panel renders
    // `riskAt(reports, t).assessments[0]`; the incident panel renders this.
    // If they are ever different objects the two can disagree about whether
    // this moment was assessed, which is the defect this replaced.
    for (const report of reports) {
      for (const t of [report.timestamp, report.timestamp + 0.01]) {
        expect(liveAssessmentAt(reports, t)).toBe(
          riskAt(reports, t)?.assessments[0] ?? null,
        );
      }
    }
  });

  it('is null where the engine assessed nothing, never a held-over value', () => {
    const empty = reports.filter((r) => r.assessments.length === 0);
    expect(empty.length).toBeGreaterThan(0);
    for (const report of empty) {
      expect(liveAssessmentAt(reports, report.timestamp)).toBeNull();
    }
  });

  it('is null before the first report', () => {
    expect(liveAssessmentAt(reports, reports[0].timestamp - 1)).toBeNull();
    expect(liveAssessmentAt([], 5)).toBeNull();
  });

  it('follows whichever situation the engine ranked first', () => {
    const reportWithTwo: RiskReportPayload = {
      ...report(1.0, 40, 'medium'),
      assessment_count: 2,
      assessments: [
        { ...riskFixture.worst!.assessments[0], risk_score: 40, involved_entity_ids: ['a', 'b'] },
        { ...riskFixture.worst!.assessments[0], risk_score: 10, involved_entity_ids: ['c', 'd'] },
      ],
    };
    expect(liveAssessmentAt([reportWithTwo], 1.0)?.involved_entity_ids).toEqual(['a', 'b']);
  });
});

describe('sameSituation', () => {
  it('ignores order', () => {
    expect(sameSituation(['person_1', 'truck_2'], ['truck_2', 'person_1'])).toBe(true);
  });

  it('is false for a different pair, and for an empty one', () => {
    expect(sameSituation(['person_1', 'truck_2'], ['person_1', 'car_9'])).toBe(false);
    expect(sameSituation([], ['person_1'])).toBe(false);
    expect(sameSituation(['person_1'], [])).toBe(false);
  });
});

describe('historyFor', () => {
  const situations = incidentFixture.situations;

  it('returns that situation own lifecycle, whatever the entity order', () => {
    const first = situations[0];
    expect(historyFor(situations, first.entity_ids)).toBe(first.history);
    expect(historyFor(situations, [...first.entity_ids].reverse())).toBe(first.history);
  });

  it('borrows nothing for a situation the clip never contained', () => {
    expect(historyFor(situations, ['ghost_1', 'ghost_2'])).toEqual([]);
    expect(historyFor(situations, [])).toEqual([]);
    expect(historyFor([], ['person_1', 'truck_2'])).toEqual([]);
  });

  it('agrees with the lifecycle_history the response also carries', () => {
    const worst = incidentFixture.evidence!.entities.map((e) => e.entity_id);
    expect(historyFor(situations, worst)).toEqual(incidentFixture.lifecycle_history);
  });
});

describe('lifecycleTransitions', () => {
  // One pair that goes imminent, quietens, then escalates again — the sequence
  // app/api/routes/incident.py :: situation_lifecycles produces for it.
  const reEscalation = [
    { timestamp: 10.0, state: 'imminent', risk_score: 40.0 },
    { timestamp: 10.5, state: 'imminent', risk_score: 41.0 },
    { timestamp: 12.0, state: 'resolved', risk_score: 2.0 },
    { timestamp: 12.5, state: 'observed', risk_score: 2.0 },
    { timestamp: 14.0, state: 'imminent', risk_score: 44.0 },
    { timestamp: 16.0, state: 'current', risk_score: 70.0 },
  ] as LifecyclePoint[];

  it('keeps every genuine change of state, in the order it happened', () => {
    expect(lifecycleTransitions(reEscalation).map((p) => [p.state, p.timestamp])).toEqual([
      ['imminent', 10.0],
      ['resolved', 12.0],
      ['observed', 12.5],
      ['imminent', 14.0],
      ['current', 16.0],
    ]);
  });

  it('keeps a state that is entered twice, as two entries', () => {
    const imminent = lifecycleTransitions(reEscalation).filter((p) => p.state === 'imminent');
    expect(imminent.map((p) => p.timestamp)).toEqual([10.0, 14.0]);
  });

  it('drops a repeat only when it is consecutive', () => {
    // 10.5 s is the same state as 10.0 s, so it is not a transition.
    expect(lifecycleTransitions(reEscalation)).toHaveLength(5);
    expect(lifecycleTransitions(reEscalation).some((p) => p.timestamp === 10.5)).toBe(false);
  });

  it('returns the backend points themselves, with no timestamp touched', () => {
    const transitions = lifecycleTransitions(reEscalation);
    expect(transitions[0]).toBe(reEscalation[0]);
    expect(transitions[3]).toBe(reEscalation[4]);
    for (const point of transitions) {
      expect(reEscalation).toContain(point);
    }
  });

  it('is ordered by time, which a per-state collapse could not promise', () => {
    // The defect: first-occurrence-only put `resolved` (12.0 s) above a second
    // `imminent` (14.0 s) because the rungs were in state order, not clip order.
    const stamps = lifecycleTransitions(reEscalation).map((p) => p.timestamp);
    expect(stamps).toEqual([...stamps].sort((a, b) => a - b));
  });

  it('has nothing to show for an empty history', () => {
    expect(lifecycleTransitions([])).toEqual([]);
  });

  it('collapses a situation that never changed state to one entry', () => {
    const steady = [
      { timestamp: 1.0, state: 'observed', risk_score: 5 },
      { timestamp: 1.5, state: 'observed', risk_score: 6 },
      { timestamp: 2.0, state: 'observed', risk_score: 7 },
    ] as LifecyclePoint[];
    expect(lifecycleTransitions(steady)).toHaveLength(1);
    expect(lifecycleTransitions(steady)[0].timestamp).toBe(1.0);
  });

  it('agrees with the captured clip history', () => {
    const transitions = lifecycleTransitions(incidentFixture.lifecycle_history);
    expect(transitions.length).toBeLessThanOrEqual(
      incidentFixture.lifecycle_history.length,
    );
    for (let i = 1; i < transitions.length; i += 1) {
      expect(transitions[i].state).not.toBe(transitions[i - 1].state);
      expect(transitions[i].timestamp).toBeGreaterThan(transitions[i - 1].timestamp);
    }
  });
});
