import { describe, expect, it } from 'vitest';

import type { EventPayload, RiskReportPayload, TimelineFramePayload } from '../api/types';
import {
  assessmentAt,
  buildMarkers,
  clipDuration,
  frameAt,
  lifecycleAt,
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

  it('holds the last assessment produced, like every other lookup here', () => {
    const worst = incidentFixture.evidence!.timestamp;
    expect(assessmentAt(reports, worst + 0.05, pair)?.timestamp).toBe(worst);
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
