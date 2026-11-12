import { describe, expect, it } from 'vitest';

import {
  LIFECYCLE_ORDER,
  MISSING,
  UNSAFE_PREDICTION_OUTCOMES,
  formatClock,
  formatQuantity,
  formatScore,
  formatSeconds,
  currentSeparation,
  modelLabel,
  recommendationFor,
  riskState,
  tenseOf,
  titleise,
  unitSuffixed,
  unsafeThreshold,
} from '../lib/format';

describe('risk state mapping', () => {
  it('maps the engine severity bands onto the four display states', () => {
    expect(riskState('normal')).toBe('NORMAL');
    expect(riskState('low')).toBe('WATCH');
    expect(riskState('medium')).toBe('WATCH');
    expect(riskState('high')).toBe('HIGH');
    expect(riskState('critical')).toBe('CRITICAL');
  });

  it('treats an unknown or missing severity as NORMAL rather than guessing', () => {
    expect(riskState(null)).toBe('NORMAL');
    expect(riskState(undefined)).toBe('NORMAL');
  });
});

describe('missing values', () => {
  it('never renders absent data as zero', () => {
    expect(formatSeconds(null)).toBe(MISSING);
    expect(formatScore(null)).toBe(MISSING);
    expect(formatQuantity(null, 'px')).toBe(MISSING);
    expect(formatClock(undefined)).toBe(MISSING);
    expect(titleise(null)).toBe(MISSING);
  });

  it('formats present values with their units', () => {
    expect(formatSeconds(1.4)).toBe('1.40 s');
    expect(formatScore(95.4499)).toBe('95.4');
    expect(formatQuantity(42.04, 'px')).toBe('42.04 px');
    expect(formatQuantity(2.5, 'm/s')).toBe('2.50 m/s');
    expect(formatClock(72.5)).toBe('1:12.50');
  });
});

describe('predicted versus current', () => {
  it('reads the present tense from the engine, not from the score', () => {
    expect(tenseOf('already_unsafe', 'current')).toBe('CURRENT');
    expect(tenseOf('already_unsafe', 'observed')).toBe('CURRENT');
    expect(tenseOf(null, 'current')).toBe('CURRENT');
  });

  it('reads the future tense from a predicted crossing', () => {
    expect(tenseOf('predicted', 'imminent')).toBe('PREDICTED');
    expect(tenseOf('predicted', 'developing')).toBe('PREDICTED');
    expect(tenseOf(null, 'developing')).toBe('PREDICTED');
  });

  it('says unknown rather than choosing when nothing establishes a tense', () => {
    expect(tenseOf('not_predicted', 'observed')).toBe('UNKNOWN');
    expect(tenseOf(null, null)).toBe('UNKNOWN');
  });
});

describe('lifecycle', () => {
  it("uses the engine's own five states in order", () => {
    expect(LIFECYCLE_ORDER).toEqual([
      'observed',
      'developing',
      'imminent',
      'current',
      'resolved',
    ]);
  });
});

describe('model label', () => {
  it('reads the weights name the API reported', () => {
    expect(modelLabel('yolov8n.pt')).toBe('YOLOv8N');
    expect(modelLabel('synthetic')).toBe('synthetic');
    expect(modelLabel(null)).toBe(MISSING);
  });
});

describe('operator recommendation', () => {
  const HOLD = 'alert both parties; hold the vehicle until paths diverge';

  it('mirrors the unsafe outcomes the prediction module defines', () => {
    // app/reasoning/prediction.py :: UNSAFE_OUTCOMES
    expect([...UNSAFE_PREDICTION_OUTCOMES].sort()).toEqual([
      'CURRENTLY_UNSAFE_PROXIMITY',
      'PREDICTED_TRAJECTORY_CONFLICT',
      'PREDICTED_UNSAFE_PROXIMITY',
    ]);
  });

  it("passes the engine's intervention through where the pair is or will be unsafe", () => {
    for (const outcome of UNSAFE_PREDICTION_OUTCOMES) {
      expect(recommendationFor(outcome, HOLD)).toBe(HOLD);
    }
  });

  it('prescribes nothing where the engine predicts no conflict', () => {
    // The defect: a static intervention keyed on incident type alone told an
    // operator to hold a vehicle beside "no predicted conflict".
    expect(recommendationFor('NO_PREDICTED_CONFLICT', HOLD)).toBe(
      'Monitor — no conflict predicted.',
    );
  });

  it('neither prescribes nor denies where there is no prediction at all', () => {
    const unavailable = 'Monitor — no prediction available for this situation.';
    expect(recommendationFor('PREDICTION_UNAVAILABLE', HOLD)).toBe(unavailable);
    expect(recommendationFor(null, HOLD)).toBe(unavailable);
    expect(recommendationFor(undefined, HOLD)).toBe(unavailable);
  });

  it('has nothing to say when an unsafe outcome carries no intervention', () => {
    expect(recommendationFor('CURRENTLY_UNSAFE_PROXIMITY', '')).toBeNull();
    expect(recommendationFor('PREDICTED_TRAJECTORY_CONFLICT', null)).toBeNull();
  });
});

describe('unit-suffixed payload readers', () => {
  it('finds a pixel key and reports its unit', () => {
    expect(unitSuffixed({ current_separation_px: 22.06 }, 'current_separation')).toEqual({
      value: 22.06,
      unit: 'px',
    });
  });

  it('finds a metre key on a calibrated clip', () => {
    expect(unitSuffixed({ current_separation_m: 1.42 }, 'current_separation')).toEqual({
      value: 1.42,
      unit: 'm',
    });
  });

  it('is missing rather than zero when the key is absent', () => {
    expect(unitSuffixed({ something_else_px: 5 }, 'current_separation')).toBeNull();
    expect(unitSuffixed(null, 'current_separation')).toBeNull();
    expect(unitSuffixed(undefined, 'current_separation')).toBeNull();
  });

  it('refuses a non-numeric or non-finite value', () => {
    expect(unitSuffixed({ current_separation_px: 'close' }, 'current_separation')).toBeNull();
    expect(unitSuffixed({ current_separation_px: NaN }, 'current_separation')).toBeNull();
  });

  it('refuses a suffix that is not a unit it knows', () => {
    expect(unitSuffixed({ current_separation_furlongs: 3 }, 'current_separation')).toBeNull();
  });
});

describe('unsafe separation threshold', () => {
  it('reads the threshold the engine used for already_unsafe', () => {
    // app/reasoning/models/risk.py :: TimeToRisk.to_dict
    expect(
      unsafeThreshold({
        status: 'already_unsafe',
        seconds: 0,
        unsafe_separation_threshold_px: 90.0,
      }),
    ).toEqual({ value: 90.0, unit: 'px' });
  });

  it('never returns the proximity saturation radius instead', () => {
    // critical_radius_px (70) is where the proximity factor saturates; it is
    // not the threshold that decides "unsafe now" (unsafe_separation_px, 90).
    const bag = {
      status: 'already_unsafe',
      critical_radius_px: 70.0,
      interaction_radius_px: 260.0,
      unsafe_separation_threshold_px: 90.0,
    };
    expect(unsafeThreshold(bag)).toEqual({ value: 90.0, unit: 'px' });
    expect(unsafeThreshold(bag)!.value).not.toBe(70.0);
  });

  it('has nothing to say when the payload carries no threshold', () => {
    expect(unsafeThreshold({ status: 'already_unsafe', seconds: 0 })).toBeNull();
    expect(unsafeThreshold(null)).toBeNull();
  });
});

describe('current separation', () => {
  it('reads it from the assessment prediction details', () => {
    expect(
      currentSeparation({ prediction: { current_separation_px: 22.06 } }),
    ).toEqual({ value: 22.06, unit: 'px' });
  });

  it('is null when there is no prediction to read', () => {
    expect(currentSeparation({})).toBeNull();
    expect(currentSeparation({ prediction: null })).toBeNull();
    expect(currentSeparation(undefined)).toBeNull();
  });

  it('is null when the prediction never measured a separation', () => {
    expect(currentSeparation({ prediction: { outcome: 'PREDICTION_UNAVAILABLE' } })).toBeNull();
  });
});
