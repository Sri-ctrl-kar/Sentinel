/**
 * The incident's lifecycle, using the engine's own five states.
 *
 * Deliberately not a hand-written SEE → UNDERSTAND → PREDICT ladder: those are
 * the workspace's sections, while this shows what
 * `app/intelligence/lifecycle.py` actually derived, with its own descriptions.
 * Inventing intermediate stages would misreport the system.
 */

import type { LifecyclePoint, LifecycleState } from '../api/types';
import { LIFECYCLE_DESCRIPTIONS, LIFECYCLE_ORDER, formatSeconds } from '../lib/format';

interface Props {
  /** The state at the clip's worst moment. */
  state: LifecycleState | null;
  /** Every step the backend tracker derived, in order. Empty is legitimate. */
  history?: LifecyclePoint[];
  /**
   * The state at the playhead, which is where `◄ NOW` belongs. `null` means the
   * situation had not been assessed yet at this point in the clip; omitting the
   * prop entirely means the caller has no playhead, and the worst moment stands
   * in for it.
   */
  now?: LifecycleState | null;
}

export function LifecycleTrail({ state, history = [], now }: Props) {
  // `now === undefined` is "no playhead", not "no state at the playhead".
  const marked = now === undefined ? state : now;
  const currentIndex = state ? LIFECYCLE_ORDER.indexOf(state) : -1;

  // When the clip's history is known, a rung is lit only if the situation
  // genuinely passed through it. Marking every earlier rung would invent a
  // progression: a situation can go straight from observed to current.
  const entered = new Map<LifecycleState, number>();
  for (const point of history) {
    if (!entered.has(point.state)) entered.set(point.state, point.timestamp);
  }

  return (
    <section className="panel" aria-label="Incident lifecycle">
      <header className="panel__head">
        <h2 className="panel__title">Incident lifecycle</h2>
        <span className="panel__aside">derived deterministically</span>
      </header>

      <div className="panel__body">
        {!state ? (
          <p className="empty">No incident lifecycle to show.</p>
        ) : (
          <ol className="lifecycle" data-testid="lifecycle-trail">
            {marked === null && (
              <li className="lifecycle__note" data-testid="lifecycle-unassessed">
                Not yet assessed at this point in the clip.
              </li>
            )}
            {LIFECYCLE_ORDER.map((step, index) => {
              const at = entered.get(step);
              const reached =
                history.length > 0
                  ? at !== undefined
                  : currentIndex >= 0 && index <= currentIndex;
              const isCurrent = step === marked;
              return (
                <li className="lifecycle__step" key={step}>
                  <span className="lifecycle__rail" aria-hidden="true">
                    <span
                      className={[
                        'lifecycle__dot',
                        reached ? 'lifecycle__dot--reached' : '',
                        isCurrent ? 'lifecycle__dot--current' : '',
                      ]
                        .filter(Boolean)
                        .join(' ')}
                    />
                    {index < LIFECYCLE_ORDER.length - 1 && (
                      <span className="lifecycle__line" />
                    )}
                  </span>
                  <span>
                    <span
                      className={`lifecycle__name ${isCurrent ? 'lifecycle__name--current' : ''}`}
                      data-testid={isCurrent ? 'lifecycle-current' : undefined}
                    >
                      {step.toUpperCase()}
                      {isCurrent && <span className="lifecycle__here">◄ NOW</span>}
                      {at !== undefined && (
                        <span className="lifecycle__at" data-testid={`lifecycle-at-${step}`}>
                          from {formatSeconds(at)}
                        </span>
                      )}
                    </span>
                    <span className="lifecycle__desc">
                      {LIFECYCLE_DESCRIPTIONS[step]}
                    </span>
                  </span>
                </li>
              );
            })}
          </ol>
        )}
      </div>
    </section>
  );
}
