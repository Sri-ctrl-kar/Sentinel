/**
 * The incident's lifecycle, using the engine's own five states.
 *
 * Deliberately not a hand-written SEE → UNDERSTAND → PREDICT ladder: those are
 * the workspace's sections, while this shows what
 * `app/intelligence/lifecycle.py` actually derived, with its own descriptions.
 * Inventing intermediate stages would misreport the system.
 *
 * With a history, the trail is that situation's transitions in the order they
 * happened — not a fixed five-rung ladder. A ladder has one rung per state, so
 * a situation that resolves and escalates again has nowhere to put the second
 * `imminent`: the rung keeps its first timestamp and the column then reads
 * `… RESOLVED from 38.03 s` above `IMMINENT from 45.03 s`, a sequence that
 * never happened. Every entry below is a genuine change of state, at the
 * timestamp the backend assessed it. The ladder remains only for the case
 * where there is no history to show.
 */

import type { LifecyclePoint, LifecycleState } from '../api/types';
import { LIFECYCLE_DESCRIPTIONS, LIFECYCLE_ORDER, formatSeconds } from '../lib/format';
import { lifecycleTransitions } from '../lib/timeline';

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
  /**
   * The timestamp of the history point the playhead is standing in. A state
   * alone cannot say *which* `imminent` the viewer is in once a situation has
   * been imminent twice; this can, so `◄ NOW` lands on the right one.
   */
  nowAt?: number | null;
}

export function LifecycleTrail({ state, history = [], now, nowAt = null }: Props) {
  // `now === undefined` is "no playhead", not "no state at the playhead".
  const marked = now === undefined ? state : now;
  const currentIndex = state ? LIFECYCLE_ORDER.indexOf(state) : -1;

  const transitions = lifecycleTransitions(history);

  // Which entry carries `◄ NOW`. By timestamp where the caller supplied one —
  // the only answer that survives a repeated state. Otherwise the most recent
  // entry in the marked state, so a single marker is shown, never several.
  let markedIndex = -1;
  if (nowAt !== null) {
    transitions.forEach((point, index) => {
      if (point.timestamp <= nowAt + 1e-9) markedIndex = index;
    });
  } else if (marked !== null) {
    transitions.forEach((point, index) => {
      if (point.state === marked) markedIndex = index;
    });
  }

  // `lifecycle-at-<state>` names the moment a state was *first* entered, so a
  // repeat renders its own time without a second element claiming that name.
  const firstOccurrence = new Map<LifecycleState, number>();
  transitions.forEach((point, index) => {
    if (!firstOccurrence.has(point.state)) firstOccurrence.set(point.state, index);
  });

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
                Not assessed at this point in the clip.
              </li>
            )}

            {transitions.length > 0
              ? transitions.map((point, index) => {
                  const isCurrent = index === markedIndex;
                  const isFirst = firstOccurrence.get(point.state) === index;
                  return (
                    <li
                      className="lifecycle__step"
                      key={`${point.state}-${point.timestamp}-${index}`}
                      data-testid="lifecycle-segment"
                      data-state={point.state}
                    >
                      <span className="lifecycle__rail" aria-hidden="true">
                        <span
                          className={[
                            'lifecycle__dot',
                            'lifecycle__dot--reached',
                            isCurrent ? 'lifecycle__dot--current' : '',
                          ]
                            .filter(Boolean)
                            .join(' ')}
                        />
                        {index < transitions.length - 1 && (
                          <span className="lifecycle__line" />
                        )}
                      </span>
                      <span>
                        <span
                          className={`lifecycle__name ${isCurrent ? 'lifecycle__name--current' : ''}`}
                          data-testid={isCurrent ? 'lifecycle-current' : undefined}
                        >
                          {point.state.toUpperCase()}
                          {isCurrent && <span className="lifecycle__here">◄ NOW</span>}
                          <span
                            className="lifecycle__at"
                            data-testid={isFirst ? `lifecycle-at-${point.state}` : undefined}
                          >
                            from {formatSeconds(point.timestamp)}
                          </span>
                        </span>
                        <span className="lifecycle__desc">
                          {LIFECYCLE_DESCRIPTIONS[point.state]}
                        </span>
                      </span>
                    </li>
                  );
                })
              : LIFECYCLE_ORDER.map((step, index) => {
                  // No history: the five states, lit up to the worst moment's.
                  const reached = currentIndex >= 0 && index <= currentIndex;
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
