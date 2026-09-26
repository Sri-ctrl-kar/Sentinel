/**
 * The incident's lifecycle, using the engine's own five states.
 *
 * Deliberately not a hand-written SEE → UNDERSTAND → PREDICT ladder: those are
 * the workspace's sections, while this shows what
 * `app/intelligence/lifecycle.py` actually derived, with its own descriptions.
 * Inventing intermediate stages would misreport the system.
 */

import type { LifecycleState } from '../api/types';
import { LIFECYCLE_DESCRIPTIONS, LIFECYCLE_ORDER } from '../lib/format';

interface Props {
  state: LifecycleState | null;
}

export function LifecycleTrail({ state }: Props) {
  const currentIndex = state ? LIFECYCLE_ORDER.indexOf(state) : -1;

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
            {LIFECYCLE_ORDER.map((step, index) => {
              const reached = currentIndex >= 0 && index <= currentIndex;
              const isCurrent = step === state;
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
