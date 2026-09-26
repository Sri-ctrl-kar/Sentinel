/**
 * WHY THIS ALERT — the grounded reasons, and only those.
 *
 * Every line comes from the evidence the backend returned: the risk factors
 * that actually scored, with their own rationales, and the measurements with
 * their own units. Nothing is paraphrased into a claim the engine did not make,
 * and a factor that scored zero is not dressed up as a reason.
 */

import type { IncidentResponse } from '../api/types';
import { formatQuantity, humanise } from '../lib/format';

interface Props {
  incident: IncidentResponse | null;
}

export function EvidencePanel({ incident }: Props) {
  const evidence = incident?.evidence ?? null;
  const active = (evidence?.factors ?? []).filter((factor) => factor.contribution > 0);
  const quantities = incident?.quantities ?? [];

  return (
    <section className="panel" aria-label="Why this alert">
      <header className="panel__head">
        <h2 className="panel__title">Why this alert?</h2>
        {evidence && (
          <span className="panel__aside">
            {evidence.coordinate_space === 'ground_plane_meters'
              ? 'calibrated · metres'
              : 'image pixels · not physical distance'}
          </span>
        )}
      </header>

      <div className="panel__body">
        {!evidence ? (
          <p className="empty">No incident evidence for this clip.</p>
        ) : (
          <>
            <ul className="evidence" data-testid="evidence-list">
              {active.length === 0 && (
                <li className="evidence__item">
                  <span className="evidence__text evidence__detail">
                    No factor scored above zero.
                  </span>
                </li>
              )}
              {active.map((factor) => (
                <li className="evidence__item" key={factor.name}>
                  <span className="evidence__check" aria-hidden="true">
                    ✓
                  </span>
                  <span className="evidence__text">
                    <span className="evidence__headline">{humanise(factor.name)}</span>
                    <span className="evidence__weight">
                      {' '}
                      · {factor.contribution.toFixed(1)} of {factor.weight.toFixed(0)} pts
                    </span>
                    <span className="evidence__detail">{factor.rationale}</span>
                  </span>
                </li>
              ))}
            </ul>

            {quantities.length > 0 && (
              <div className="measures" data-testid="evidence-measures">
                {quantities.map((quantity) => (
                  <div className="measure" key={quantity.name}>
                    <span className="measure__name">{quantity.name}</span>
                    <span className="measure__value">
                      {formatQuantity(quantity.value, quantity.unit)}
                    </span>
                  </div>
                ))}
              </div>
            )}

            {evidence.triggered_event_actions.length > 0 && (
              <div className="measures">
                <div className="measure">
                  <span className="measure__name">Temporal evidence</span>
                  <span className="measure__value">
                    {evidence.triggered_event_actions.map(humanise).join(', ')}
                  </span>
                </div>
                <div className="measure">
                  <span className="measure__name">Backing events</span>
                  <span className="measure__value">
                    {evidence.triggered_event_ids.length}
                  </span>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </section>
  );
}
