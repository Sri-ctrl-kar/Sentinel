/**
 * The AI interpretation, fenced off from the deterministic signal.
 *
 * Mirrors the CLI's DETERMINISTIC / AI boundary: this panel states which
 * provider wrote the text, whether it was a language model at all, and the
 * grounding verdict the backend computed against the evidence. The explanation
 * explains; it never decides.
 */

import type { IncidentResponse } from '../api/types';

interface Props {
  incident: IncidentResponse | null;
}

export function NarrativePanel({ incident }: Props) {
  const explanation = incident?.explanation ?? null;
  const grounding = incident?.grounding ?? null;

  if (!explanation) return null;

  return (
    <section className="panel" aria-label="AI incident interpretation">
      <header className="panel__head">
        <h2 className="panel__title">AI incident interpretation</h2>
        <span className="panel__aside">explains · does not decide</span>
      </header>

      <div className="panel__body">
        <p className="narrative__summary" data-testid="ai-summary">
          {explanation.summary}
        </p>

        <div className="narrative__block">
          <div className="narrative__label">Severity</div>
          <p className="narrative__summary">{explanation.severity_explanation}</p>
        </div>

        {explanation.evidence_points.length > 0 && (
          <div className="narrative__block">
            <div className="narrative__label">Evidence points</div>
            <ul className="narrative__points">
              {explanation.evidence_points.map((point) => (
                <li key={point}>{point}</li>
              ))}
            </ul>
          </div>
        )}

        <div className="narrative__block">
          <div className="narrative__label">Uncertainty</div>
          <p className="narrative__summary">{explanation.uncertainty}</p>
        </div>

        <div className="narrative__provenance" data-testid="ai-provenance">
          {explanation.provider} · {explanation.model} ·{' '}
          {explanation.is_language_model
            ? 'language model'
            : 'deterministic template, not a language model'}
          {grounding && (
            <>
              {' · grounding '}
              <span
                className={grounding.ok ? 'trust__value--pass' : 'trust__value--fail'}
              >
                {grounding.ok ? 'PASS' : `FAIL (${grounding.violations.length})`}
              </span>
            </>
          )}
        </div>

        {grounding && !grounding.ok && (
          <ul className="narrative__points" data-testid="grounding-violations">
            {grounding.violations.map((violation) => (
              <li key={`${violation.code}-${violation.excerpt}`}>
                <strong>{violation.code}</strong>: {violation.detail}
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}
