/**
 * PREDICT — what Sentinel expects to happen, marked as expectation.
 *
 * The tense badge is the point of this panel. A predicted conflict is labelled
 * PREDICTED and worded in the future; a present breach is labelled CURRENT. The
 * distinction comes from the engine's own `time_to_risk.status` and lifecycle
 * state, never from the score.
 */

import type { IncidentResponse } from '../api/types';
import {
  MISSING,
  formatQuantity,
  formatScore,
  formatSeconds,
  humanise,
  tenseOf,
  titleise,
} from '../lib/format';

interface Props {
  incident: IncidentResponse | null;
}

export function PredictionPanel({ incident }: Props) {
  if (!incident || !incident.incident_found || !incident.evidence) {
    return (
      <section className="panel" aria-label="Incident prediction">
        <header className="panel__head">
          <span className="panel__step">PREDICT</span>
          <h2 className="panel__title">Incident prediction</h2>
        </header>
        <div className="panel__body">
          <p className="empty" data-testid="no-incident">
            {incident?.note ?? 'No incident was raised for this clip.'}
          </p>
        </div>
      </section>
    );
  }

  const evidence = incident.evidence;
  const prediction = evidence.prediction;
  const timeToRisk = evidence.time_to_risk;
  const tense = tenseOf(timeToRisk?.status, evidence.incident_state);
  const outcome = incident.explanation?.predicted_outcome ?? null;

  return (
    <section className="panel" aria-label="Incident prediction">
      <header className="panel__head">
        <span className="panel__step">PREDICT</span>
        <h2 className="panel__title">Incident prediction</h2>
        <span className="panel__aside">{evidence.incident_id}</span>
      </header>

      <div className="panel__body">
        <span className={`tense tense--${tense}`} data-testid="tense-badge">
          {tense === 'CURRENT'
            ? 'CURRENT — happening now'
            : tense === 'PREDICTED'
              ? 'PREDICTED — has not happened'
              : prediction?.outcome === 'NO_PREDICTED_CONFLICT'
                ? 'OBSERVED — no conflict predicted'
                : 'STATE NOT ESTABLISHED'}
        </span>

        <h3 className="prediction__headline" data-testid="incident-type">
          {titleise(evidence.incident_type)}
        </h3>

        {prediction && (
          <p className="prediction__outcome" data-testid="prediction-outcome">
            {humanise(prediction.outcome)}
            {prediction.horizon_seconds !== null
              ? ` · horizon ${formatSeconds(prediction.horizon_seconds, 1)}`
              : ''}
            {prediction.unavailable_reason
              ? ` · unavailable: ${humanise(prediction.unavailable_reason)}`
              : ''}
          </p>
        )}

        <div className="facts">
          <div>
            <div className="fact__label">Severity</div>
            <div className="fact__value">{titleise(evidence.severity)}</div>
          </div>
          <div>
            <div className="fact__label">Lifecycle</div>
            <div className="fact__value" data-testid="lifecycle-value">
              {titleise(evidence.incident_state)}
            </div>
          </div>
          <div>
            <div className="fact__label">Risk score</div>
            <div className="fact__value">{formatScore(evidence.risk_score)} /100</div>
          </div>
          <div>
            <div className="fact__label">Time to risk</div>
            <div className="fact__value" data-testid="prediction-ttr">
              {timeToRisk?.status === 'predicted' && timeToRisk.seconds !== null
                ? formatSeconds(timeToRisk.seconds)
                : timeToRisk?.status === 'already_unsafe'
                  ? 'unsafe now'
                  : 'unavailable'}
            </div>
          </div>
          <div>
            <div className="fact__label">Engine confidence</div>
            <div className="fact__value">{evidence.confidence.toFixed(2)}</div>
          </div>
          <div>
            <div className="fact__label">Predicted min. separation</div>
            <div className="fact__value">
              {prediction
                ? formatQuantity(prediction.minimum_separation, evidence.distance_unit)
                : MISSING}
            </div>
          </div>
        </div>

        <div className="entities" data-testid="involved-entities">
          {evidence.entities.map((entity) => (
            <span key={entity.entity_id} className="entity">
              <strong>{entity.entity_id}</strong>
              {entity.class_name ? ` · ${entity.class_name}` : ''}
              {entity.speed !== null
                ? ` · ${formatQuantity(entity.speed, evidence.speed_unit)}`
                : ''}
              {entity.zones.length ? ` · ${entity.zones.join(', ')}` : ''}
            </span>
          ))}
        </div>

        {outcome && (
          <p className="prediction__disclaimer" data-testid="predicted-outcome-text">
            {outcome}
          </p>
        )}

        {evidence.recommended_intervention && (
          <div className="narrative__block">
            <div className="narrative__label">Recommended for a human operator</div>
            <div className="narrative__summary">{evidence.recommended_intervention}</div>
          </div>
        )}
      </div>
    </section>
  );
}
