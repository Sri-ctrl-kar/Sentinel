/**
 * PREDICT — what Sentinel expects to happen, marked as expectation.
 *
 * The tense badge is the point of this panel. A predicted conflict is labelled
 * PREDICTED and worded in the future; a present breach is labelled CURRENT. The
 * distinction comes from the engine's own `time_to_risk.status` and lifecycle
 * state, never from the score.
 *
 * The panel reads the playhead. `live` is this same situation's assessment at
 * the current video time and `liveState` the lifecycle the backend derived for
 * it, so scrubbing to 4 s shows what Sentinel knew at 4 s. Material that exists
 * only for the clip's worst moment — the AI narrative, the entity details — is
 * kept below a divider that names that moment, never shown as if it were now.
 */

import type {
  IncidentResponse,
  LifecycleState,
  RiskAssessmentPayload,
} from '../api/types';
import {
  MISSING,
  formatScore,
  formatSeconds,
  formatQuantity,
  humanise,
  recommendationFor,
  tenseOf,
  titleise,
} from '../lib/format';

interface Props {
  incident: IncidentResponse | null;
  /** The situation live at the playhead, or null if nothing was assessed there. */
  live?: RiskAssessmentPayload | null;
  /** The lifecycle state the backend derived for that same moment. */
  liveState?: LifecycleState | null;
  /** Is the live situation the one the worst-moment material below describes? */
  isWorstSituation?: boolean;
}

export function PredictionPanel({
  incident,
  live = null,
  liveState = null,
  isWorstSituation = false,
}: Props) {
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
  const outcome = incident.explanation?.predicted_outcome ?? null;

  // Where the playhead has an assessment for this situation, every live value
  // comes from it. Where it does not, the panel says so: the worst moment's
  // numbers are not a stand-in for a moment the engine never assessed.
  const established = live !== null;
  const timeToRisk = live?.time_to_risk ?? null;
  const state = liveState ?? null;
  const tense = established ? tenseOf(timeToRisk?.status, state) : 'UNKNOWN';
  // An intervention is prescribed only where the engine's prediction outcome
  // says the pair is unsafe now or is predicted to be. See `recommendationFor`.
  const recommendation = established
    ? recommendationFor(live!.prediction_outcome, live!.recommended_intervention)
    : null;

  return (
    <section className="panel" aria-label="Incident prediction">
      <header className="panel__head">
        <span className="panel__step">PREDICT</span>
        <h2 className="panel__title">Incident prediction</h2>
        <span className="panel__aside" data-testid="live-situation">
          {established
            ? live!.involved_entity_ids.join(' + ')
            : 'no situation at this moment'}
        </span>
      </header>

      <div className="panel__body">
        <span className={`tense tense--${tense}`} data-testid="tense-badge">
          {!established
            ? 'NOT ASSESSED AT THIS MOMENT'
            : tense === 'CURRENT'
              ? 'CURRENT — happening now'
              : tense === 'PREDICTED'
                ? 'PREDICTED — has not happened'
                : live?.prediction_outcome === 'NO_PREDICTED_CONFLICT'
                  ? 'OBSERVED — no conflict predicted'
                  : 'STATE NOT ESTABLISHED'}
        </span>

        <h3 className="prediction__headline" data-testid="incident-type">
          {titleise(live?.incident_type ?? evidence.incident_type)}
        </h3>

        {established && (
          <p className="prediction__scope" data-testid="situation-scope">
            {isWorstSituation
              ? "This is the clip's worst situation — the one described below."
              : "A different situation from the clip's worst moment, which is kept below."}
          </p>
        )}

        {!established ? (
          <p className="prediction__outcome" data-testid="prediction-outcome">
            Sentinel produced no assessment for this situation at this point in
            the clip.
          </p>
        ) : (
          <p className="prediction__outcome" data-testid="prediction-outcome">
            {humanise(live?.prediction_outcome ?? null)}
            {timeToRisk?.reason
              ? ` · ${humanise(timeToRisk.reason)}`
              : ''}
          </p>
        )}

        <div className="facts">
          <div>
            <div className="fact__label">Severity</div>
            <div className="fact__value">
              {established ? titleise(live!.severity) : MISSING}
            </div>
          </div>
          <div>
            <div className="fact__label">Lifecycle</div>
            <div className="fact__value" data-testid="lifecycle-value">
              {state ? titleise(state) : MISSING}
            </div>
          </div>
          <div>
            <div className="fact__label">Risk score</div>
            <div className="fact__value" data-testid="prediction-score">
              {established ? `${formatScore(live!.risk_score)} /100` : MISSING}
            </div>
          </div>
          <div>
            <div className="fact__label">Time to risk</div>
            <div className="fact__value" data-testid="prediction-ttr">
              {timeToRisk?.status === 'predicted' && timeToRisk.seconds !== null
                ? formatSeconds(timeToRisk.seconds)
                : timeToRisk?.status === 'already_unsafe'
                  ? 'unsafe now'
                  : established
                    ? 'unavailable'
                    : MISSING}
            </div>
          </div>
          <div>
            <div className="fact__label">Engine confidence</div>
            <div className="fact__value">
              {established ? live!.confidence.toFixed(2) : MISSING}
            </div>
          </div>
          <div>
            <div className="fact__label">As assessed at</div>
            <div className="fact__value" data-testid="prediction-as-of">
              {established ? formatSeconds(live!.timestamp) : MISSING}
            </div>
          </div>
        </div>

        {recommendation && (
          <div className="narrative__block">
            <div className="narrative__label">Recommended for a human operator</div>
            <div className="narrative__summary" data-testid="prediction-recommendation">
              {recommendation}
            </div>
          </div>
        )}

        <div className="panel__divider" data-testid="worst-moment-divider">
          Worst moment · {evidence.incident_id} ·{' '}
          {formatSeconds(evidence.timestamp)} ·{' '}
          {formatScore(evidence.risk_score)} /100
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

      </div>
    </section>
  );
}
