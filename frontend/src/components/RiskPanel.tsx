/**
 * UNDERSTAND — the risk signal, as the engine reported it.
 *
 * The number is the engine's ordinal score. It is never restated as a
 * percentage, a probability or a likelihood, and the disclaimer the API sends
 * with it travels with it here.
 *
 * "As of" matters: risk is assessed on its own grid, coarser than the frame
 * rate, so the panel names the moment the shown assessment belongs to. Before
 * the first assessment it says there is none rather than showing zero.
 */

import type { RiskReportPayload, RiskResponse } from '../api/types';
import { formatScore, formatSeconds, riskState, titleise } from '../lib/format';

interface Props {
  risk: RiskResponse | null;
  current: RiskReportPayload | null;
  currentTime: number;
}

export function RiskPanel({ risk, current, currentTime }: Props) {
  const assessment = current?.assessments[0] ?? null;
  const severity = assessment?.severity ?? null;
  const state = riskState(severity);
  const score = assessment?.risk_score ?? null;
  const timeToRisk = assessment?.time_to_risk ?? null;

  return (
    <section className="panel" aria-label="Predictive risk">
      <header className="panel__head">
        <span className="panel__step">UNDERSTAND</span>
        <h2 className="panel__title">Predictive risk</h2>
        <span className="panel__aside">
          {risk ? `${risk.report_count} assessments · step ${risk.risk_step}s` : ''}
        </span>
      </header>

      <div className="panel__body">
        {assessment === null ? (
          <p className="empty" data-testid="risk-none">
            {currentTime > 0
              ? 'No risk assessment at this moment'
              : 'No risk assessment yet'}
          </p>
        ) : (
          <>
            <div className="risk">
              <div className={`risk__score state--${state}`} data-testid="risk-score">
                {formatScore(score)}
                <span className="risk__of"> /100</span>
              </div>
              <div>
                <div className={`risk__state state--${state}`} data-testid="risk-state">
                  {state}
                </div>
                <div className="risk__band">
                  engine severity band: {titleise(severity)}
                </div>
                <div className="bar" role="presentation">
                  <div
                    className={`bar__fill bar__fill--${state}`}
                    style={{ width: `${Math.min(score ?? 0, 100)}%` }}
                  />
                </div>
                <div className="risk__asof" data-testid="risk-asof">
                  as of {formatSeconds(current?.timestamp, 2)}
                </div>
              </div>
            </div>

            <div className="ttr">
              <div className="ttr__label" id="ttr-label">
                Time to risk
              </div>
              <TimeToRisk timeToRisk={timeToRisk} />
            </div>
          </>
        )}

        {risk && <Sparkline reports={risk.timeline} currentTime={currentTime} />}
      </div>
    </section>
  );
}

function TimeToRisk({
  timeToRisk,
}: {
  timeToRisk: { status: string; seconds: number | null; reason: string | null } | null;
}) {
  if (!timeToRisk) {
    return (
      <>
        <div
          className="ttr__value ttr__value--unavailable"
          data-testid="ttr-value"
          aria-labelledby="ttr-label"
        >
          unavailable
        </div>
        <div className="ttr__note">No time-to-risk was produced for this moment.</div>
      </>
    );
  }

  if (timeToRisk.status === 'already_unsafe') {
    return (
      <>
        <div
          className="ttr__value state--CRITICAL"
          data-testid="ttr-value"
          aria-labelledby="ttr-label"
        >
          UNSAFE NOW
        </div>
        <div className="ttr__note">
          The unsafe separation threshold is crossed at this moment — this is a
          present condition, not a prediction.
        </div>
      </>
    );
  }

  if (timeToRisk.status === 'predicted' && timeToRisk.seconds !== null) {
    return (
      <>
        <div className="ttr__value state--HIGH" data-testid="ttr-value" aria-labelledby="ttr-label">
          {timeToRisk.seconds.toFixed(2)} s
        </div>
        <div className="ttr__note">
          Predicted time until the unsafe separation threshold is crossed.
        </div>
      </>
    );
  }

  return (
    <>
      <div
        className="ttr__value ttr__value--unavailable"
        data-testid="ttr-value"
        aria-labelledby="ttr-label"
      >
        unavailable
      </div>
      <div className="ttr__note">
        {timeToRisk.reason
          ? `Not predicted: ${timeToRisk.reason.replace(/_/g, ' ')}.`
          : 'Not predicted within the prediction horizon.'}
      </div>
    </>
  );
}

/** Risk over the clip. A step line, because that is how it was assessed. */
function Sparkline({
  reports,
  currentTime,
}: {
  reports: RiskReportPayload[];
  currentTime: number;
}) {
  if (reports.length < 2) return null;
  const width = 300;
  const height = 42;
  const last = reports[reports.length - 1].timestamp || 1;

  const points = reports
    .map((report) => {
      const x = (report.timestamp / last) * width;
      const score = report.assessment_count > 0 ? report.max_risk_score : 0;
      const y = height - (Math.min(score, 100) / 100) * (height - 3) - 1.5;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' L ');

  const playheadX = Math.min((currentTime / last) * width, width);

  return (
    <div className="sparkline">
      <svg
        className="sparkline__svg"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        role="img"
        aria-label={`Risk score across the clip, peaking at ${formatScore(
          Math.max(...reports.map((r) => r.max_risk_score)),
        )} out of 100`}
        data-testid="risk-sparkline"
      >
        <path d={`M ${points}`} fill="none" stroke="#22d3ee" strokeWidth="1.5" />
        <line
          x1={playheadX}
          y1={0}
          x2={playheadX}
          y2={height}
          stroke="#e6edf3"
          strokeWidth="1"
          opacity="0.6"
        />
      </svg>
    </div>
  );
}
