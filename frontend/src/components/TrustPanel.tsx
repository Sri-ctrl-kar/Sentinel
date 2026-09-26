/**
 * The trust panel: what ran this, and how far to trust the words above it.
 *
 * Every value is read from API metadata. Where the API does not report
 * something — precision, for instance, which the status endpoint does not
 * carry — the panel says "not reported" rather than asserting a plausible
 * default. A trust panel that guesses is worse than no trust panel.
 */

import type { GroundingPayload, PipelineSummary, DeviceInfo } from '../api/types';
import { MISSING, modelLabel } from '../lib/format';

interface Props {
  device: DeviceInfo | null;
  pipeline: PipelineSummary | null;
  grounding: GroundingPayload | null;
  coordinateSpace: string | null;
}

export function TrustPanel({ device, pipeline, grounding, coordinateSpace }: Props) {
  const detector = pipeline?.detector ?? null;

  return (
    <section className="panel" aria-label="Technical provenance">
      <header className="panel__head">
        <h2 className="panel__title">Provenance</h2>
      </header>

      <div className="panel__body">
        <div className="trust">
          <div>
            <div className="trust__label">Device</div>
            <div className="trust__value" data-testid="trust-device">
              {device ? device.label : MISSING}
            </div>
          </div>
          <div>
            <div className="trust__label">Model</div>
            <div className="trust__value" data-testid="trust-model">
              {modelLabel(detector?.model)}
            </div>
          </div>
          <div>
            <div className="trust__label">Backend</div>
            <div className="trust__value">{detector?.backend ?? MISSING}</div>
          </div>
          <div>
            <div className="trust__label">Precision</div>
            <div
              className="trust__value trust__value--missing"
              data-testid="trust-precision"
              title="The API does not report inference precision"
            >
              not reported
            </div>
          </div>
          <div>
            <div className="trust__label">Coordinate space</div>
            <div className="trust__value">
              {coordinateSpace === 'ground_plane_meters'
                ? 'ground plane · metres'
                : coordinateSpace === 'image_pixels'
                  ? 'image pixels'
                  : MISSING}
            </div>
          </div>
          <div>
            <div className="trust__label">Grounding</div>
            <div
              className={`trust__value ${
                grounding
                  ? grounding.ok
                    ? 'trust__value--pass'
                    : 'trust__value--fail'
                  : 'trust__value--missing'
              }`}
              data-testid="trust-grounding"
            >
              {grounding ? (grounding.ok ? 'PASS' : 'FAIL') : 'not run'}
            </div>
          </div>

          <p className="trust__note" data-testid="score-disclaimer">
            Risk score is an engineering risk signal, not a probability.
          </p>
          {coordinateSpace === 'image_pixels' && (
            <p className="trust__note">
              No camera calibration was supplied, so distances are image pixels
              and are not physical distances.
            </p>
          )}
        </div>
      </div>
    </section>
  );
}
