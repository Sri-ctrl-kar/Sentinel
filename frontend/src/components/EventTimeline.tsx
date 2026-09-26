/**
 * The event timeline: what happened, when, and a way to get there.
 *
 * Two views of the same markers — a scrub track for shape, a list for detail.
 * Both seek the video, so reading and watching are the same act. Every marker
 * is a real recorded event or a real severity change between two assessments
 * the engine produced.
 */

import type { EventPayload, RiskReportPayload } from '../api/types';
import { formatSeconds, humanise } from '../lib/format';
import { buildMarkers } from '../lib/timeline';

interface Props {
  events: EventPayload[];
  reports: RiskReportPayload[];
  duration: number;
  currentTime: number;
  onSeek: (time: number) => void;
}

export function EventTimeline({
  events,
  reports,
  duration,
  currentTime,
  onSeek,
}: Props) {
  const markers = buildMarkers(events, reports);
  const span = duration > 0 ? duration : 1;

  return (
    <section className="panel" aria-label="Event timeline">
      <header className="panel__head">
        <h2 className="panel__title">Event timeline</h2>
        <span className="panel__aside">{markers.length} markers</span>
      </header>

      <div className="panel__body--flush">
        <div className="timeline__track" data-testid="timeline-track">
          <div
            className="timeline__playhead"
            style={{ left: `${Math.min((currentTime / span) * 100, 100)}%` }}
            aria-hidden="true"
          />
          {markers.map((marker) => (
            <button
              key={`tick-${marker.key}`}
              type="button"
              className={[
                'timeline__marker',
                marker.kind === 'risk' ? 'timeline__marker--risk' : '',
                marker.severity === 'high' ? 'timeline__marker--risk-high' : '',
                marker.severity === 'critical' ? 'timeline__marker--risk-critical' : '',
              ]
                .filter(Boolean)
                .join(' ')}
              style={{ left: `${Math.min((marker.timestamp / span) * 100, 100)}%` }}
              onClick={() => onSeek(marker.timestamp)}
              title={`${formatSeconds(marker.timestamp)} — ${marker.label}`}
            >
              <span className="visually-hidden">
                Seek to {formatSeconds(marker.timestamp)}, {humanise(marker.label)}
              </span>
            </button>
          ))}
        </div>

        <div className="timeline__list">
          {markers.length === 0 ? (
            <p className="empty">No events recorded.</p>
          ) : (
            markers.map((marker) => {
              const active = Math.abs(marker.timestamp - currentTime) < 0.2;
              return (
                <button
                  key={marker.key}
                  type="button"
                  className={`timeline__row ${active ? 'timeline__row--active' : ''}`}
                  onClick={() => onSeek(marker.timestamp)}
                  data-testid={`timeline-row-${marker.key}`}
                >
                  <span className="timeline__time">
                    {marker.timestamp.toFixed(2)}s
                  </span>
                  <span className="timeline__action">{humanise(marker.label)}</span>
                  <span className="timeline__detail">{marker.detail}</span>
                </button>
              );
            })
          )}
        </div>
      </div>
    </section>
  );
}
