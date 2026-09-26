/**
 * Bounding boxes drawn over the video.
 *
 * SVG rather than canvas, for three reasons: the `viewBox` does the pixel→CSS
 * scaling for free (so a box lands in the right place at any player size), the
 * shapes are inspectable by tests and assistive tech, and it stays crisp.
 *
 * The boxes are the tracker's own, in image pixels, exactly as the API returned
 * them. Nothing is smoothed or interpolated between frames.
 */

import type { TrackPayload } from '../api/types';

/** Stable per-track colours. Track identity is also printed, never colour-only. */
const TRACK_COLOURS = [
  '#22d3ee',
  '#a78bfa',
  '#4ade80',
  '#fbbf24',
  '#f472b6',
  '#60a5fa',
];

export function trackColour(trackId: number): string {
  return TRACK_COLOURS[Math.abs(trackId) % TRACK_COLOURS.length];
}

interface Props {
  tracks: TrackPayload[];
  width: number;
  height: number;
  /** Entity IDs the risk engine named in the current assessment. */
  involved?: string[];
}

export function TrackOverlay({ tracks, width, height, involved = [] }: Props) {
  if (width <= 0 || height <= 0) return null;

  return (
    <svg
      className="stage__overlay"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label={
        tracks.length
          ? `${tracks.length} tracked ${tracks.length === 1 ? 'object' : 'objects'}`
          : 'no tracked objects in this frame'
      }
      data-testid="track-overlay"
    >
      {tracks.map((track) => {
        const [x1, y1, x2, y2] = track.bbox;
        const boxWidth = Math.max(x2 - x1, 1);
        const boxHeight = Math.max(y2 - y1, 1);
        const colour = trackColour(track.track_id);
        const isInvolved = involved.includes(track.entity_id);
        const label = `#${track.track_id} ${track.class_name} ${(track.confidence * 100).toFixed(0)}%`;
        // Scale the label with the video so it stays legible at any size.
        const fontSize = Math.max(height * 0.028, 9);
        const padding = fontSize * 0.35;
        const labelWidth = label.length * fontSize * 0.58 + padding * 2;
        const labelY = y1 - fontSize - padding * 2 > 0 ? y1 - fontSize - padding * 2 : y1;

        return (
          <g key={track.track_id} data-testid={`track-${track.track_id}`}>
            <rect
              x={x1}
              y={y1}
              width={boxWidth}
              height={boxHeight}
              fill="none"
              stroke={colour}
              strokeWidth={isInvolved ? 3 : 1.5}
              strokeDasharray={isInvolved ? undefined : '6 3'}
              opacity={isInvolved ? 1 : 0.85}
            />
            <rect
              x={x1}
              y={labelY}
              width={labelWidth}
              height={fontSize + padding * 2}
              fill="#07090c"
              opacity={0.85}
            />
            <text
              x={x1 + padding}
              y={labelY + fontSize + padding * 0.4}
              fill={colour}
              fontSize={fontSize}
              fontFamily="ui-monospace, monospace"
            >
              {label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
