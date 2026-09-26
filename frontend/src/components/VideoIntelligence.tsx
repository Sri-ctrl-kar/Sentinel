/**
 * SEE — the original video with Sentinel's tracking drawn over it.
 *
 * The `<video>` element plays the file the user uploaded, served back byte for
 * byte by the API. The server never draws on it; the overlay is rendered here,
 * from timeline coordinates, and follows the video clock.
 */

import { useEffect, useRef, useState } from 'react';

import type { TimelineFramePayload, VideoPayload } from '../api/types';
import { formatClock } from '../lib/format';
import { frameAt } from '../lib/timeline';
import { TrackOverlay, trackColour } from './TrackOverlay';

interface Props {
  videoSrc: string;
  video: VideoPayload | null;
  frames: TimelineFramePayload[];
  currentTime: number;
  onTimeChange: (time: number) => void;
  /** Set by the timeline when the operator clicks an event. */
  seekTo: number | null;
  involved: string[];
}

export function VideoIntelligence({
  videoSrc,
  video,
  frames,
  currentTime,
  onTimeChange,
  seekTo,
  involved,
}: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [duration, setDuration] = useState(video?.duration_seconds ?? 0);
  // A browser can decode the analysis but not the file: Sentinel accepts any
  // container OpenCV can read, which is a wider set than Chrome can play. When
  // that happens the dashboard stays fully usable — only the pixels are gone.
  const [videoError, setVideoError] = useState(false);

  useEffect(() => {
    if (seekTo === null) return;
    const element = videoRef.current;
    if (element) element.currentTime = seekTo;
    onTimeChange(seekTo);
  }, [seekTo, onTimeChange]);

  const frame = frameAt(frames, currentTime);
  const tracks = frame?.tracks ?? [];

  function togglePlay() {
    const element = videoRef.current;
    if (!element || videoError) return;
    if (playing) {
      element.pause();
      setPlaying(false);
    } else {
      void element.play();
      setPlaying(true);
    }
  }

  function scrub(value: number) {
    const element = videoRef.current;
    // Without decodable pixels the slider still drives the analysis clock, so
    // tracks, risk and the timeline remain explorable.
    if (element && !videoError) element.currentTime = value;
    onTimeChange(value);
  }

  return (
    <section className="panel" aria-label="Video intelligence">
      <header className="panel__head">
        <span className="panel__step">SEE</span>
        <h2 className="panel__title">Video intelligence</h2>
        <span className="panel__aside">
          {frame ? `frame ${frame.frame_index}` : 'no frame data'}
          {video ? ` · ${video.width}×${video.height}` : ''}
        </span>
      </header>

      <div
        className={`stage ${videoError ? 'stage--fallback' : ''}`}
        style={
          videoError && video
            ? { aspectRatio: `${video.width} / ${video.height}` }
            : undefined
        }
      >
        <video
          hidden={videoError}
          ref={videoRef}
          className="stage__video"
          src={videoSrc}
          data-testid="video"
          preload="auto"
          playsInline
          onTimeUpdate={(event) => onTimeChange(event.currentTarget.currentTime)}
          onLoadedMetadata={(event) => {
            const element = event.currentTarget;
            if (Number.isFinite(element.duration) && element.duration > 0) {
              setDuration(element.duration);
            }
          }}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => setPlaying(false)}
          onError={() => setVideoError(true)}
        />
        {videoError && (
          <div className="stage__fallback" data-testid="video-unplayable">
            <strong>This browser cannot decode this video file.</strong>
            <span>
              The analysis is unaffected — tracks, risk and evidence below are
              complete. Browsers play H.264/MP4; other codecs Sentinel can read
              may not render here.
            </span>
          </div>
        )}
        {video && (
          <TrackOverlay
            tracks={tracks}
            width={video.width}
            height={video.height}
            involved={involved}
          />
        )}
      </div>

      <div className="transport">
        <button
          type="button"
          className="transport__button"
          onClick={togglePlay}
          disabled={videoError}
          aria-label={playing ? 'Pause video' : 'Play video'}
        >
          {playing ? 'PAUSE' : 'PLAY'}
        </button>
        <span className="transport__time">{formatClock(currentTime)}</span>
        <input
          className="transport__scrub"
          type="range"
          min={0}
          max={duration || video?.duration_seconds || 1}
          step={0.02}
          value={Math.min(currentTime, duration || video?.duration_seconds || 1)}
          onChange={(event) => scrub(Number(event.target.value))}
          aria-label={videoError ? 'Scrub analysis timeline' : 'Scrub video'}
        />
        <span className="transport__time">
          {formatClock(duration || video?.duration_seconds || null)}
        </span>
      </div>

      <div className="tracklist">
        {tracks.length === 0 ? (
          <span className="empty" style={{ padding: 0 }}>
            No tracks in this frame
          </span>
        ) : (
          tracks.map((track) => (
            <span
              key={track.track_id}
              className="tracktag"
              style={{ borderLeftColor: trackColour(track.track_id) }}
            >
              #{track.track_id} {track.class_name} ·{' '}
              {(track.confidence * 100).toFixed(0)}%
            </span>
          ))
        )}
      </div>
    </section>
  );
}
