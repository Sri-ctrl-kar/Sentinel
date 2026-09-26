/**
 * The analysing state.
 *
 * Progress is the API's own count of frames. When it reports a percentage the
 * bar is proportional; when it does not, the bar is indeterminate and the
 * numbers still say exactly how many frames are done. Nothing here invents
 * motion to look busy.
 */

import type { AnalysisStatusResponse } from '../api/types';
import type { Phase } from '../hooks/useAnalysis';

const STAGE_LABEL: Record<string, string> = {
  uploading: 'Uploading video',
  queued: 'Queued for analysis',
  running: 'Analysing — detection, tracking, events, risk',
  loading: 'Loading results',
};

interface Props {
  phase: Phase;
  filename: string | null;
  status: AnalysisStatusResponse | null;
}

export function AnalysisProgress({ phase, filename, status }: Props) {
  const progress = status?.progress ?? null;
  const percent = progress?.percent ?? null;

  return (
    <div className="progress" role="status" aria-live="polite">
      <div className="progress__label" data-testid="progress-stage">
        {STAGE_LABEL[phase] ?? 'Working'}
      </div>
      <div className="progress__file">{filename ?? status?.original_filename ?? ''}</div>

      <div className="progress__track">
        <div
          className={`progress__fill ${
            percent === null ? 'progress__fill--indeterminate' : ''
          }`}
          style={percent === null ? undefined : { width: `${percent}%` }}
          data-testid="progress-fill"
        />
      </div>

      <div className="progress__meta">
        <span data-testid="progress-frames">
          {progress && progress.frames_processed > 0
            ? `${progress.frames_processed}${
                progress.total_frames ? ` / ${progress.total_frames}` : ''
              } frames`
            : 'waiting for the first frame'}
        </span>
        <span>{percent === null ? 'progress not reported' : `${percent.toFixed(0)}%`}</span>
      </div>
    </div>
  );
}
