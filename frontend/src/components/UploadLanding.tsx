/**
 * The landing state: one action, stated plainly.
 *
 * Accepted formats are the backend's own list, so the picker and the server
 * cannot disagree about what will be rejected.
 */

import { useRef, useState } from 'react';

/** app/api/routes/analysis.py:ALLOWED_SUFFIXES */
export const ACCEPTED_SUFFIXES = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'];
/** app/api/routes/analysis.py:MAX_UPLOAD_BYTES */
export const MAX_UPLOAD_MB = 512;

interface Props {
  onSelect: (file: File) => void;
  disabled?: boolean;
}

export function UploadLanding({ onSelect, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragging, setDragging] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  function accept(file: File | undefined) {
    if (!file) return;
    const suffix = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
    if (!ACCEPTED_SUFFIXES.includes(suffix)) {
      setLocalError(
        `${suffix || 'that file'} is not a supported video format. Use ${ACCEPTED_SUFFIXES.join(', ')}.`,
      );
      return;
    }
    setLocalError(null);
    onSelect(file);
  }

  return (
    <div className="landing">
      <h1 className="landing__title">SENTINEL</h1>
      <p className="landing__lede">Predict the incident. Prevent the outcome.</p>
      <p className="landing__sub">
        Upload surveillance footage. Sentinel shows what it sees, what is
        happening, what is likely to happen next — and why it believes that.
      </p>

      <div
        className={`dropzone ${dragging ? 'dropzone--over' : ''}`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          accept(event.dataTransfer.files?.[0]);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_SUFFIXES.join(',')}
          className="visually-hidden"
          id="video-input"
          data-testid="video-input"
          disabled={disabled}
          onChange={(event) => accept(event.target.files?.[0])}
        />
        <label htmlFor="video-input" className="visually-hidden">
          Surveillance video file
        </label>
        <button
          type="button"
          className="button"
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
        >
          Upload surveillance video
        </button>
        <p className="formats">
          {ACCEPTED_SUFFIXES.join(' · ')} — up to {MAX_UPLOAD_MB} MB
        </p>
      </div>

      {localError && (
        <p className="error__message" role="alert" data-testid="upload-error">
          {localError}
        </p>
      )}
    </div>
  );
}
