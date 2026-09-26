/** A failure, stated plainly, with the one action that can follow it. */

interface Props {
  title?: string;
  message: string;
  hint?: string;
  onRetry?: () => void;
  retryLabel?: string;
}

export function ErrorState({
  title = 'Analysis failed',
  message,
  hint,
  onRetry,
  retryLabel = 'Try another video',
}: Props) {
  return (
    <div className="error" role="alert" data-testid="error-state">
      <div className="error__title">{title}</div>
      <p className="error__message">{message}</p>
      {hint && <p className="error__hint">{hint}</p>}
      {onRetry && (
        <button type="button" className="button" onClick={onRetry}>
          {retryLabel}
        </button>
      )}
    </div>
  );
}
