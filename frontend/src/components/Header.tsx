/**
 * The masthead: identity, connection, and what hardware answered.
 *
 * The accelerator chip appears only when the API reports a non-CPU device, and
 * it prints the device's own label — an AMD card says "AMD ROCm / HIP", never
 * "CUDA".
 */

import type { DeviceInfo } from '../api/types';

interface Props {
  connected: boolean | null;
  device: DeviceInfo | null;
  apiBase: string;
  onReset?: () => void;
  canReset?: boolean;
}

export function Header({ connected, device, apiBase, onReset, canReset }: Props) {
  const accelerated = device !== null && device.kind !== 'cpu';

  return (
    <header className="app__header">
      <div className="brand">
        <span className="brand__mark">SENTINEL</span>
        <span className="brand__tagline">
          Predict the incident. Prevent the outcome.
        </span>
      </div>

      <div className="header__status">
        <span
          className={`chip ${connected ? 'chip--live' : connected === false ? 'chip--down' : ''}`}
          data-testid="connection-chip"
          title={apiBase}
        >
          <span className="chip__dot" aria-hidden="true" />
          {connected === null ? 'connecting' : connected ? 'API online' : 'API offline'}
        </span>

        {device && (
          <span
            className={`chip ${accelerated ? 'chip--accel' : ''}`}
            data-testid="device-chip"
          >
            <span className="chip__dot" aria-hidden="true" />
            {device.label}
            {device.runtime_version ? ` ${device.runtime_version}` : ''}
          </span>
        )}

        {canReset && onReset && (
          <button type="button" className="button button--ghost" onClick={onReset}>
            New video
          </button>
        )}
      </div>
    </header>
  );
}
