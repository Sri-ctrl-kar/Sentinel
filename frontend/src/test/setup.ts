import '@testing-library/jest-dom/vitest';

// jsdom does not implement media playback. The overlay and the transport read
// and write currentTime, so it needs to behave like a real property.
Object.defineProperty(window.HTMLMediaElement.prototype, 'play', {
  configurable: true,
  value: () => Promise.resolve(),
});
Object.defineProperty(window.HTMLMediaElement.prototype, 'pause', {
  configurable: true,
  value: () => undefined,
});
