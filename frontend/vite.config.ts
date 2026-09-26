import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The frontend runs independently of the API during development: `npm run dev`
// serves on :5173 and talks to the FastAPI service over CORS at
// VITE_API_BASE_URL (default http://127.0.0.1:8000). No proxy, no coupling —
// either process can be restarted without the other.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, host: '127.0.0.1' },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    css: false,
  },
});
