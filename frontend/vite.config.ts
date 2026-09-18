import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

// Judge demo: Vite serves the dashboard locally. API calls go directly to
// the local FastAPI backend (default http://127.0.0.1:8000, override with
// VITE_API_BASE_URL). No production/public endpoint is hard-coded.
export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test-setup.ts'],
  },
})
