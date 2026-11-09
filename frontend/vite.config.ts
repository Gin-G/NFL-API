import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// Every API prefix the dashboard calls, in one list. A missing entry means
// that page silently fails in dev while working in production, which is how
// /optimizer and /dfs went unproxied — the optimizer page could not be run
// locally at all.
const API_PREFIXES = [
  '/teams', '/schedules', '/players', '/coaches', '/ratings', '/player-grades',
  '/projections', '/optimizer', '/dfs', '/coverage', '/trenches', '/external',
  '/chat', '/health',
]

export default defineConfig(({ mode }) => {
  // Defaults to production, so `npm run dev` behaves as it always has. Point
  // it at a local uvicorn to work on an endpoint that is not deployed yet:
  //   VITE_API_TARGET=http://127.0.0.1:8000 npm run dev
  const env = loadEnv(mode, process.cwd(), '')
  const target = env.VITE_API_TARGET || 'https://nfl-api.nickknows.net'
  return {
    plugins: [react()],
    base: '/dashboard/',
    server: {
      proxy: Object.fromEntries(
        API_PREFIXES.map((prefix) => [prefix, { target, changeOrigin: true }]),
      ),
    },
  }
})
