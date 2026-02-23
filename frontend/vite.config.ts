import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/dashboard/',
  server: {
    proxy: {
      '/teams': 'https://nfl-api.nickknows.net',
      '/schedules': 'https://nfl-api.nickknows.net',
      '/players': 'https://nfl-api.nickknows.net',
      '/coaches': 'https://nfl-api.nickknows.net',
      '/health': 'https://nfl-api.nickknows.net',
    },
  },
})
