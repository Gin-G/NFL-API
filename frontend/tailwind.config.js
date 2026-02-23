/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        brand: {
          green: '#1a7a4a',
          red: '#c8102e',
          gold: '#d4b94e',
        },
      },
    },
  },
  plugins: [],
}
