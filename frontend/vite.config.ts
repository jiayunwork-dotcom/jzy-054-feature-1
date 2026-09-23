import { defineConfig } from 'vite';

// During `npm run dev` the frontend talks to the FastAPI backend through
// this proxy; in production nginx performs the same routing.
export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    target: 'es2022',
    sourcemap: true,
  },
});
