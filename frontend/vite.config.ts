import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During `npm run dev`, /api requests are proxied to the FastAPI backend on
// http://127.0.0.1:8002 so the frontend can use relative URLs in production
// builds (where both are typically served from the same origin). Override
// with the VITE_BACKEND_URL env var if your backend runs on a different host.
//
// We use 127.0.0.1 (not "localhost") on purpose: on Windows, "localhost"
// resolves to ::1 (IPv6) first, but uvicorn's default bind is 127.0.0.1
// (IPv4) only, which makes the proxy fail with ECONNREFUSED ("Backend
// unreachable" in the UI).
//
// Port 8002 is used (rather than the FastAPI/uvicorn default 8000) to avoid
// colliding with other local dev servers on the same machine.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_BACKEND_URL ?? "http://127.0.0.1:8002",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
