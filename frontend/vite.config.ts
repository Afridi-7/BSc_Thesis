import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `process` is a Node.js global available to vite.config.ts at config time.
// Declared inline to avoid pulling in @types/node just for one env-var lookup.
declare const process: { env: Record<string, string | undefined> };

// During `npm run dev`, /api requests are proxied to the FastAPI backend on
// http://localhost:8767 so the frontend can use relative URLs in production
// builds (where both are typically served from the same origin). Override
// with the VITE_BACKEND_URL env var if your backend runs on a different host.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_BACKEND_URL ?? "http://localhost:8767",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
