import { resolve } from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Multi-page build: /xr/ (Quest Browser, WebXR) and /dashboard/ (desktop).
// `npm run dev` proxies the Python server so both pages hot-reload.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
    chunkSizeWarningLimit: 1500,
    rollupOptions: {
      input: {
        xr: resolve(import.meta.dirname, "xr/index.html"),
        dashboard: resolve(import.meta.dirname, "dashboard/index.html"),
      },
    },
  },
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
});
