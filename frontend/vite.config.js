import path from "path";
import { fileURLToPath } from "url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// In Docker Compose use backend service name; when running Vite on the host use mapped port 5050.
const apiProxyTarget =
  process.env.VITE_API_PROXY_TARGET || "http://127.0.0.1:5050";

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    include: ["vega", "vega-lite", "vega-embed", "react-vega"],
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/api": {
        target: apiProxyTarget,
        changeOrigin: true,
        secure: false,
        timeout: 3_600_000,
        proxyTimeout: 3_600_000,
      },
    },
  },
});
