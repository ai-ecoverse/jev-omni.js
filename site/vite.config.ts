import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";

// GitHub Pages serves the project at /<repo>/; PAGES_BASE sets that prefix in CI. The page loads the weights from
// Hugging Face, so the build is the page, its preset images and onnxruntime-web (the pinned pre-release, bundled).
// No cross-origin isolation: Pages cannot send COOP/COEP, and WebGPU does not need it (wasm runs single-threaded).
export default defineConfig({
  base: process.env.PAGES_BASE ?? "/",
  root: fileURLToPath(new URL(".", import.meta.url)),
  // dev only: public/models/jev-omni from jev_omni_web_export.package, served at /models/jev-omni
  publicDir: fileURLToPath(new URL("../public", import.meta.url)),
  server: { host: "127.0.0.1", port: 5176, allowedHosts: [".getbb.app"] },
  preview: { host: "127.0.0.1", port: 4176 },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  worker: { format: "es" },
  build: { outDir: fileURLToPath(new URL("../dist-site", import.meta.url)), target: "es2023", copyPublicDir: false, emptyOutDir: true },
});
