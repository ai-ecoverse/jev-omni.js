// Harness server for scripts/browser-eval.ts: the page, onnxruntime-web's wasm at /ort/, and a bundle directory
// (JEV_BUNDLE) at /models/, streamed from disk.
import { createReadStream, statSync } from "node:fs";
import { join, resolve } from "node:path";
import { defineConfig, type Plugin } from "vite";

const bundle = resolve(process.env.JEV_BUNDLE ?? "public/models/jev-omni");
const ortDist = resolve("node_modules/onnxruntime-web/dist");

function serve(prefix: string, dir: string): Plugin {
  return {
    name: `serve-${prefix}`,
    configureServer(server) {
      server.middlewares.use(prefix, (req, res, next) => {
        const file = join(dir, decodeURIComponent((req.url ?? "/").split("?")[0]));
        if (!file.startsWith(dir)) return next();
        try {
          const { size } = statSync(file);
          res.writeHead(200, { "content-length": size, "content-type": file.endsWith(".mjs") ? "text/javascript" : file.endsWith(".wasm") ? "application/wasm" : "application/octet-stream" });
          createReadStream(file).pipe(res);
        } catch { next(); }
      });
    },
  };
}

export default defineConfig({
  root: "test/browser",
  server: { port: Number(process.env.JEV_PORT ?? 5175), strictPort: true, fs: { allow: [resolve(".")] } },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  plugins: [serve("/models", bundle), serve("/ort", ortDist)],
});
