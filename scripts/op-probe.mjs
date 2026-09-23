// Diagnostic: run single-op GroupQueryAttention graphs (written by hand into <dir>) on WebGPU at several lengths.
//   node scripts/op-probe.mjs <dir> <model> <len>...
import { createServer } from "node:http";
import { createReadStream, statSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const [dirArg, model, ...lens] = process.argv.slice(2);
const root = resolve(dirArg);
const ortDir = fileURLToPath(new URL("../node_modules/onnxruntime-web/dist/", import.meta.url));
const page = `<!doctype html><script type="module">window.ort = await import("/ort/ort.webgpu.bundle.min.mjs"); window.ready = true;</script>`;
const server = createServer((req, res) => {
  const url = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (url === "/") return res.writeHead(200, { "content-type": "text/html" }).end(page);
  const file = url.startsWith("/ort/") ? join(ortDir, url.slice(5)) : join(root, url.slice(3));
  try {
    res.writeHead(200, { "content-length": statSync(file).size, "content-type": file.endsWith("mjs") ? "text/javascript" : file.endsWith("wasm") ? "application/wasm" : "application/octet-stream" });
    createReadStream(file).pipe(res);
  } catch { res.writeHead(404).end(); }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const browser = await chromium.launch({ channel: "chromium", args: ["--enable-unsafe-webgpu"] });
try {
  const tab = await browser.newPage();
  await tab.goto(`http://127.0.0.1:${server.address().port}/`);
  await tab.waitForFunction(() => window.ready);
  for (const n of lens.map(Number)) {
    const r = await tab.evaluate(async ({ model, n }) => {
      const { ort } = window;
      const s = await ort.InferenceSession.create(`/m/${model}.onnx`, { executionProviders: ["webgpu"] });
      const feeds = {};
      for (const name of s.inputNames) {
        const meta = s.inputMetadata.find((m) => m.name === name);
        if (name === "seqlens_k") feeds[name] = new ort.Tensor("int32", Int32Array.from([n - 1]), [1]);
        else if (name === "total") feeds[name] = new ort.Tensor("int32", Int32Array.from([n]), []);
        else {
          const shape = meta.shape.map((d) => (typeof d === "string" ? n : d));
          const size = shape.reduce((a, b) => a * b, 1);
          feeds[name] = new ort.Tensor("float32", Float32Array.from({ length: size }, (_, i) => Math.sin(i * 0.37) * 0.5), shape);
        }
      }
      const t0 = performance.now();
      try { await s.run(feeds); return `ok ${Math.round(performance.now() - t0)} ms`; }
      catch (e) { return String(e.message).split("\n")[0].slice(-120); }
      finally { await s.release(); }
    }, { model, n });
    console.log(`${model} len ${n}: ${r}`);
  }
} finally {
  await browser.close();
  server.close();
}
