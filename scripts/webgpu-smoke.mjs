// Runs exported hidden-state graphs in Chromium on WebGPU and compares the last position with ORT-CPU references.
// Usage: node scripts/webgpu-smoke.mjs <dir with ref.json and one subdirectory per variant>
import { createServer } from "node:http";
import { createReadStream, statSync } from "node:fs";
import { join, resolve, extname } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const root = resolve(process.argv[2] ?? "/tmp/jev-tiny");
const ortDir = fileURLToPath(new URL("../node_modules/onnxruntime-web/dist/", import.meta.url));
const types = { ".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm", ".json": "application/json", ".html": "text/html" };
const page = `<!doctype html><script type="module">window.ort = await import("/ort/ort.webgpu.bundle.min.mjs"); window.ready = true;</script>`;

const server = createServer((req, res) => {
  const url = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (url === "/") return res.writeHead(200, { "content-type": "text/html" }).end(page);
  const file = url.startsWith("/ort/") ? join(ortDir, url.slice(5)) : url.startsWith("/m/") ? join(root, url.slice(3)) : null;
  if (!file || !file.startsWith(url.startsWith("/ort/") ? ortDir : root)) return res.writeHead(404).end();
  try {
    const { size } = statSync(file);
    res.writeHead(200, { "content-type": types[extname(file)] ?? "application/octet-stream", "content-length": size });
    createReadStream(file).pipe(res);
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const base = `http://127.0.0.1:${server.address().port}/`;

const browser = await chromium.launch({ channel: "chromium", args: ["--enable-unsafe-webgpu"] });
try {
  const tab = await browser.newPage();
  tab.on("console", (m) => m.type() === "error" && console.error("[page]", m.text()));
  await tab.goto(base);
  await tab.waitForFunction(() => window.ready);
  const adapter = await tab.evaluate(async () => {
    const a = await navigator.gpu?.requestAdapter();
    return a && { vendor: a.info.vendor, arch: a.info.architecture, maxBufferSize: a.limits.maxBufferSize, maxStorageBufferBindingSize: a.limits.maxStorageBufferBindingSize };
  });
  console.log("adapter", adapter);
  const ref = await (await fetch(base + "m/ref.json")).json();
  for (const [variant, { past, cases }] of Object.entries(ref)) {
    const result = await tab.evaluate(async ({ variant, past, cases }) => {
      const { ort } = window;
      ort.env.logLevel = "error";
      const session = await ort.InferenceSession.create(`/m/${variant}/model.onnx`, {
        executionProviders: ["webgpu"],
        externalData: [{ path: "model.onnx.data", data: `/m/${variant}/model.onnx.data` }],
      });
      const out = [];
      for (const { ids, ortcpu } of cases) {
        const n = ids.length;
        const feeds = {
          input_ids: new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, n]),
          attention_mask: new ort.Tensor("int64", new BigInt64Array(n).fill(1n), [1, n]),
          position_ids: new ort.Tensor("int64", BigInt64Array.from(ids.map((_, i) => BigInt(i))), [1, n]),
        };
        for (const [name, shape] of Object.entries(past)) feeds[name] = new ort.Tensor("float32", new Float32Array(0), shape);
        const t0 = performance.now();
        const { hidden_states } = await session.run(feeds);
        const ms = performance.now() - t0;
        const h = hidden_states.data, d = h.length / n, last = h.subarray((n - 1) * d);
        let maxd = 0, maxr = 0;
        for (let i = 0; i < d; i++) { maxd = Math.max(maxd, Math.abs(last[i] - ortcpu[i])); maxr = Math.max(maxr, Math.abs(ortcpu[i])); }
        out.push({ n, maxd, rel: maxd / maxr, ms: Math.round(ms) });
      }
      await session.release();
      return out;
    }, { variant, past, cases });
    for (const r of result) console.log(`${variant.padEnd(14)} len ${String(r.n).padStart(4)}  WebGPU vs ORT-CPU last-pos max|Δ| ${r.maxd.toExponential(3)}  rel ${r.rel.toExponential(2)}  ${r.ms} ms`);
  }
} finally {
  await browser.close();
  server.close();
}
