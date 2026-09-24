// Runs the attention graphs written by export/jev_omni_web_export/attn_probe.py on WebGPU: correctness against the
// numpy reference at the check length, then success and time at longer lengths.
//   node scripts/attn-probe.mjs <dir> <len>...
import { createServer } from "node:http";
import { createReadStream, readFileSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const [dirArg, ...lens] = process.argv.slice(2);
const root = resolve(dirArg);
const meta = JSON.parse(readFileSync(join(root, "meta.json"), "utf8"));
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
  tab.on("console", (m) => m.type() === "error" && console.log("[page]", m.text().slice(0, 300)));
  await tab.goto(`http://127.0.0.1:${server.address().port}/`);
  await tab.waitForFunction(() => window.ready);
  for (const [kind, m] of Object.entries(meta)) {
    for (const prec of ["fp32", "fp16"]) {
      const name = `mha_${kind}_${prec}`;
      const r = await tab.evaluate(async ({ name, kind, m, lens }) => {
        const { ort } = window;
        ort.env.logLevel = "error";
        const bin = async (n) => new Float32Array(await (await fetch(`/m/${kind}_${n}.bin`)).arrayBuffer());
        const s = await ort.InferenceSession.create(`/m/${name}.onnx`, { executionProviders: ["webgpu"] });
        const out = [];
        const feedsFor = (n, q, k, v, bias) => ({
          q: new ort.Tensor("float32", q, [1, n, m.heads * m.head_dim]),
          k: new ort.Tensor("float32", k, [1, n, m.kv_heads * m.head_dim]),
          v: new ort.Tensor("float32", v, [1, n, m.kv_heads * m.head_dim]),
          bias: new ort.Tensor("float32", bias, [1, 1, n, n]),
        });
        try {
          const ref = await bin("ref");
          const { o } = await s.run(feedsFor(m.n, await bin("q"), await bin("k"), await bin("v"), await bin("bias")));
          let md = 0, mr = 0;
          for (let i = 0; i < ref.length; i++) { md = Math.max(md, Math.abs(o.data[i] - ref[i])); mr = Math.max(mr, Math.abs(ref[i])); }
          out.push(`check n=${m.n} max|Δ| ${md.toExponential(2)} (ref max ${mr.toFixed(2)})`);
        } catch (e) { out.push(`check failed: ${String(e.message).split("\n")[0].slice(-160)}`); }
        for (const n of lens) {
          const q = Float32Array.from({ length: n * m.heads * m.head_dim }, (_, i) => Math.sin(i * 0.37) * 0.25);
          const kv = Float32Array.from({ length: n * m.kv_heads * m.head_dim }, (_, i) => Math.cos(i * 0.11) * 0.25);
          const bias = new Float32Array(n * n);
          for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
            const ok = (j <= i || (i >= 40 && i < 313 && j >= 40 && j < 313)) && (!m.window || Math.abs(i - j) < m.window);
            if (!ok) bias[i * n + j] = -3.4e38;
          }
          try {
            await s.run(feedsFor(n, q, kv, kv, bias));
            const t0 = performance.now();
            await s.run(feedsFor(n, q, kv, kv, bias));
            out.push(`n=${n} ok ${Math.round(performance.now() - t0)} ms`);
          } catch (e) { out.push(`n=${n} failed: ${String(e.message).split("\n")[0].slice(-160)}`); }
        }
        await s.release();
        return out;
      }, { name, kind, m, lens: lens.map(Number) });
      console.log(name, "\n  " + r.join("\n  "));
    }
    for (const graph of [`if_${kind}`, `gqa_${kind}`]) {
      const r = await tab.evaluate(async ({ graph, kind, m, lens }) => {
        const { ort } = window;
        const bin = async (n) => new Float32Array(await (await fetch(`/m/${kind}_${n}.bin`)).arrayBuffer());
        const out = [];
        let s;
        try { s = await ort.InferenceSession.create(`/m/${graph}.onnx`, { executionProviders: ["webgpu"] }); }
        catch (e) { return [`create failed: ${String(e.message).split("\n")[0].slice(-200)}`]; }
        const feedsFor = (n, q, k, v, bias, useBias) => {
          const f = {
            q: new ort.Tensor("float32", q, [1, n, m.heads * m.head_dim]),
            k: new ort.Tensor("float32", k, [1, n, m.kv_heads * m.head_dim]),
            v: new ort.Tensor("float32", v, [1, n, m.kv_heads * m.head_dim]),
            seqlens_k: new ort.Tensor("int32", Int32Array.from([n - 1]), [1]),
            total: new ort.Tensor("int32", Int32Array.from([n]), []),
          };
          if (s.inputNames.includes("bias")) {
            f.bias = new ort.Tensor("float32", bias ?? new Float32Array(n * n), [1, 1, n, n]);
            f.use_bias = new ort.Tensor("bool", Uint8Array.from([useBias ? 1 : 0]), []);
          }
          return f;
        };
        if (s.inputNames.includes("bias")) {
          try {
            const ref = await bin("ref");
            const { o } = await s.run(feedsFor(m.n, await bin("q"), await bin("k"), await bin("v"), await bin("bias"), true));
            let md = 0;
            for (let i = 0; i < ref.length; i++) md = Math.max(md, Math.abs(o.data[i] - ref[i]));
            out.push(`check (bias branch) n=${m.n} max|Δ| ${md.toExponential(2)}`);
          } catch (e) { out.push(`check failed: ${String(e.message).split("\n")[0].slice(-200)}`); }
        }
        for (const n of lens) {
          const q = Float32Array.from({ length: n * m.heads * m.head_dim }, (_, i) => Math.sin(i * 0.37) * 0.25);
          const kv = Float32Array.from({ length: n * m.kv_heads * m.head_dim }, (_, i) => Math.cos(i * 0.11) * 0.25);
          try {
            await s.run(feedsFor(n, q, kv, kv, null, false));
            const t0 = performance.now();
            await s.run(feedsFor(n, q, kv, kv, null, false));
            out.push(`n=${n} (causal branch) ok ${Math.round(performance.now() - t0)} ms`);
          } catch (e) { out.push(`n=${n} failed: ${String(e.message).split("\n")[0].slice(-200)}`); }
        }
        await s.release();
        return out;
      }, { graph, kind, m, lens: lens.map(Number) });
      console.log(graph, "\n  " + r.join("\n  "));
    }
  }
} finally {
  await browser.close();
  server.close();
}
