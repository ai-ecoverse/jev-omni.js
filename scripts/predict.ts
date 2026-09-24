// Jev-Omni in Chromium (onnxruntime-web, WebGPU) over a JSONL file of requests, text or image: the browser
// counterpart of export/jev_omni_web_export/predict.py, with the same request and output formats.
//
//   node --import tsx scripts/predict.ts --requests reqs.jsonl --out out.jsonl [--bundle public/models/jev-omni]
//       [--variant q8f32] [--limit N] [--no-verify] [--headed]
//   node --import tsx scripts/predict.ts --image cat.png --state "A photo." --question "Is it a cat?" --options '["Yes","No"]'
//
// Request lines: {"id", "state", "question", "options", "image"?: absolute or relative path}; other fields are passed
// through. Output lines add {"probs", "tokens", "image_tokens", "embed_ms", "decoder_ms", "ms"} (embed_ms: image
// decode excluded, preprocessing + vision embedder included). The output is appended to and resumed from. Without
// --out, results go to stdout. The bundle is loaded once (about 15 s for 13 GB from local disk).
import { appendFileSync, existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { chromium } from "@playwright/test";
import { createServer } from "vite";

const { values: a } = parseArgs({ options: {
  requests: { type: "string" }, out: { type: "string" }, bundle: { type: "string", default: process.env.JEV_BUNDLE ?? "public/models/jev-omni" },
  variant: { type: "string", default: "q8f32" }, limit: { type: "string" }, "no-verify": { type: "boolean", default: false },
  headed: { type: "boolean", default: false }, image: { type: "string" }, state: { type: "string" }, question: { type: "string" },
  options: { type: "string" }, port: { type: "string" },
} });

type Req = { id: string; state: string; question: string; options: string[]; image?: string; [k: string]: unknown };
let reqs: Req[];
if (a.requests) {
  reqs = readFileSync(a.requests, "utf8").split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l) as Req);
} else {
  if (a.state === undefined || !a.question || !a.options) throw new Error("--state, --question and --options are required without --requests");
  reqs = [{ id: "q", state: a.state, question: a.question, options: JSON.parse(a.options) as string[], image: a.image }];
}
reqs = reqs.slice(0, a.limit ? Number(a.limit) : undefined);
const done = new Set(a.out && existsSync(a.out) ? readFileSync(a.out, "utf8").split("\n").filter(Boolean).map((l) => (JSON.parse(l) as Req).id) : []);
const todo = reqs.filter((r) => !done.has(r.id));
console.error(`${done.size} done, ${todo.length} to go`);
if (!todo.length) process.exit(0);

process.env.JEV_BUNDLE = resolve(a.bundle!);
if (a.port) process.env.JEV_PORT = a.port;
const server = await createServer({ configFile: "test/browser/vite.config.ts", logLevel: "warn" });
await server.listen();
const url = server.resolvedUrls!.local[0];
const browser = await chromium.launch({ channel: "chromium", headless: !a.headed, args: ["--enable-unsafe-webgpu"] });
try {
  const page = await browser.newPage();
  page.setDefaultTimeout(0);
  page.on("console", (m) => { if (m.type() === "error") console.error("[page]", m.text().slice(0, 300)); });
  page.on("pageerror", (e) => console.error("[pageerror]", e.message));
  await page.goto(url);
  await page.waitForFunction(() => window.jevReady);
  const loaded = await page.evaluate(([v, verify]) => window.jevLoad(v as string, verify as boolean), [a.variant, !a["no-verify"]]);
  console.error(`loaded ${a.variant} in ${(loaded.ms / 1000).toFixed(1)} s`, loaded.adapter);
  for (const [i, r] of todo.entries()) {
    const image = r.image ? `/abs${resolve(r.image)}` : undefined;
    const res = await page.evaluate((q) => window.jevPredict(q), { state: r.state, question: r.question, options: r.options, image });
    const row = JSON.stringify({ ...r, ...res, variant: a.variant });
    if (a.out) appendFileSync(a.out, row + "\n");
    else console.log(row);
    const top = res.probs.indexOf(Math.max(...res.probs));
    console.error(`[${done.size + i + 1}/${reqs.length}] ${r.id} ${res.tokens} tok embed ${Math.round(res.embed_ms)} ms decoder ${Math.round(res.decoder_ms)} ms -> ${JSON.stringify(r.options[top])} p=${res.probs[top].toFixed(3)}${"label" in r ? ` label ${r.label}` : ""}`);
  }
} finally {
  await browser.close();
  await server.close();
}
