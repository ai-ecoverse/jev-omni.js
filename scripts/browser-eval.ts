// Runs an eval set through a bundle in Chromium (WebGPU) and appends one JSON line per question, in the format of
// export/jev_omni_web_export/reference.py, so compare.py scores it. Resumes from an existing output file.
//
//   JEV_BUNDLE=public/models/jev-omni node --import tsx scripts/browser-eval.ts --variant q8f32 \
//     --evalset build/eval/decisionbench-medium.json --out build/eval/browser-q8f32.jsonl [--limit N] [--no-verify]
import { appendFileSync, existsSync, readFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { chromium } from "@playwright/test";
import { createServer } from "vite";

const { values: a } = parseArgs({ options: {
  variant: { type: "string", default: "q8f32" }, evalset: { type: "string" }, out: { type: "string" },
  limit: { type: "string" }, "no-verify": { type: "boolean", default: false }, headed: { type: "boolean", default: false },
} });
if (!a.evalset || !a.out) throw new Error("--evalset and --out are required");

type Rec = { id: string; ids: number[]; options: string[]; state: string; question: string };
const records = (JSON.parse(readFileSync(a.evalset, "utf8")).records as Rec[]).slice(0, a.limit ? Number(a.limit) : undefined);
const done = new Set(existsSync(a.out) ? readFileSync(a.out, "utf8").trim().split("\n").filter(Boolean).map((l) => JSON.parse(l).id) : []);
const todo = records.filter((r) => !done.has(r.id));
console.log(`${done.size} done, ${todo.length} to go`);

const server = await createServer({ configFile: "test/browser/vite.config.ts", logLevel: "warn" });
await server.listen();
const url = server.resolvedUrls!.local[0];
const browser = await chromium.launch({ channel: "chromium", headless: !a.headed, args: ["--enable-unsafe-webgpu"] });
try {
  const page = await browser.newPage();
  page.setDefaultTimeout(0);
  page.on("console", (m) => { if (m.type() === "error" || m.text().startsWith("phase")) console.log("[page]", m.text()); });
  page.on("pageerror", (e) => console.log("[pageerror]", e.message));
  await page.goto(url);
  await page.waitForFunction(() => window.jevReady);
  const loaded = await page.evaluate(([v, verify]) => window.jevLoad(v as string, verify as boolean), [a.variant, !a["no-verify"]]);
  console.log(`loaded ${a.variant} in ${(loaded.ms / 1000).toFixed(1)} s`, loaded.adapter);
  let encodeChecked = 0;
  for (const [i, r] of todo.entries()) {
    if (encodeChecked < 5) {
      const ids = await page.evaluate((q) => window.jevEncode(q), { state: r.state, question: r.question, options: r.options });
      if (JSON.stringify(ids) !== JSON.stringify(r.ids)) throw new Error(`${r.id}: in-browser encoding differs from the eval set`);
      encodeChecked++;
    }
    const res = await page.evaluate(([ids, n]) => window.jevRun(ids as number[], n as number), [r.ids, r.options.length]);
    appendFileSync(a.out, JSON.stringify({ id: r.id, probs: res.probs, hidden: res.hidden, ms: Math.round(res.ms), tokens: r.ids.length }) + "\n");
    console.log(`[${done.size + i + 1}/${records.length}] ${r.id} ${r.ids.length} tok ${Math.round(res.ms)} ms`);
  }
} finally {
  await browser.close();
  await server.close();
}
