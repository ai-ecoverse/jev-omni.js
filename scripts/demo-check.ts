// Drives the demo page end to end in a real Chrome: load the model (download or cache), run presets, and compare the
// probabilities with fixtures (scripts/predict.ts or predict.py output for the same presets).
//
//   node --import tsx scripts/demo-check.ts --url https://ai-ecoverse.github.io/jev-omni.js/ \
//     --preset "Meeting status" --preset "Image: photo (coffee)" [--fixtures build/eval/demo-presets-webgpu.jsonl]
//     [--profile build/chrome-demo] [--channel chrome] [--headed]
//
// The profile directory keeps Cache Storage between runs, so a second run measures a cached load.
import { existsSync, readFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { chromium } from "@playwright/test";

const { values: a } = parseArgs({ options: {
  url: { type: "string", default: "https://ai-ecoverse.github.io/jev-omni.js/" },
  preset: { type: "string", multiple: true, default: ["Meeting status", "Image: photo (coffee)"] },
  fixtures: { type: "string" }, profile: { type: "string", default: "build/chrome-demo" },
  channel: { type: "string", default: process.env.JEV_BROWSER_CHANNEL ?? "chrome" }, headed: { type: "boolean", default: false },
} });

type Fixture = { id: string; probs: number[] };
const fixtures = new Map<string, Fixture>();
if (a.fixtures && existsSync(a.fixtures)) {
  for (const l of readFileSync(a.fixtures, "utf8").split("\n").filter(Boolean)) { const f = JSON.parse(l) as Fixture; fixtures.set(f.id, f); }
}

const ctx = await chromium.launchPersistentContext(a.profile!, { channel: a.channel, headless: !a.headed, args: ["--enable-unsafe-webgpu"] });
const page = ctx.pages()[0] ?? await ctx.newPage();
page.setDefaultTimeout(0);
page.on("dialog", (d) => void d.accept());   // the size warning before a download
page.on("pageerror", (e) => console.error("[pageerror]", e.message));
page.on("console", (m) => { if (m.type() === "error") console.error("[console]", m.text()); });
try {
  await page.goto(a.url!);
  await page.waitForFunction(() => !!(window as unknown as { jev?: unknown }).jev);
  const loadLabel = await page.locator("#load").textContent();
  const statusBefore = await page.locator("#load-status").textContent();
  console.log(`page loaded: button "${loadLabel}", status "${statusBefore}"`);
  const t0 = Date.now();
  const progress = setInterval(async () => {
    const s = await page.locator("#load-status").textContent().catch(() => "");
    console.log(`  ${Math.round((Date.now() - t0) / 1000)} s: ${s}`);
  }, 30_000);
  await page.locator("#load").click();
  await page.waitForFunction(() => (window as unknown as { jev: { ready: boolean } }).jev.ready || document.getElementById("load-status")!.classList.contains("err"));
  clearInterval(progress);
  const status = await page.locator("#load-status").textContent();
  console.log(`load finished after ${((Date.now() - t0) / 1000).toFixed(1)} s: ${status}`);
  if (await page.locator("#load-status.err").count()) throw new Error(status ?? "load failed");
  for (const name of a.preset!) {
    const t = Date.now();
    const p = await page.evaluate((n) => (window as unknown as { jev: { runPreset(n: string): Promise<{ probs: number[]; latency_ms: number; embed_ms: number; decoder_ms: number; input_tokens: number; image_tokens: number; prediction: string }> } }).jev.runPreset(n), name);
    const f = fixtures.get(name);
    const maxDp = f ? Math.max(...p.probs.map((x, i) => Math.abs(x - f.probs[i]))) : undefined;
    const row = { preset: name, prediction: p.prediction, probs: p.probs.map((x) => +x.toFixed(4)), latency_ms: p.latency_ms, embed_ms: p.embed_ms,
      decoder_ms: p.decoder_ms, tokens: p.input_tokens, image_tokens: p.image_tokens, wall_ms: Date.now() - t, max_abs_dp_vs_fixture: maxDp };
    console.log(JSON.stringify(row));
  }
} finally {
  await ctx.close();
}
