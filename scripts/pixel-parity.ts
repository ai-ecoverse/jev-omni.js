// Browser image decoding + src/vision.ts preprocessing against the Gemma 4 processor's pixel values
// (export/jev_omni_web_export/image_refs.py), image by image.
//   node --import tsx scripts/pixel-parity.ts [build/eval/image-refs] [--out report.json]
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { chromium } from "@playwright/test";
import { createServer } from "vite";

const { values: a, positionals } = parseArgs({ options: { out: { type: "string" } }, allowPositionals: true });
const refs = resolve(positionals[0] ?? "build/eval/image-refs");
const index = JSON.parse(readFileSync(`${refs}/index.json`, "utf8")) as { images: Record<string, { levels: string; num_soft_tokens: number }> };
const server = await createServer({ configFile: "test/browser/vite.config.ts", logLevel: "warn" });
await server.listen();
const browser = await chromium.launch({ channel: "chromium", args: ["--enable-unsafe-webgpu"] });
const rows: Record<string, unknown>[] = [];
try {
  const page = await browser.newPage();
  await page.goto(server.resolvedUrls!.local[0]);
  await page.waitForFunction(() => window.jevReady);
  for (const [path, im] of Object.entries(index.images)) {
    const r = await page.evaluate(([i, f]) => window.jevPixelParity(i, f), [`/abs${path}`, `/abs${refs}/${im.levels}`]);
    if (r.numSoftTokens !== im.num_soft_tokens) throw new Error(`${path}: ${r.numSoftTokens} patches, processor ${im.num_soft_tokens}`);
    rows.push({ image: path, ...r });
    console.log(`${path.split("/").slice(-3).join("/")}: ${r.differing}/${r.values} values differ, max ${r.max_levels} levels`);
  }
} finally {
  await browser.close();
  await server.close();
}
const total = rows.reduce((s, r) => s + (r.values as number), 0), diff = rows.reduce((s, r) => s + (r.differing as number), 0);
const summary = { images: rows.length, values: total, differing: diff, fraction: diff / total, max_levels: Math.max(...rows.map((r) => r.max_levels as number)),
  exact_images: rows.filter((r) => r.differing === 0).length };
console.log(JSON.stringify(summary));
if (a.out) writeFileSync(a.out, JSON.stringify({ summary, rows }, null, 1));
