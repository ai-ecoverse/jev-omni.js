// Writes the browser's image features (decode + src/vision.ts + the vision embedder on WebGPU) for each image to
// <out>/<n>.bin (float32 [N, hidden]) and <out>/index.json, for export/jev_omni_web_export/vision_parity.py.
//   node --import tsx scripts/vision-features.ts --bundle <dir> --variant <v> --out <dir> image...
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { chromium } from "@playwright/test";
import { createServer } from "vite";

const { values: a, positionals } = parseArgs({ options: {
  bundle: { type: "string", default: "public/models/jev-omni" }, variant: { type: "string", default: "q8f32" }, out: { type: "string" },
} , allowPositionals: true });
if (!a.out || !positionals.length) throw new Error("--out and at least one image are required");
mkdirSync(a.out, { recursive: true });
process.env.JEV_BUNDLE = resolve(a.bundle!);
const server = await createServer({ configFile: "test/browser/vite.config.ts", logLevel: "warn" });
await server.listen();
const browser = await chromium.launch({ channel: "chromium", args: ["--enable-unsafe-webgpu"] });
const index: { image: string; file: string; num_soft_tokens: number; ms: number }[] = [];
try {
  const page = await browser.newPage();
  page.setDefaultTimeout(0);
  await page.goto(server.resolvedUrls!.local[0]);
  await page.waitForFunction(() => window.jevReady);
  await page.evaluate((v) => window.jevLoad(v, false), a.variant!);
  for (const [k, img] of positionals.entries()) {
    const f = await page.evaluate((u) => window.jevImageFeatures(u), `/abs${resolve(img)}`);
    const file = `${String(k).padStart(3, "0")}.bin`;
    writeFileSync(`${a.out}/${file}`, new Uint8Array(Float32Array.from(f.data).buffer));
    index.push({ image: resolve(img), file, num_soft_tokens: f.numSoftTokens, ms: f.ms });
    console.log(img, f.numSoftTokens, `${Math.round(f.ms)} ms`);
  }
} finally {
  await browser.close();
  await server.close();
}
writeFileSync(`${a.out}/index.json`, JSON.stringify(index, null, 1));
