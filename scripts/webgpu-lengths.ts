// Diagnostic: run a bundle variant in Chromium/WebGPU on random ids of several lengths and report errors/timings.
//   JEV_BUNDLE=... node --import tsx scripts/webgpu-lengths.ts <variant> <len>...
import { chromium } from "@playwright/test";
import { createServer } from "vite";

const [variant, ...lens] = process.argv.slice(2);
const server = await createServer({ configFile: "test/browser/vite.config.ts", logLevel: "warn" });
await server.listen();
const browser = await chromium.launch({ channel: "chromium", args: ["--enable-unsafe-webgpu"] });
try {
  const page = await browser.newPage();
  page.setDefaultTimeout(0);
  page.on("console", (m) => { if (m.type() === "error") console.log("[page]", m.text().slice(0, 300)); });
  await page.goto(server.resolvedUrls!.local[0]);
  await page.waitForFunction(() => window.jevReady);
  await page.evaluate((v) => window.jevLoad(v, false), variant);
  console.log("onnxruntime-web", await page.evaluate(() => (window as unknown as { ortVersion: string }).ortVersion));
  for (const n of lens.map(Number)) {
    const ids = Array.from({ length: n }, (_, i) => 1000 + ((i * 7919) % 200000));
    try {
      const r = await page.evaluate(([ids]) => window.jevRun(ids as number[], 2), [ids]);
      console.log(`len ${n}: ok ${Math.round(r.ms)} ms, hidden[0..2] ${r.hidden.slice(0, 3).map((x) => x.toFixed(4))}`);
    } catch (e) {
      console.log(`len ${n}: ${(e as Error).message.split("\n")[0]}`);
    }
  }
} finally {
  await browser.close();
  await server.close();
}
