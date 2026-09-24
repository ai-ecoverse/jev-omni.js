// Browser side of scripts/browser-eval.ts: loads a bundle served at /models/ and answers pre-encoded questions.
import * as ort from "onnxruntime-web/webgpu";
import { loadJevOmni, preprocessImage, type ImageLike, type JevOmni, type OrtModule, type Question } from "../../src/index.ts";

ort.env.wasm.wasmPaths = "/ort/";
ort.env.logLevel = "error";
const log = (s: string) => { document.getElementById("log")!.textContent += s + "\n"; };

let jev: JevOmni | null = null;

declare global {
  interface Window {
    jevLoad(variant: string, verify: boolean): Promise<{ ms: number; adapter: unknown }>;
    jevRun(ids: number[], nOptions: number): Promise<{ probs: number[]; hidden: number[]; ms: number }>;
    jevEncode(q: Question): number[];
    jevPredict(q: Question & { image?: string }): Promise<{ probs: number[]; tokens: number; image_tokens: number; embed_ms: number; decoder_ms: number; ms: number }>;
    jevImageFeatures(url: string): Promise<{ data: number[]; numSoftTokens: number; ms: number }>;
    jevPixelParity(imageUrl: string, refUrl: string): Promise<{ values: number; differing: number; max_levels: number; numSoftTokens: number }>;
    jevReady: boolean;
  }
}

window.jevLoad = async (variant, verify) => {
  const t0 = performance.now();
  const a = await navigator.gpu?.requestAdapter();
  let last = 0;
  jev = await loadJevOmni("/models", {
    ort: ort as unknown as OrtModule, variant, verify, cacheName: null, executionProviders: ["webgpu"],
    onPhase: (p) => log(`phase ${p} at ${Math.round(performance.now() - t0)} ms`),
    onProgress: (p) => { if (p.loaded === p.total && performance.now() - last > 5000) { last = performance.now(); log(`${p.file} ${p.total}`); } },
  });
  return { ms: performance.now() - t0, adapter: a && { vendor: a.info.vendor, architecture: a.info.architecture, maxBufferSize: a.limits.maxBufferSize } };
};

window.jevRun = async (ids, nOptions) => {
  const t0 = performance.now();
  const hidden = await jev!.hidden(ids);
  const ms = performance.now() - t0;
  return { probs: jev!.probsFromHidden(hidden, nOptions), hidden: Array.from(hidden), ms };
};

window.jevEncode = (q) => jev!.encode(q);

/** Decode an image as PIL does: no color management, straight (not premultiplied) alpha. */
async function decode(url: string): Promise<ImageLike> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const bmp = await createImageBitmap(await res.blob(), { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const c = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = c.getContext("2d", { colorSpace: "srgb" })!;
  ctx.drawImage(bmp, 0, 0);
  const d = ctx.getImageData(0, 0, bmp.width, bmp.height);
  bmp.close();
  return { width: d.width, height: d.height, data: d.data };
}

window.jevPredict = async (q) => {
  const image = q.image ? await decode(q.image) : undefined;
  const p = await jev!.predict({ state: q.state, question: q.question, options: q.options, image });
  return { probs: p.probs, tokens: p.input_tokens, image_tokens: p.image_tokens, embed_ms: p.embed_ms, decoder_ms: p.decoder_ms, ms: p.latency_ms };
};

window.jevImageFeatures = async (url) => {
  const f = await jev!.imageFeatures(await decode(url));
  return { data: Array.from(f.data), numSoftTokens: f.numSoftTokens, ms: f.ms };
};

window.jevPixelParity = async (imageUrl, refUrl) => {
  const p = preprocessImage(await decode(imageUrl));
  const ref = new Uint8Array(await (await fetch(refUrl)).arrayBuffer());
  const n = ref.length;
  let maxd = 0, diff = 0;
  for (let i = 0; i < n; i++) {
    const d = Math.abs(Math.round(p.pixelValues[i] * 255) - ref[i]);
    if (d) { diff++; if (d > maxd) maxd = d; }
  }
  return { values: n, differing: diff, max_levels: maxd, numSoftTokens: p.numSoftTokens };
};
(window as unknown as { ortVersion: string }).ortVersion = ort.env.versions.web ?? "?";
window.jevReady = true;
