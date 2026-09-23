// Browser side of scripts/browser-eval.ts: loads a bundle served at /models/ and answers pre-encoded questions.
import * as ort from "onnxruntime-web/webgpu";
import { loadJevOmni, type JevOmni, type OrtModule, type Question } from "../../src/index.ts";

ort.env.wasm.wasmPaths = "/ort/";
ort.env.logLevel = "error";
const log = (s: string) => { document.getElementById("log")!.textContent += s + "\n"; };

let jev: JevOmni | null = null;

declare global {
  interface Window {
    jevLoad(variant: string, verify: boolean): Promise<{ ms: number; adapter: unknown }>;
    jevRun(ids: number[], nOptions: number): Promise<{ probs: number[]; hidden: number[]; ms: number }>;
    jevEncode(q: Question): number[];
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
(window as unknown as { ortVersion: string }).ortVersion = ort.env.versions.web ?? "?";
window.jevReady = true;
