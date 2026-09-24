// Jev-Omni runs in a worker: loading 13.6 GB and a multi-second forward pass would otherwise freeze the page.
import * as ort from "onnxruntime-web/webgpu";
import wasm from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url";
import mjs from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url";
import { loadJevOmni, type ImageLike, type JevOmni, type OrtModule, type Prediction } from "../src/index.ts";

ort.env.wasm.wasmPaths = { wasm, mjs };
ort.env.wasm.numThreads = 1;   // everything runs on the GPU; Pages cannot send the headers threads would need
ort.env.logLevel = "error";

export interface DemoQuestion { state: string; question: string; options: string[]; image?: ImageLike }

export type WorkerRequest =
  | { type: "load"; baseUrl: string; verify: boolean }
  | { type: "run"; id: number; q: DemoQuestion };

export type WorkerResponse =
  | { type: "progress"; loaded: number; total: number }
  | { type: "phase"; phase: "manifest" | "download" | "session" | "warmup" }
  | { type: "ready"; loadMs: number; warmupMs: number; revision: string; images: boolean; maxTokens: number }
  | { type: "result"; id: number; prediction: Prediction }
  | { type: "error"; id?: number; message: string };

let jev: JevOmni | null = null;
const post = (m: WorkerResponse) => (self as unknown as Worker).postMessage(m);

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  const m = e.data;
  try {
    if (m.type === "load") {
      await jev?.release(); jev = null;
      // 13.6 GB arrive in ~64 kB chunks: sum them here and post a few times a second, not per chunk
      const files = new Map<string, { loaded: number; total: number }>();
      let last = 0;
      const t0 = performance.now();
      jev = await loadJevOmni(m.baseUrl, {
        ort: ort as unknown as OrtModule, executionProviders: ["webgpu"], verify: m.verify,
        onPhase: (phase) => { if (phase !== "ready") post({ type: "phase", phase }); },
        onProgress: (p) => {
          files.set(p.file, p);
          const now = performance.now();
          if (now - last < 200 && p.loaded < p.total) return;
          last = now;
          let loaded = 0, total = 0;
          for (const f of files.values()) { loaded += f.loaded; total += f.total; }
          post({ type: "progress", loaded, total });
        },
      });
      const t1 = performance.now();
      post({ type: "phase", phase: "warmup" });
      // the first run of each graph compiles its shaders
      await jev.predict({ state: "warm up", question: "Is this a warm-up?", options: ["Yes", "No"] });
      if (jev.supportsImages) {
        await jev.predict({ state: "warm up", question: "Is this a warm-up?", options: ["Yes", "No"],
          image: { width: 64, height: 64, data: new Uint8ClampedArray(64 * 64 * 4).fill(128) } });
      }
      post({ type: "ready", loadMs: t1 - t0, warmupMs: performance.now() - t1, revision: jev.manifest.revision ?? jev.manifest.model.revision, images: jev.supportsImages, maxTokens: jev.maxTokens });
    } else if (m.type === "run") {
      if (!jev) throw new Error("model not loaded");
      post({ type: "result", id: m.id, prediction: await jev.predict(m.q) });
    }
  } catch (err) {
    post({ type: "error", id: m.type === "run" ? m.id : undefined, message: err instanceof Error ? err.message : String(err) });
  }
};
