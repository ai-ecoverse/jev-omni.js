// Load a packaged Jev-Omni bundle (jev_omni_web_export.package): manifest, tokenizer, head, ONNX graph and its
// external weights. Every file is checked against the manifest's size and, unless verify is false, its SHA-256.
// From a URL, files are kept in Cache Storage when available; a bundle already on disk (OPFS, a picked directory, a
// virtual file system) is read in place.

import { Tokenizer } from "@huggingface/tokenizers";
import type { InferenceSession } from "onnxruntime-common";
import { DecisionHead } from "./head.ts";
import { JevOmni, type JevManifest, type OrtModule } from "./model.ts";

export interface Progress { file: string; loaded: number; total: number }

/** Reads one file of a bundle by its path in the bundle ("manifest.json", "r-<rev>/q8f32/model.onnx"). */
export type ReadModelFile = (path: string) => Promise<Uint8Array | ArrayBuffer | Blob>;

/** A base URL (fetched, kept in Cache Storage), a directory holding the bundle, or a function reading a file. */
export type ModelSource = string | FileSystemDirectoryHandle | ReadModelFile;

export type LoadPhase = "manifest" | "download" | "verify" | "session" | "ready";

export interface LoadOptions {
  ort: OrtModule;
  /** variant in manifest.variants; default: the first one listed */
  variant?: string;
  /** default ["webgpu"] */
  executionProviders?: InferenceSession.SessionOptions["executionProviders"];
  /** check SHA-256 of every file (default true) */
  verify?: boolean;
  /** Cache Storage name for URL sources; null disables caching */
  cacheName?: string | null;
  onProgress?: (p: Progress) => void;
  onPhase?: (phase: LoadPhase) => void;
  sessionOptions?: InferenceSession.SessionOptions;
  /** load the vision embedder when the bundle has one (default true); false skips its download for text-only use */
  vision?: boolean;
}

/** Every file a variant loads besides manifest.json, with its size (the vision embedder's last, when included). */
export function modelFiles(manifest: JevManifest, variant = Object.keys(manifest.variants)[0], vision = true): { path: string; bytes: number }[] {
  const v = manifest.variants[variant];
  if (!v) throw new Error(`unknown variant ${variant}; have ${Object.keys(manifest.variants).join(", ")}`);
  const vis = vision && manifest.vision ? [manifest.vision.model, ...manifest.vision.data] : [];
  return [manifest.files.tokenizer, manifest.files.tokenizer_config, manifest.files.head, v.model, ...v.data, ...vis]
    .map((path) => ({ path, bytes: manifest.sizes[path] }));
}

export function directoryReader(dir: FileSystemDirectoryHandle): ReadModelFile {
  return async (path) => {
    const parts = path.split("/");
    let d = dir;
    for (const part of parts.slice(0, -1)) d = await d.getDirectoryHandle(part);
    return (await d.getFileHandle(parts[parts.length - 1])).getFile();
  };
}

const join = (base: string, path: string) => `${base.replace(/\/$/, "")}/${path}`;

async function toBytes(got: Uint8Array | ArrayBuffer | Blob): Promise<Uint8Array> {
  // views and buffers may come from another realm (a VFS over postMessage), so recognise them by shape
  if (ArrayBuffer.isView(got)) return new Uint8Array(got.buffer, got.byteOffset, got.byteLength);
  if (typeof (got as Blob).arrayBuffer === "function") return new Uint8Array(await (got as Blob).arrayBuffer());
  return new Uint8Array(got as ArrayBuffer);
}

/** The body as bytes, with progress. With a known size the buffer is allocated once; a longer body is an error. */
async function readResponse(res: Response, file: string, onProgress?: (p: Progress) => void, total?: number): Promise<Uint8Array> {
  if (!res.body || total === undefined) return new Uint8Array(await res.arrayBuffer());
  const out = new Uint8Array(total);
  let loaded = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (loaded + value.length > total) { await reader.cancel(); throw new Error(`${file}: more than the expected ${total} bytes`); }
    out.set(value, loaded);
    loaded += value.length;
    onProgress?.({ file, loaded, total });
  }
  return loaded === total ? out : out.subarray(0, loaded);
}

export async function sha256Hex(data: Uint8Array): Promise<string> {
  const d = await crypto.subtle.digest("SHA-256", data as BufferSource);
  return Array.from(new Uint8Array(d), (b) => b.toString(16).padStart(2, "0")).join("");
}

function sourceReader(source: ModelSource, o: LoadOptions): (path: string, bytes?: number) => Promise<Uint8Array> {
  if (typeof source === "function" || typeof source !== "string") {
    const read = typeof source === "function" ? source : directoryReader(source);
    return async (path, bytes) => {
      const data = await toBytes(await read(path));
      o.onProgress?.({ file: path, loaded: data.length, total: data.length });
      if (bytes !== undefined && data.length !== bytes) throw new Error(`${path}: expected ${bytes} bytes, read ${data.length} (incomplete download?)`);
      return data;
    };
  }
  return async (path, bytes) => {
    const url = join(source, path);
    const cache = o.cacheName !== null && typeof caches !== "undefined" ? await caches.open(o.cacheName ?? "jev-omni-v1") : null;
    const hit = await cache?.match(url);
    if (hit) {
      const data = await readResponse(hit, path, o.onProgress, bytes);
      if (bytes === undefined || data.length === bytes) return data;
      await cache!.delete(url);
    }
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    const data = await readResponse(res, path, o.onProgress, bytes);
    if (bytes !== undefined && data.length !== bytes) throw new Error(`${path}: expected ${bytes} bytes, received ${data.length}`);
    if (cache && path !== "manifest.json") {
      try { await cache.put(url, new Response(data as BodyInit, { headers: { "content-length": String(data.length) } })); } catch { /* quota: run uncached */ }
    }
    return data;
  };
}

export async function loadJevOmni(source: ModelSource, o: LoadOptions): Promise<JevOmni> {
  const read = sourceReader(source, o);
  o.onPhase?.("manifest");
  const manifest = JSON.parse(new TextDecoder().decode(await read("manifest.json"))) as JevManifest;
  const variant = o.variant ?? Object.keys(manifest.variants)[0];
  const withVision = o.vision !== false && !!manifest.vision;
  const files = modelFiles(manifest, variant, withVision);
  o.onPhase?.("download");
  const data: Uint8Array[] = [];
  for (const f of files) {
    const bytes = await read(f.path, f.bytes);
    if (o.verify !== false) {
      const want = manifest.sha256[f.path];
      if (!want) throw new Error(`${f.path}: no SHA-256 in manifest`);
      const got = await sha256Hex(bytes);
      if (got !== want) throw new Error(`${f.path}: SHA-256 ${got} does not match manifest ${want}`);
    }
    data.push(bytes);
  }
  const v = manifest.variants[variant];
  const [tokJson, tokCfg, headBytes, graph, ...rest] = data;
  const weights = rest.slice(0, v.data.length);
  const [visionGraph, ...visionWeights] = rest.slice(v.data.length);
  const dec = new TextDecoder();
  const tokenizer = new Tokenizer(JSON.parse(dec.decode(tokJson)), JSON.parse(dec.decode(tokCfg)));
  const head = DecisionHead.fromSafetensors(headBytes.slice().buffer);
  if (head.hidden !== manifest.head.hidden) throw new Error(`head hidden size ${head.hidden} != manifest ${manifest.head.hidden}`);
  o.onPhase?.("session");
  const create = (g: Uint8Array, paths: string[], bufs: Uint8Array[]) => o.ort.InferenceSession.create(g, {
    executionProviders: o.executionProviders ?? ["webgpu"],
    graphOptimizationLevel: "all",
    externalData: paths.map((p, i) => ({ path: p.split("/").pop()!, data: bufs[i] })),
    ...o.sessionOptions,
  });
  const session = await create(graph, v.data, weights);
  const vision = withVision ? await create(visionGraph, manifest.vision!.data, visionWeights) : null;
  data.length = 0; weights.length = 0; rest.length = 0;
  o.onPhase?.("ready");
  return new JevOmni({ ort: o.ort, session, head, tokenizer, manifest, variant, vision });
}
