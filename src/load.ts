// Load a packaged Jev-Omni bundle (jev_omni_web_export.package): manifest, tokenizer, head, ONNX graph and its
// external weights. Every file is checked against the manifest's size and, unless verify is false, its SHA-256.
// From a URL, files are kept in Cache Storage keyed by the manifest's content digest (`revision`), and other
// revisions are evicted first; a bundle already on disk (OPFS, a picked directory, a VFS) is read in place.

import { Tokenizer } from "@huggingface/tokenizers";
import type { InferenceSession } from "onnxruntime-common";
import { DecisionHead } from "./head.ts";
import { JevOmni, type JevManifest, type OrtModule } from "./model.ts";

export interface Progress { file: string; loaded: number; total: number }

/** Reads one file of a bundle by its path in the bundle ("manifest.json", "r-<rev>/q8f32/model.onnx"). */
export type ReadModelFile = (path: string) => Promise<Uint8Array | ArrayBuffer | Blob>;

/** A base URL (fetched, kept in Cache Storage), a directory holding the bundle, or a function reading a file. */
export type ModelSource = string | FileSystemDirectoryHandle | ReadModelFile;

export type LoadPhase = "manifest" | "download" | "session" | "ready";

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
  /** files fetched at once from a URL (default 4) */
  concurrency?: number;
  /** longest prompt to accept, in tokens; default: what the WebGPU adapter's buffer limit allows (webgpuTokenLimit) */
  maxTokens?: number;
}

/** Attention heads of Jev-Omni's decoder (Gemma 4 12B). */
const ATTENTION_HEADS = 16;

/** The longest prompt the WebGPU adapter can run, or undefined without WebGPU. onnxruntime-web 1.30 computes attention
 * with a full n×n fp32 score buffer per head, so heads·n²·4 bytes must fit in one storage buffer; past that, OrtRun
 * fails with "Integer overflow" (measured: 8173 tokens run, 8937 fail, with Apple's 4 GiB limit). */
export async function webgpuTokenLimit(): Promise<number | undefined> {
  type Adapter = { limits: { maxStorageBufferBindingSize: number; maxBufferSize: number } };
  const gpu = (globalThis.navigator as unknown as { gpu?: { requestAdapter(): Promise<Adapter | null> } } | undefined)?.gpu;
  const adapter = await gpu?.requestAdapter().catch(() => null);
  if (!adapter) return undefined;
  const bytes = Math.min(adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize);
  return Math.floor(Math.sqrt(bytes / (ATTENTION_HEADS * 4)));
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

/** The Cache Storage key for one file of one bundle revision. Files are republished under the same URLs when only a
 * graph changes, so the key carries the manifest's content digest: a new bundle never reads an old one's bytes. */
const cacheKey = (url: string, rev: string) => `${url}${url.includes("?") ? "&" : "?"}jev-rev=${encodeURIComponent(rev)}`;
const DEFAULT_CACHE = "jev-omni-v1";

/** Delete this bundle's cached files from other revisions, before fetching: 13.6 GB of quota fits one bundle, not two. */
async function dropOtherRevisions(baseUrl: string, rev: string, cacheName?: string | null) {
  if (cacheName === null || typeof caches === "undefined") return;
  const cache = await caches.open(cacheName ?? DEFAULT_CACHE);
  const prefix = `${baseUrl.replace(/\/$/, "")}/`;
  for (const req of await cache.keys()) {
    if (req.url.startsWith(prefix) && new URL(req.url).searchParams.get("jev-rev") !== rev) await cache.delete(req);
  }
}

/** Whether every file of this revision is in Cache Storage already, so a load will not download. */
export async function isCached(baseUrl: string, manifest: JevManifest, o: { variant?: string; vision?: boolean; cacheName?: string | null } = {}): Promise<boolean> {
  if (o.cacheName === null || typeof caches === "undefined") return false;
  const cache = await caches.open(o.cacheName ?? DEFAULT_CACHE);
  const rev = manifest.revision ?? manifest.model.revision;
  for (const f of modelFiles(manifest, o.variant, o.vision !== false && !!manifest.vision)) {
    if (!(await cache.match(cacheKey(join(baseUrl, f.path), rev)))) return false;
  }
  return true;
}

async function checkSha(path: string, data: Uint8Array, want: string | undefined) {
  if (!want) throw new Error(`${path}: no SHA-256 in manifest`);
  const got = await sha256Hex(data);
  if (got !== want) throw new Error(`${path}: SHA-256 ${got} does not match manifest ${want}`);
}

/** Run jobs with a bounded number in flight: hundreds of parallel requests are slower than a handful. */
async function pool<T>(jobs: (() => Promise<T>)[], limit: number): Promise<T[]> {
  const out = new Array<T>(jobs.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(limit, jobs.length) }, async () => {
    for (let i = next++; i < jobs.length; i = next++) out[i] = await jobs[i]();
  }));
  return out;
}

type FileReader = (path: string, bytes?: number, sha?: string) => Promise<Uint8Array>;

function localReader(source: FileSystemDirectoryHandle | ReadModelFile, o: LoadOptions): FileReader {
  const read = typeof source === "function" ? source : directoryReader(source);
  return async (path, bytes, sha) => {
    const data = await toBytes(await read(path));
    o.onProgress?.({ file: path, loaded: data.length, total: data.length });
    if (bytes !== undefined && data.length !== bytes) throw new Error(`${path}: expected ${bytes} bytes, read ${data.length} (incomplete download?)`);
    if (sha !== undefined && o.verify !== false) await checkSha(path, data, sha);
    return data;
  };
}

/** Fetches files of one revision, from Cache Storage when there. A fetched file is checked (size, and SHA-256 unless
 * verify is false) before it is cached, so a cache hit of the right size is trusted without hashing it again. */
function urlReader(baseUrl: string, rev: string, o: LoadOptions): FileReader {
  return async (path, bytes, sha) => {
    const url = join(baseUrl, path), key = cacheKey(url, rev);
    const cache = o.cacheName !== null && typeof caches !== "undefined" ? await caches.open(o.cacheName ?? DEFAULT_CACHE) : null;
    const hit = await cache?.match(key);
    if (hit) {
      const data = await readResponse(hit, path, o.onProgress, bytes);
      if (bytes === undefined || data.length === bytes) return data;
      await cache!.delete(key);   // truncated entry: fetch it again
    }
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    const data = await readResponse(res, path, o.onProgress, bytes);
    if (bytes !== undefined && data.length !== bytes) throw new Error(`${path}: expected ${bytes} bytes, received ${data.length}`);
    if (sha !== undefined && o.verify !== false) await checkSha(path, data, sha);
    if (cache) {
      try { await cache.put(key, new Response(data as BodyInit, { headers: { "content-length": String(data.length) } })); } catch { /* quota: run uncached */ }
    }
    return data;
  };
}

export async function loadJevOmni(source: ModelSource, o: LoadOptions): Promise<JevOmni> {
  const baseUrl = typeof source === "string" ? source : null;
  o.onPhase?.("manifest");
  const dec = new TextDecoder();
  let manifestBytes: Uint8Array;
  if (baseUrl !== null) {
    // always fresh: it names the revision everything else is cached under
    const res = await fetch(join(baseUrl, "manifest.json"), { cache: "no-cache" });
    if (!res.ok) throw new Error(`${join(baseUrl, "manifest.json")}: HTTP ${res.status}`);
    manifestBytes = new Uint8Array(await res.arrayBuffer());
  } else {
    manifestBytes = await localReader(source as FileSystemDirectoryHandle | ReadModelFile, { ...o, onProgress: undefined })("manifest.json");
  }
  const manifest = JSON.parse(dec.decode(manifestBytes)) as JevManifest;
  const variant = o.variant ?? Object.keys(manifest.variants)[0];
  const withVision = o.vision !== false && !!manifest.vision;
  const files = modelFiles(manifest, variant, withVision);
  const rev = manifest.revision ?? manifest.model.revision;
  if (baseUrl !== null) await dropOtherRevisions(baseUrl, rev, o.cacheName);
  const read = baseUrl !== null ? urlReader(baseUrl, rev, o) : localReader(source as FileSystemDirectoryHandle | ReadModelFile, o);
  o.onPhase?.("download");
  // announce every file up front so the total does not grow as downloads start
  for (const f of files) o.onProgress?.({ file: f.path, loaded: 0, total: f.bytes });
  const data = await pool(files.map((f) => () => read(f.path, f.bytes, manifest.sha256[f.path])), o.concurrency ?? 4);
  const v = manifest.variants[variant];
  const [tokJson, tokCfg, headBytes, graph, ...rest] = data;
  const weights = rest.slice(0, v.data.length);
  const [visionGraph, ...visionWeights] = rest.slice(v.data.length);
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
  const gpu = (o.executionProviders ?? ["webgpu"]).some((e) => (typeof e === "string" ? e : e.name) === "webgpu");
  const maxTokens = o.maxTokens ?? (gpu ? await webgpuTokenLimit() : undefined);
  return new JevOmni({ ort: o.ort, session, head, tokenizer, manifest, variant, vision, maxTokens });
}
