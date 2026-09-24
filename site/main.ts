import { imagePresets, textPresets, type Preset } from "./presets.ts";
import type { DemoQuestion, WorkerRequest, WorkerResponse } from "./worker.ts";
import { isCached, type ImageLike, type JevManifest, type Prediction } from "../src/index.ts";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

// Weights live on Hugging Face. A dev server serves public/models/ locally; ?models=<url> overrides both.
const HF_BASE = "https://huggingface.co/ai-ecoverse/jev-omni.js/resolve/main/jev-omni";
const params = new URLSearchParams(location.search);
const MODEL_URL = new URL(params.get("models") ?? (import.meta.env.VITE_MODEL_BASE as string | undefined)
  ?? (import.meta.env.DEV ? "models/jev-omni" : HF_BASE), location.href).href;
const GB = (n: number) => (n / 1e9).toFixed(1);

function status(s: string, cls = "") { const el = $("load-status"); el.textContent = s; el.className = cls; }

// ---- WebGPU ----
type GpuNavigator = Navigator & { gpu?: { requestAdapter(): Promise<{ info?: { vendor: string; architecture: string } } | null> } };
const adapter = await (navigator as GpuNavigator).gpu?.requestAdapter().catch(() => null);
if (!adapter) {
  $("demo-body").classList.add("no-gpu");
  status("This browser has no WebGPU adapter, so the demo cannot run here. Use a recent Chrome or Edge on a machine with a GPU (and about 16 GB of GPU memory for this model).", "err");
  $<HTMLButtonElement>("load").disabled = true;
}

// ---- manifest: size and whether this revision is in Cache Storage already ----
let manifest: JevManifest | null = null;
let totalBytes = 0;
try {
  const res = await fetch(`${MODEL_URL}/manifest.json`, { cache: "no-cache" });
  manifest = res.ok ? await res.json() as JevManifest : null;
} catch { manifest = null; }
async function refreshLoadButton(updateStatus = true) {
  if (!manifest) { status(`Could not read the model manifest at ${MODEL_URL}.`, "err"); $<HTMLButtonElement>("load").disabled = true; return; }
  const v = manifest.variants[Object.keys(manifest.variants)[0]];
  const files = [manifest.files.tokenizer, manifest.files.tokenizer_config, manifest.files.head, v.model, ...v.data,
    ...(manifest.vision ? [manifest.vision.model, ...manifest.vision.data] : [])];
  totalBytes = files.reduce((s, f) => s + (manifest!.sizes[f] ?? 0), 0);
  const cached = await isCached(MODEL_URL, manifest);
  $("load").textContent = cached ? "Load (cached)" : `Download ${GB(totalBytes)} GB & load`;
  $("size").textContent = `${GB(totalBytes)} GB`;
  if (adapter && updateStatus) status(cached ? "The weights are in this browser's cache. Loading reads them from disk and uploads them to the GPU." : "Not loaded.");
}
await refreshLoadButton();

// ---- worker ----
const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
const send = (m: WorkerRequest, transfer: Transferable[] = []) => worker.postMessage(m, transfer);
let ready = false, loadStart = 0, nextId = 1;
const pending = new Map<number, { resolve: (p: Prediction) => void; reject: (e: Error) => void }>();
const setBusy = (busy: boolean) => {
  $<HTMLButtonElement>("run").disabled = busy || !ready;
  $<HTMLButtonElement>("load").disabled = busy || ready || !adapter || !manifest;
  document.body.classList.toggle("busy", busy);
};

$("load").onclick = async () => {
  if (!(await isCached(MODEL_URL, manifest!)) &&
      !confirm(`This downloads ${GB(totalBytes)} GB from Hugging Face and keeps it in the browser's cache. The model needs about 16 GB of GPU memory, and the tab holds the files in memory while loading: 32 GB of RAM at the very least, 64 GB or more recommended. Continue?`)) return;
  ready = false; loadStart = performance.now();
  setBusy(true);
  $("progress").classList.add("active");
  $("bar").style.width = "0%";
  status("Starting…");
  send({ type: "load", baseUrl: MODEL_URL, verify: true });
};

$("reset").onclick = async () => {
  if (ready && !confirm("The model stays loaded in this tab; the cache is cleared for the next visit. Continue?")) return;
  for (const k of await caches.keys()) if (k.startsWith("jev-omni")) await caches.delete(k);
  await refreshLoadButton();
  status("Cache cleared. The next load downloads the weights again.");
};

worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
  const m = e.data;
  if (m.type === "progress") {
    $("bar").style.width = `${(100 * m.loaded) / Math.max(m.total, 1)}%`;
    const secs = (performance.now() - loadStart) / 1000, rate = m.loaded / Math.max(secs, 0.001);
    const left = m.total > m.loaded ? ` · about ${Math.ceil((m.total - m.loaded) / Math.max(rate, 1) / 60)} min left` : "";
    status(`${GB(m.loaded)} / ${GB(m.total)} GB · ${(rate / 1e6).toFixed(0)} MB/s${left}`);
  } else if (m.type === "phase") {
    if (m.phase === "manifest") status("Fetching the manifest…");
    else if (m.phase === "session") { $("bar").style.width = "100%"; status("Creating the WebGPU session (uploading 13 GB of weights to the GPU)…"); }
    else if (m.phase === "warmup") status("Compiling shaders…");
  } else if (m.type === "ready") {
    ready = true;
    $("progress").classList.remove("active");
    const limit = Number.isFinite(m.maxTokens) ? ` · prompts up to ${m.maxTokens} tokens` : "";
    status(`Ready · revision ${m.revision}${m.images ? " · text and images" : ""}${limit} · load ${(m.loadMs / 1000).toFixed(1)} s, warm-up ${(m.warmupMs / 1000).toFixed(1)} s`, "ok");
    void refreshLoadButton(false);
    setBusy(false);
  } else if (m.type === "result") {
    pending.get(m.id)?.resolve(m.prediction); pending.delete(m.id);
  } else if (m.type === "error") {
    $("progress").classList.remove("active");
    if (m.id !== undefined) { pending.get(m.id)?.reject(new Error(m.message)); pending.delete(m.id); }
    else {
      status(`Error: ${m.message}. Files downloaded so far stay in the cache; load again to continue from there.`, "err");
      void refreshLoadButton(false);
      setBusy(false);
    }
  }
};

function predict(q: DemoQuestion): Promise<Prediction> {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    send({ type: "run", id, q }, q.image ? [q.image.data.buffer] : []);
  });
}

// ---- inputs ----
let image: ImageLike | null = null;
async function setImage(src: Blob | string, name: string) {
  try {
    const blob = typeof src === "string" ? await (await fetch(src)).blob() : src;
    // no colour management and straight alpha: the pixels as stored, which is what the Hugging Face processor sees
    const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(bitmap, 0, 0);
    const d = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
    bitmap.close();
    image = { width: d.width, height: d.height, data: d.data };
    const thumb = $<HTMLImageElement>("thumb");
    if (thumb.src.startsWith("blob:")) URL.revokeObjectURL(thumb.src);
    thumb.src = typeof src === "string" ? src : URL.createObjectURL(blob);
    $("image-box").classList.add("has-image");
    $("image-name").textContent = `${name} · ${d.width}×${d.height}`;
  } catch (err) {
    clearImage();
    $("image-name").textContent = `Could not read the image: ${(err as Error).message}`;
  }
}
function clearImage() {
  image = null;
  $("image-box").classList.remove("has-image");
  $("image-name").textContent = "No image. Drop one here or pick a file; it goes in front of the state.";
}
$<HTMLInputElement>("image-file").onchange = (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (f) void setImage(f, f.name);
  (e.target as HTMLInputElement).value = "";
};
$("image-clear").onclick = clearImage;
$("image-box").ondragover = (e) => { e.preventDefault(); $("image-box").classList.add("drag"); };
$("image-box").ondragleave = () => $("image-box").classList.remove("drag");
$("image-box").ondrop = (e) => {
  e.preventDefault();
  $("image-box").classList.remove("drag");
  const f = e.dataTransfer?.files[0];
  if (f?.type.startsWith("image/")) void setImage(f, f.name);
};

const presets: Record<string, Preset> = { ...textPresets, ...imagePresets };
const presetSel = $<HTMLSelectElement>("preset");
for (const [group, set] of [["Text", textPresets], ["Image", imagePresets]] as const) {
  const og = document.createElement("optgroup"); og.label = group;
  for (const name of Object.keys(set)) og.append(new Option(name, name));
  presetSel.append(og);
}
async function applyPreset() {
  const p = presets[presetSel.value];
  $<HTMLTextAreaElement>("state").value = p.state;
  $<HTMLInputElement>("question").value = p.question;
  $<HTMLTextAreaElement>("options").value = p.options.join("\n");
  if (p.image) await setImage(p.image, p.imageName ?? "image"); else clearImage();
}
presetSel.onchange = () => void applyPreset();
void applyPreset();

// ---- run ----
function render(p: Prediction, options: string[]) {
  const el = $("answers"); el.innerHTML = "";
  const meta = document.createElement("div"); meta.className = "meta";
  const img = p.image_tokens ? `image ${Math.round(p.embed_ms)} ms (${p.image_tokens} image tokens) · ` : "";
  meta.textContent = `${Math.round(p.latency_ms)} ms · ${img}decoder ${Math.round(p.decoder_ms)} ms · ${p.input_tokens} input tokens`;
  el.append(meta);
  const top = Math.max(...p.probs);
  options.forEach((name, i) => {
    const row = document.createElement("div"); row.className = `opt${p.probs[i] === top ? " top" : ""}`;
    row.innerHTML = `<span class="name"></span><span class="track"><span class="fill"></span></span><span class="val"></span>`;
    const n = row.querySelector(".name") as HTMLElement; n.textContent = name; n.title = name;
    (row.querySelector(".val") as HTMLElement).textContent = p.probs[i].toFixed(3);
    el.append(row);
    requestAnimationFrame(() => { (row.querySelector(".fill") as HTMLElement).style.width = `${p.probs[i] * 100}%`; });
  });
  $("raw").textContent = JSON.stringify(p, null, 2);
}

async function run() {
  const options = $<HTMLTextAreaElement>("options").value.split("\n").map((s) => s.trim()).filter(Boolean);
  const q: DemoQuestion = { state: $<HTMLTextAreaElement>("state").value, question: $<HTMLInputElement>("question").value, options };
  if (options.length < 2 || options.length > 256) { $("answers").innerHTML = `<p class="err">Give 2 to 256 options, one per line.</p>`; return; }
  // a copy: the pixels are transferred to the worker, and the picked image should stay usable for the next run
  if (image) q.image = { width: image.width, height: image.height, data: new Uint8ClampedArray(image.data) };
  setBusy(true);
  $("answers").innerHTML = `<p class="muted">Running…</p>`;
  try {
    render(await predict(q), options);
  } catch (err) {
    const p = document.createElement("p"); p.className = "err"; p.textContent = (err as Error).message; $("answers").replaceChildren(p);
  } finally {
    setBusy(false);
  }
}
$("run").onclick = run;
document.addEventListener("keydown", (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && !$<HTMLButtonElement>("run").disabled) void run(); });

// console and automation access: await jev.predict({ state, question, options }), await jev.runPreset(name)
(window as unknown as { jev: unknown }).jev = {
  predict: (q: DemoQuestion) => predict(q),
  runPreset: async (name: string) => { presetSel.value = name; await applyPreset(); await run(); return JSON.parse($("raw").textContent || "null"); },
  get ready() { return ready; },
};
setBusy(false);
