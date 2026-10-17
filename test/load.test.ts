// loadJevOmni's local sources against a synthetic bundle and stub runtimes: graphs opened by path (filePath, as on
// onnxruntime-node) and weights read on demand (readSync, as onnxruntime-web asks for them), and the plain read path.
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import * as ort from "onnxruntime-node";
import { loadJevOmni, sha256Hex, type OrtModule, type Progress, type ReadModelFile } from "../src/index.ts";

const enc = (s: string) => new TextEncoder().encode(s);
const text = (b: Uint8Array) => new TextDecoder().decode(b);

/** A safetensors file with a 3-class head over 2 hidden values. */
function headFile(): Uint8Array {
  const t = (off: number, shape: number[]) => ({ dtype: "F32", shape, data_offsets: [off * 4, (off + shape.reduce((a, b) => a * b)) * 4] });
  const header = enc(JSON.stringify({ mu: t(0, [2]), sd: t(2, [2]), "linear.weight": t(4, [3, 2]), "linear.bias": t(10, [3]) }));
  const out = new Uint8Array(8 + header.length + 13 * 4);
  new DataView(out.buffer).setBigUint64(0, BigInt(header.length), true);
  out.set(header, 8);
  new Float32Array(out.buffer, 8 + header.length + 8, 2).set([1, 1]);
  return out;
}

const TOKENIZER = JSON.stringify({
  version: "1.0", truncation: null, padding: null, added_tokens: [], normalizer: null,
  pre_tokenizer: { type: "Whitespace" }, post_processor: null, decoder: null,
  model: { type: "WordLevel", vocab: { "[UNK]": 0, a: 1 }, unk_token: "[UNK]" },
});

/** bundle path -> bytes, manifest.json included: a decoder with two weight files and a vision embedder with one. */
async function bundle(edit?: (files: Map<string, Uint8Array>) => void): Promise<Map<string, Uint8Array>> {
  const files = new Map<string, Uint8Array>([
    ["r-abc/tokenizer.json", enc(TOKENIZER)], ["r-abc/tokenizer_config.json", enc("{}")], ["r-abc/head.safetensors", headFile()],
    ["r-abc/q8/model.onnx", enc("graph")], ["r-abc/q8/model.onnx.data", enc("shard zero")], ["r-abc/q8/model.onnx.data_1", enc("shard one!")],
    ["r-abc/vision/vision.onnx", enc("vision")], ["r-abc/vision/vision.onnx.data", enc("eyes")],
  ]);
  const sizes = Object.fromEntries([...files].map(([p, b]) => [p, b.length]));
  const sha256 = Object.fromEntries(await Promise.all([...files].map(async ([p, b]) => [p, await sha256Hex(b)])));
  edit?.(files);
  const manifest = {
    name: "jev-test", revision: "abc", model: { repo: "x/jev", revision: "abc" }, base: { repo: "x/base", revision: "abc" },
    files: { tokenizer: "r-abc/tokenizer.json", tokenizer_config: "r-abc/tokenizer_config.json", head: "r-abc/head.safetensors" },
    vision: { model: "r-abc/vision/vision.onnx", data: ["r-abc/vision/vision.onnx.data"], inputs: [], outputs: [] },
    head: { hidden: 2, classes: 3 }, sizes, sha256,
    variants: { q8: { model: "r-abc/q8/model.onnx", data: ["r-abc/q8/model.onnx.data", "r-abc/q8/model.onnx.data_1"], inputs: [], outputs: [] } },
  };
  files.set("manifest.json", enc(JSON.stringify(manifest)));
  return files;
}

interface Created { graph: Uint8Array | string; options: { externalData?: { path: string; data: Uint8Array }[] } }

/** An ort whose sessions record what they were created from. With `ranges`, it reads every external-data file the way
 * onnxruntime-web does while it creates a session: byteLength, then subarray(start, end) per tensor, and it answers
 * anything those throw with its own error. */
function stubOrt(ranges?: Record<string, [number, number][]>) {
  const created: Created[] = [];
  const seen: { path: string; isBytes: boolean; size: number; parts: string[] }[] = [];
  const create = async (graph: Uint8Array | string, options: Created["options"]) => {
    created.push({ graph, options });
    if (ranges) for (const { path, data } of options.externalData ?? []) {
      const parts: string[] = [];
      try {
        for (const [start, end] of ranges[path] ?? [[0, data.byteLength]]) parts.push(text(data.subarray(start, end)));
      } catch {
        throw new Error("Failed to load external data file");
      }
      seen.push({ path, isBytes: data instanceof Uint8Array, size: data.byteLength, parts });
    }
    return { inputNames: ["input_ids"], release: async () => {} };
  };
  return { ort: { InferenceSession: { create }, Tensor: ort.Tensor } as unknown as OrtModule, created, seen };
}

function reader(files: Map<string, Uint8Array>) {
  const reads: string[] = [];
  const read: ReadModelFile = async (p) => {
    reads.push(p);
    const b = files.get(p);
    if (!b) throw new Error(`no ${p}`);
    return b;
  };
  return { read, reads };
}

test("filePath hands ORT both graphs' paths and reads no graph or weight", async () => {
  const files = await bundle();
  const { read, reads } = reader(files);
  const { ort: stub, created } = stubOrt();
  const progress: Progress[] = [];
  await loadJevOmni(read, { ort: stub, executionProviders: ["cpu"], filePath: (p) => `/models/jev/${p}`, onProgress: (p) => progress.push(p) });
  assert.deepEqual(created.map((c) => c.graph), ["/models/jev/r-abc/q8/model.onnx", "/models/jev/r-abc/vision/vision.onnx"]);
  assert.ok(created.every((c) => c.options.externalData === undefined));
  assert.deepEqual(reads.sort(), ["manifest.json", "r-abc/head.safetensors", "r-abc/tokenizer.json", "r-abc/tokenizer_config.json"]);
  const done = new Set(progress.filter((p) => p.loaded === p.total).map((p) => p.file));
  for (const p of files.keys()) if (p !== "manifest.json") assert.ok(done.has(p), `no progress for ${p}`);
});

test("readSync hands ORT weights that read just the bytes it asks for, when it asks", async () => {
  const files = await bundle();
  const { read, reads } = reader(files);
  const syncReads: [string, number, number][] = [];
  const readSync = (p: string, offset: number, length: number) => {
    syncReads.push([p, offset, length]);
    const b = files.get(p)!;
    const padded = new Uint8Array(b.length + 8); padded.set(b, 4);
    return padded.subarray(4 + offset, 4 + Math.min(b.length, offset + length));
  };
  const { ort: stub, created, seen } = stubOrt({ "model.onnx.data": [[0, 5], [6, 10]], "model.onnx.data_1": [[6, 10], [0, 5]] });
  const progress: Progress[] = [];
  await loadJevOmni(read, { ort: stub, executionProviders: ["wasm"], readSync, onProgress: (p) => progress.push(p) });
  assert.ok(!reads.some((p) => p.includes(".data")), "no weight file is read before the session");
  assert.deepEqual(created.map((c) => text(c.graph as Uint8Array)), ["graph", "vision"]);
  assert.deepEqual(syncReads, [
    ["r-abc/q8/model.onnx.data", 0, 5], ["r-abc/q8/model.onnx.data", 6, 4],
    ["r-abc/q8/model.onnx.data_1", 6, 4], ["r-abc/q8/model.onnx.data_1", 0, 5],
    ["r-abc/vision/vision.onnx.data", 0, 4],
  ]);
  assert.deepEqual(seen, [
    { path: "model.onnx.data", isBytes: true, size: 10, parts: ["shard", "zero"] },
    { path: "model.onnx.data_1", isBytes: true, size: 10, parts: ["one!", "shard"] },
    { path: "vision.onnx.data", isBytes: true, size: 4, parts: ["eyes"] },
  ]);
  assert.deepEqual(progress.filter((p) => p.file.endsWith("data_1")).map((p) => p.loaded), [0, 4, 9, 10]);
});

test("readSync: a weight file cut short or a failing read fails the load with its own error", async () => {
  const short = await bundle((files) => files.set("r-abc/q8/model.onnx.data_1", enc("shard")));
  await assert.rejects(
    loadJevOmni(reader(short).read, { ort: stubOrt({}).ort, readSync: (p, at, n) => short.get(p)!.subarray(at, at + n) }),
    /model\.onnx\.data_1: expected 10 bytes, but it ends before byte 10/,
  );
  const files = await bundle();
  await assert.rejects(
    loadJevOmni(reader(files).read, { ort: stubOrt({}).ort, readSync: () => { throw new DOMException("the handle is closed", "InvalidStateError"); } }),
    (e: Error) => e.name === "InvalidStateError",
  );
  await assert.rejects(loadJevOmni(reader(files).read, { ort: stubOrt({}).ort, readSync: () => { throw "no handle"; } }), /^Error: no handle$/);
});

test("without filePath or readSync every file is read, checked and handed over as bytes", async () => {
  const files = await bundle();
  const { read, reads } = reader(files);
  const { ort: stub, created } = stubOrt();
  await loadJevOmni(read, { ort: stub, vision: false });
  assert.equal(reads.length, 7);
  assert.deepEqual(created[0].options.externalData!.map((x) => [x.path, text(x.data)]), [["model.onnx.data", "shard zero"], ["model.onnx.data_1", "shard one!"]]);
  const bad = await bundle((f) => f.set("r-abc/q8/model.onnx.data", enc("shard ZERO")));
  await assert.rejects(loadJevOmni(reader(bad).read, { ort: stubOrt().ort }), /model\.onnx\.data: SHA-256 .* does not match/);
});

test("a URL source ignores filePath and readSync", async (t) => {
  const files = await bundle();
  t.mock.method(globalThis, "fetch", async (input: string | URL | Request) => new Response(files.get(String(input).replace("https://example.test/b/", "")) as BodyInit));
  const { ort: stub, created } = stubOrt();
  await loadJevOmni("https://example.test/b", { ort: stub, cacheName: null, vision: false, filePath: (p) => `/nowhere/${p}`, readSync: () => { throw new Error("read"); } });
  assert.equal(text(created[0].graph as Uint8Array), "graph");
  assert.equal(created[0].options.externalData!.length, 2);
});

const BUNDLE = process.env.JEV_BUNDLE ?? "";
test("the published bundle opened by path on onnxruntime-node answers as when read whole", { skip: !existsSync(`${BUNDLE}/manifest.json`) && "set JEV_BUNDLE" }, async () => {
  const q = { state: "The meeting starts at 10 AM. It is now 9 AM.", question: "Has the meeting started?", options: ["Yes", "No"] };
  const o = { ort: ort as unknown as OrtModule, executionProviders: ["cpu"], vision: false };
  const byPath = await loadJevOmni((p) => readFile(`${BUNDLE}/${p}`), { ...o, filePath: (p) => `${BUNDLE}/${p}` });
  const a = await byPath.predict(q);
  await byPath.release();
  const whole = await loadJevOmni((p) => readFile(`${BUNDLE}/${p}`), { ...o, verify: false });
  const b = await whole.predict(q);
  await whole.release();
  assert.equal(a.prediction, "No");
  assert.deepEqual(a.probs, b.probs);
});
