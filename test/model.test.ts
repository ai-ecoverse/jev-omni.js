// loadJevOmni + JevOmni on onnxruntime-node against a bundle on disk, compared with the Python runs of the same
// graph (ort_eval.py) or PyTorch (reference.py). Point JEV_BUNDLE / JEV_VARIANT / JEV_RUNS at a bundle and a JSON-lines
// file with hidden states and probabilities; skipped when absent.
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import * as ort from "onnxruntime-node";
import { loadJevOmni, type OrtModule } from "../src/index.ts";

const BUNDLE = process.env.JEV_BUNDLE ?? "/tmp/jev-tiny/pub";
const VARIANT = process.env.JEV_VARIANT ?? "q8";
const RUNS = process.env.JEV_RUNS ?? "/tmp/jev-tiny/ort-q8.jsonl";
const EVALSET = process.env.JEV_EVALSET ?? "build/eval/decisionbench-medium.json";
const TOL_HIDDEN = Number(process.env.JEV_TOL_HIDDEN ?? 1e-3);
const TOL_P = Number(process.env.JEV_TOL_P ?? 1e-4);
const ready = existsSync(`${BUNDLE}/manifest.json`) && existsSync(RUNS) && existsSync(EVALSET);

test("hidden states and probabilities match the Python run", { skip: !ready && "no bundle/runs" }, async () => {
  const jev = await loadJevOmni((p) => readFile(`${BUNDLE}/${p}`), { ort: ort as unknown as OrtModule, variant: VARIANT, executionProviders: ["cpu"] });
  const records = new Map((JSON.parse(readFileSync(EVALSET, "utf8")).records as { id: string; ids: number[]; options: string[]; state: string; question: string }[]).map((r) => [r.id, r]));
  const runs = readFileSync(RUNS, "utf8").trim().split("\n").map((l) => JSON.parse(l) as { id: string; hidden: number[]; probs: number[] }).slice(0, 4);
  for (const run of runs) {
    const r = records.get(run.id)!;
    assert.deepEqual(jev.encode(r), r.ids);
    const h = await jev.hidden(r.ids);
    const scale = Math.max(...run.hidden.map(Math.abs));
    const dh = Math.max(...Array.from(h, (v, i) => Math.abs(v - run.hidden[i])));
    assert.ok(dh / scale < TOL_HIDDEN, `${run.id}: hidden rel diff ${dh / scale}`);
    const pred = await jev.predict(r);
    const dp = Math.max(...pred.probs.map((p, i) => Math.abs(p - run.probs[i])));
    assert.ok(dp < TOL_P, `${run.id}: max |dp| ${dp}`);
  }
  await jev.release();
});

/** fetch over the bundle directory; `fail(path, attempt)` returns an error to throw or an HTTP status to answer with. */
function bundleFetch(fail: (path: string, attempt: number) => Error | number | undefined) {
  const attempts = new Map<string, number>();
  const fetch = async (input: string | URL | Request) => {
    const path = String(input).replace("https://example.test/b/", "");
    const n = (attempts.get(path) ?? 0) + 1;
    attempts.set(path, n);
    const f = fail(path, n);
    if (f instanceof Error) throw f;
    if (typeof f === "number") return new Response(null, { status: f });
    return new Response(await readFile(`${BUNDLE}/${path}`));
  };
  return { fetch: fetch as typeof globalThis.fetch, attempts };
}

test("a URL load retries files whose download fails", { skip: !existsSync(`${BUNDLE}/manifest.json`) && "no bundle" }, async (t) => {
  const f = bundleFetch((path, n) => (path !== "manifest.json" && n === 1 ? new TypeError("Failed to fetch") : undefined));
  t.mock.method(globalThis, "fetch", f.fetch);
  const jev = await loadJevOmni("https://example.test/b", { ort: ort as unknown as OrtModule, variant: VARIANT, executionProviders: ["cpu"], vision: false });
  await jev.release();
  const files = [...f.attempts].filter(([p]) => p !== "manifest.json");
  assert.ok(files.length > 3);
  assert.ok(files.every(([, n]) => n === 2), JSON.stringify(files));
});

test("a URL load fails at once on a missing file", { skip: !existsSync(`${BUNDLE}/manifest.json`) && "no bundle" }, async (t) => {
  const f = bundleFetch((path) => (path.endsWith("head.safetensors") ? 404 : undefined));
  t.mock.method(globalThis, "fetch", f.fetch);
  await assert.rejects(loadJevOmni("https://example.test/b", { ort: ort as unknown as OrtModule, variant: VARIANT, executionProviders: ["cpu"], vision: false }), /HTTP 404/);
  assert.equal([...f.attempts].find(([p]) => p.endsWith("head.safetensors"))?.[1], 1);
});
