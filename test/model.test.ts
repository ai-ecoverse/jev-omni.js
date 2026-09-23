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
