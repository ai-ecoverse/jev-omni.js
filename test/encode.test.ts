import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { Tokenizer } from "@huggingface/tokenizers";
import { chatText, pyStrip, renderPrompt } from "../src/prompt.ts";

const SNAPSHOT = process.env.JEV_SNAPSHOT ?? "build/Jev-Omni";
const EVALSET = process.env.JEV_EVALSET ?? "build/eval/decisionbench-medium.json";

test("pyStrip matches Python str.strip", () => {
  assert.equal(pyStrip(" \t\n\x1c\x85a b\u3000\u2029"), "a b");
  assert.equal(pyStrip("\ufeffx\ufeff"), "\ufeffx\ufeff");   // not whitespace to Python
  assert.equal(pyStrip("   "), "");
});

test("renderPrompt is load_model.predict's prompt", () => {
  assert.equal(renderPrompt({ state: "S", question: "Q?", options: ["a", "b"] }),
    "S\n\n---\n\nQUESTION: Q?\n\nOPTIONS:\n1. a\n2. b\n\nReply with only the number of the correct option (1-2).\nOutput a single number and nothing else.");
});

test("token ids match the Python chat template on every eval question", { skip: !existsSync(EVALSET) && `no ${EVALSET}` }, () => {
  const tok = new Tokenizer(JSON.parse(readFileSync(`${SNAPSHOT}/tokenizer.json`, "utf8")), JSON.parse(readFileSync(`${SNAPSHOT}/tokenizer_config.json`, "utf8")));
  const { records } = JSON.parse(readFileSync(EVALSET, "utf8")) as { records: { id: string; state: string; question: string; options: string[]; ids: number[] }[] };
  let n = 0;
  for (const r of records) {
    const ids = tok.encode(chatText(renderPrompt(r)), { add_special_tokens: false }).ids;
    assert.deepEqual(ids, r.ids, r.id);
    n++;
  }
  assert.ok(n > 0);
});
