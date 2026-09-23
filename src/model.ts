// Jev-Omni in onnxruntime-web: one forward pass over the chat-templated prompt, the last position's hidden state,
// then the 256-way head masked to the number of options (jev_omni.JevOmni.predict / load_model.predict).

import type { Tokenizer } from "@huggingface/tokenizers";
import type { InferenceSession, Tensor } from "onnxruntime-common";
import { DecisionHead } from "./head.ts";
import { chatText, MAX_OPTIONS, MIN_OPTIONS, renderPrompt, type Question } from "./prompt.ts";

export interface OrtModule {
  InferenceSession: typeof InferenceSession;
  Tensor: typeof Tensor;
}

export interface VariantManifest { model: string; data: string[]; inputs: string[]; outputs: string[]; parity?: unknown }

export interface JevManifest {
  name: string;
  model: { repo: string; revision: string };
  base: { repo: string; revision: string };
  files: { tokenizer: string; tokenizer_config: string; head: string };
  head: { hidden: number; classes: number };
  sizes: Record<string, number>;
  sha256: Record<string, string>;
  variants: Record<string, VariantManifest>;
}

export interface Prediction {
  prediction: string;
  prediction_index: number;
  confidence: number;
  probabilities: Record<string, number>;
  /** probabilities in option order (options may repeat, the record above cannot hold duplicates) */
  probs: number[];
  input_tokens: number;
  latency_ms: number;
}

export class JevOmni {
  readonly manifest: JevManifest;
  readonly variant: string;
  readonly tokenizer: Tokenizer;
  private ort: OrtModule;
  private session: InferenceSession;
  private head: DecisionHead;
  private queue: Promise<unknown> = Promise.resolve();

  constructor(a: { ort: OrtModule; session: InferenceSession; head: DecisionHead; tokenizer: Tokenizer; manifest: JevManifest; variant: string }) {
    this.ort = a.ort; this.session = a.session; this.head = a.head; this.tokenizer = a.tokenizer;
    this.manifest = a.manifest; this.variant = a.variant;
  }

  /** Token ids for one question, as load_model.predict builds them. */
  encode(q: Question): number[] {
    return this.tokenizer.encode(chatText(renderPrompt(q)), { add_special_tokens: false }).ids;
  }

  /** Last-position hidden state for a token sequence. Runs are serialized: one session, one GPU. */
  hidden(ids: number[]): Promise<Float32Array> {
    const run = this.queue.then(() => this.run(ids));
    this.queue = run.catch(() => undefined);
    return run;
  }

  private async run(ids: number[]): Promise<Float32Array> {
    const n = ids.length;
    const T = this.ort.Tensor;
    const feeds: Record<string, Tensor> = {
      input_ids: new T("int64", BigInt64Array.from(ids, BigInt), [1, n]),
      attention_mask: new T("int64", new BigInt64Array(n).fill(1n), [1, n]),
      position_ids: new T("int64", BigInt64Array.from({ length: n }, (_, i) => BigInt(i)), [1, n]),
    };
    const out = await this.session.run(feeds, ["hidden_states"]);
    const h = out.hidden_states;
    const d = this.head.hidden;
    const last = new Float32Array((h.data as Float32Array).subarray((n - 1) * d, n * d));
    h.dispose();
    for (const t of Object.values(feeds)) t.dispose();
    return last;
  }

  /** Probabilities over the options for pre-encoded ids. */
  async probsForIds(ids: number[], nOptions: number): Promise<number[]> {
    return this.head.probs(await this.hidden(ids), nOptions);
  }

  probsFromHidden(hidden: Float32Array, nOptions: number): number[] {
    return this.head.probs(hidden, nOptions);
  }

  async predict(q: Question): Promise<Prediction> {
    if (q.options.length < MIN_OPTIONS || q.options.length > MAX_OPTIONS) throw new Error(`Jev-Omni needs ${MIN_OPTIONS}-${MAX_OPTIONS} options`);
    const ids = this.encode(q);
    const t0 = performance.now();
    const probs = await this.probsForIds(ids, q.options.length);
    const latency = performance.now() - t0;
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    return {
      prediction: q.options[best], prediction_index: best, confidence: probs[best],
      probabilities: Object.fromEntries(q.options.map((o, i) => [o, probs[i]])), probs,
      input_tokens: ids.length, latency_ms: Math.round(latency * 10) / 10,
    };
  }

  async release() { await this.session.release(); }
}
