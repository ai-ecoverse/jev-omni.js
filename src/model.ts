// Jev-Omni in onnxruntime-web: one forward pass over the chat-templated prompt, the last position's hidden state,
// then the 256-way head masked to the number of options (jev_omni.JevOmni.predict / load_model.predict).

import type { Tokenizer } from "@huggingface/tokenizers";
import type { InferenceSession, Tensor } from "onnxruntime-common";
import { DecisionHead } from "./head.ts";
import { chatText, IMAGE_TOKEN_ID, imageChatText, MAX_OPTIONS, MIN_OPTIONS, renderPrompt, type Question } from "./prompt.ts";
import { preprocessImage, type ImageLike } from "./vision.ts";

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
  /** the vision embedder graph, shared by all variants (absent in text-only bundles) */
  vision?: { model: string; data: string[]; inputs: string[]; outputs: string[]; parity?: unknown };
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
  /** image soft tokens in the prompt (0 for text) */
  image_tokens: number;
  /** image preprocessing + vision embedder (0 for text) */
  embed_ms: number;
  /** decoder pass + head */
  decoder_ms: number;
  latency_ms: number;
}

/** Vision embedder output for one image: `numSoftTokens` rows of `hidden` floats. */
export interface ImageFeatures { data: Float32Array; numSoftTokens: number; ms: number }

export interface ImageQuestion extends Question { image?: ImageLike }

const round1 = (x: number) => Math.round(x * 10) / 10;

export class JevOmni {
  readonly manifest: JevManifest;
  readonly variant: string;
  readonly tokenizer: Tokenizer;
  private ort: OrtModule;
  private session: InferenceSession;
  private vision: InferenceSession | null;
  private head: DecisionHead;
  private queue: Promise<unknown> = Promise.resolve();

  constructor(a: { ort: OrtModule; session: InferenceSession; head: DecisionHead; tokenizer: Tokenizer; manifest: JevManifest; variant: string; vision?: InferenceSession | null }) {
    this.ort = a.ort; this.session = a.session; this.head = a.head; this.tokenizer = a.tokenizer;
    this.manifest = a.manifest; this.variant = a.variant; this.vision = a.vision ?? null;
  }

  /** Whether image questions can be answered (a vision embedder is loaded and the graph has the image inputs). */
  get supportsImages(): boolean { return !!this.vision && this.session.inputNames.includes("image_features"); }

  /** Token ids for one question, as load_model.predict builds them (text) or the Gemma 4 processor (image). */
  encode(q: Question, numSoftTokens = 0): number[] {
    const text = renderPrompt(q);
    return this.tokenizer.encode(numSoftTokens ? imageChatText(text, numSoftTokens) : chatText(text), { add_special_tokens: false }).ids;
  }

  private serialize<T>(f: () => Promise<T>): Promise<T> {
    const run = this.queue.then(f);
    this.queue = run.catch(() => undefined);
    return run;
  }

  /** The vision embedder's features for one image (preprocessing included in `ms`). */
  imageFeatures(img: ImageLike): Promise<ImageFeatures> {
    if (!this.vision) throw new Error("this bundle was loaded without the vision embedder");
    const vision = this.vision;
    return this.serialize(async () => {
      const t0 = performance.now();
      const p = preprocessImage(img);
      const T = this.ort.Tensor;
      const rows = p.positions.length / 2, dim = p.pixelValues.length / rows;
      const feeds = { pixel_values: new T("float32", p.pixelValues, [1, rows, dim]), image_position_ids: new T("int64", p.positions, [1, rows, 2]) };
      const out = await vision.run(feeds, ["image_features"]);
      const f = out.image_features;
      const d = this.head.hidden;
      const data = new Float32Array((f.data as Float32Array).subarray(0, p.numSoftTokens * d));
      f.dispose();
      for (const t of Object.values(feeds)) t.dispose();
      return { data, numSoftTokens: p.numSoftTokens, ms: performance.now() - t0 };
    });
  }

  /** Last-position hidden state for a token sequence, with image features at the image-token positions. Runs are
   * serialized: one GPU. */
  hidden(ids: number[], image?: ImageFeatures): Promise<Float32Array> {
    return this.serialize(() => this.run(ids, image));
  }

  private async run(ids: number[], image?: ImageFeatures): Promise<Float32Array> {
    const n = ids.length;
    const T = this.ort.Tensor;
    const d = this.head.hidden;
    const feeds: Record<string, Tensor> = {
      input_ids: new T("int64", BigInt64Array.from(ids, BigInt), [1, n]),
      attention_mask: new T("int64", new BigInt64Array(n).fill(1n), [1, n]),
      position_ids: new T("int64", BigInt64Array.from({ length: n }, (_, i) => BigInt(i)), [1, n]),
    };
    if (this.session.inputNames.includes("image_features")) {
      const pos: bigint[] = [];
      if (image) ids.forEach((id, i) => { if (id === IMAGE_TOKEN_ID) pos.push(BigInt(i)); });
      if (image && pos.length !== image.numSoftTokens) throw new Error(`${pos.length} image tokens in the prompt, ${image.numSoftTokens} image features`);
      feeds.image_features = new T("float32", image ? image.data : new Float32Array(0), [pos.length, d]);
      feeds.image_positions = new T("int64", BigInt64Array.from(pos), [pos.length]);
      feeds.image_blocks = new T("int64", new BigInt64Array(pos.length), [pos.length]);
    } else if (image) {
      throw new Error("this graph has no image inputs");
    }
    const out = await this.session.run(feeds, ["hidden_states"]);
    const h = out.hidden_states;
    const last = new Float32Array((h.data as Float32Array).subarray((n - 1) * d, n * d));
    h.dispose();
    for (const t of Object.values(feeds)) t.dispose();
    return last;
  }

  /** Probabilities over the options for pre-encoded ids. */
  async probsForIds(ids: number[], nOptions: number, image?: ImageFeatures): Promise<number[]> {
    return this.head.probs(await this.hidden(ids, image), nOptions);
  }

  probsFromHidden(hidden: Float32Array, nOptions: number): number[] {
    return this.head.probs(hidden, nOptions);
  }

  async predict(q: ImageQuestion): Promise<Prediction> {
    if (q.options.length < MIN_OPTIONS || q.options.length > MAX_OPTIONS) throw new Error(`Jev-Omni needs ${MIN_OPTIONS}-${MAX_OPTIONS} options`);
    const t0 = performance.now();
    const image = q.image ? await this.imageFeatures(q.image) : undefined;
    const ids = this.encode(q, image?.numSoftTokens ?? 0);
    const t1 = performance.now();
    const probs = await this.probsForIds(ids, q.options.length, image);
    const t2 = performance.now();
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    return {
      prediction: q.options[best], prediction_index: best, confidence: probs[best],
      probabilities: Object.fromEntries(q.options.map((o, i) => [o, probs[i]])), probs,
      input_tokens: ids.length, image_tokens: image?.numSoftTokens ?? 0,
      embed_ms: round1(image?.ms ?? 0), decoder_ms: round1(t2 - t1), latency_ms: round1(t2 - t0),
    };
  }

  async release() {
    await this.session.release();
    await this.vision?.release();
  }
}
