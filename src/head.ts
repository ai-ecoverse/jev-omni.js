// Jev-Omni's Head256: z = W((h - mu) / sd) + b over the first n of 256 outputs, softmax. Weights ship as fp32
// safetensors (head.safetensors, converted from head.pt by jev_omni_web_export.package).

export interface F32Tensor { shape: number[]; data: Float32Array }

export function parseSafetensors(buf: ArrayBuffer): Record<string, F32Tensor> {
  const n = Number(new DataView(buf).getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n))) as Record<string, { dtype: string; shape: number[]; data_offsets: [number, number] }>;
  const out: Record<string, F32Tensor> = {};
  for (const [name, t] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    if (t.dtype !== "F32") throw new Error(`${name}: expected F32, got ${t.dtype}`);
    const [a, b] = t.data_offsets;
    out[name] = { shape: t.shape, data: new Float32Array(buf.slice(8 + n + a, 8 + n + b)) };   // copy: offsets need not be 4-aligned
  }
  return out;
}

export class DecisionHead {
  readonly hidden: number;
  readonly classes: number;
  private mu: Float32Array; private sd: Float32Array; private w: Float32Array; private b: Float32Array;

  constructor(t: Record<string, F32Tensor>) {
    for (const k of ["mu", "sd", "linear.weight", "linear.bias"]) if (!t[k]) throw new Error(`head weights must contain ${k}`);
    [this.classes, this.hidden] = t["linear.weight"].shape;
    this.mu = t.mu.data; this.sd = t.sd.data; this.w = t["linear.weight"].data; this.b = t["linear.bias"].data;
  }

  static fromSafetensors(buf: ArrayBuffer) { return new DecisionHead(parseSafetensors(buf)); }

  logits(h: Float32Array, n: number): number[] {
    if (h.length !== this.hidden) throw new Error(`hidden state has ${h.length} values, head expects ${this.hidden}`);
    if (n < 1 || n > this.classes) throw new Error(`head supports 1-${this.classes} options, got ${n}`);
    const x = new Float64Array(this.hidden);
    for (let j = 0; j < this.hidden; j++) x[j] = (h[j] - this.mu[j]) / this.sd[j];
    const z: number[] = [];
    for (let i = 0; i < n; i++) {
      let s = this.b[i];
      const row = i * this.hidden;
      for (let j = 0; j < this.hidden; j++) s += this.w[row + j] * x[j];
      z.push(s);
    }
    return z;
  }

  probs(h: Float32Array, n: number): number[] {
    const z = this.logits(h, n);
    const m = Math.max(...z);
    const e = z.map((v) => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }
}
