// Gemma 4 unified's image processor (transformers' Gemma4UnifiedImageProcessor) for one RGB image: an
// aspect-ratio-preserving resize to at most 280 merged 48 x 48 patches, rescaling to [0, 1], and the merged patches
// with their (x, y) positions, padded to 280 rows. The vision embedder graph turns these into image features.

/** RGBA pixels, as canvas getImageData and createImageBitmap produce them. */
export interface ImageLike { width: number; height: number; data: Uint8ClampedArray | Uint8Array }

export interface VisionConfig {
  patch_size: number;           // 16: the processor's "teacher" patch
  pooling_kernel_size: number;  // 3: 3 x 3 teacher patches merge into one 48 x 48 model patch
  max_soft_tokens: number;      // 280
  rescale_factor: number;       // 1 / 255
}

export const GEMMA4_VISION: VisionConfig = { patch_size: 16, pooling_kernel_size: 3, max_soft_tokens: 280, rescale_factor: 0.00392156862745098 };

/** get_aspect_ratio_preserving_size: the largest size with at most max_soft_tokens * k^2 patches whose sides are
 * multiples of patch * k. Returns [height, width]. */
export function targetSize(height: number, width: number, c: VisionConfig = GEMMA4_VISION): [number, number] {
  const maxPatches = c.max_soft_tokens * c.pooling_kernel_size ** 2;
  const targetPx = maxPatches * c.patch_size ** 2;
  const factor = Math.sqrt(targetPx / (height * width));
  const side = c.pooling_kernel_size * c.patch_size;
  let h = Math.floor((factor * height) / side) * side, w = Math.floor((factor * width) / side) * side;
  if (h === 0 && w === 0) throw new RangeError(`cannot resize ${height} x ${width}: both sides round to 0`);
  const maxSide = Math.floor(maxPatches / c.pooling_kernel_size ** 2) * side;
  if (h === 0) { h = side; w = Math.min(Math.floor(width / height) * side, maxSide); }
  else if (w === 0) { w = side; h = Math.min(Math.floor(height / width) * side, maxSide); }
  return [h, w];
}

// torchvision resizes uint8 images with torch's antialiased bicubic, which uses Pillow's kernel (a = -0.5) and rounds
// to uint8 after each pass. Ported from cua-s1.js (src/four-b-vision.ts); torch rounds its weights to fixed point, so
// a few pixels can differ by a level or two.
const cubic = (x: number, a = -0.5) => {
  x = Math.abs(x);
  return x <= 1 ? ((a + 2) * x - (a + 3)) * x * x + 1 : x < 2 ? ((a * x - 5 * a) * x + 8 * a) * x - 4 * a : 0;
};

function axis(inSize: number, outSize: number) {
  const scale = inSize / outSize, support = scale > 1 ? 2 * scale : 2, inv = scale > 1 ? 1 / scale : 1;
  const out: { start: number; w: number[] }[] = [];
  for (let i = 0; i < outSize; i++) {
    const center = (i + 0.5) * scale;
    const start = Math.max(0, Math.trunc(center - support + 0.5)), end = Math.min(inSize, Math.trunc(center + support + 0.5));
    const w: number[] = [];
    let sum = 0;
    for (let j = start; j < end; j++) { const k = cubic((j - center + 0.5) * inv); w.push(k); sum += k; }
    out.push({ start, w: w.map((k) => (sum ? k / sum : 0)) });
  }
  return out;
}

/** Interleaved RGB (3 bytes per pixel) resized from h x w to H x W. */
function resizeRGB(src: Uint8Array, h: number, w: number, H: number, W: number): Uint8Array {
  const ax = axis(w, W), ay = axis(h, H);
  const tmp = new Uint8Array(h * W * 3);
  for (let y = 0; y < h; y++) for (let x = 0; x < W; x++) {
    const { start, w: k } = ax[x];
    for (let c = 0; c < 3; c++) {
      let s = 0;
      for (let t = 0; t < k.length; t++) s += src[(y * w + start + t) * 3 + c] * k[t];
      tmp[(y * W + x) * 3 + c] = Math.min(255, Math.max(0, Math.round(s)));
    }
  }
  const out = new Uint8Array(H * W * 3);
  for (let y = 0; y < H; y++) {
    const { start, w: k } = ay[y];
    for (let x = 0; x < W; x++) for (let c = 0; c < 3; c++) {
      let s = 0;
      for (let t = 0; t < k.length; t++) s += tmp[((start + t) * W + x) * 3 + c] * k[t];
      out[(y * W + x) * 3 + c] = Math.min(255, Math.max(0, Math.round(s)));
    }
  }
  return out;
}

export interface ImagePatches {
  /** [max_soft_tokens, (patch * k)^2 * 3], each row a 48 x 48 x 3 block (row-major, RGB), zero rows for padding */
  pixelValues: Float32Array;
  /** [max_soft_tokens, 2]: (x, y) of each merged patch, (-1, -1) for padding */
  positions: BigInt64Array;
  /** number of real patches = image tokens in the prompt */
  numSoftTokens: number;
  height: number;
  width: number;
}

/** The processor's pixel_values / image_position_ids for one image (alpha is dropped, as PIL's convert("RGB")). */
export function preprocessImage(img: ImageLike, c: VisionConfig = GEMMA4_VISION): ImagePatches {
  const { width: w, height: h } = img;
  let rgb: Uint8Array = new Uint8Array(w * h * 3);
  for (let i = 0; i < w * h; i++) { rgb[i * 3] = img.data[i * 4]; rgb[i * 3 + 1] = img.data[i * 4 + 1]; rgb[i * 3 + 2] = img.data[i * 4 + 2]; }
  const [H, W] = targetSize(h, w, c);
  if (H !== h || W !== w) rgb = resizeRGB(rgb, h, w, H, W);
  const side = c.patch_size * c.pooling_kernel_size, dim = side * side * 3;
  const gw = W / side, gh = H / side, n = gw * gh;
  const pixelValues = new Float32Array(c.max_soft_tokens * dim);
  const positions = new BigInt64Array(c.max_soft_tokens * 2).fill(-1n);
  const scale = Math.fround(c.rescale_factor);
  const lut = Float32Array.from({ length: 256 }, (_, v) => Math.fround(v * scale));
  for (let ky = 0; ky < gh; ky++) for (let kx = 0; kx < gw; kx++) {
    const m = ky * gw + kx;
    let o = m * dim;
    for (let y = 0; y < side; y++) {
      const row = ((ky * side + y) * W + kx * side) * 3;
      for (let x = 0; x < side * 3; x++) pixelValues[o++] = lut[rgb[row + x]];
    }
    positions[m * 2] = BigInt(kx);
    positions[m * 2 + 1] = BigInt(ky);
  }
  return { pixelValues, positions, numSoftTokens: n, height: H, width: W };
}
