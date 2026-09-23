# Phase 0: feasibility

Status: **export works on a reduced model; the full build is blocked on disk space** (35 GB free, about 80 GB
needed for a safe run).

## Pinned revisions

| Repo | Commit |
|---|---|
| `akhilaaa3/Jev-Omni` | `c050d51354147985d13286cf4acf90f562f2c631` (2026-09-22) |
| `google/gemma-4-12B-it` | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` |
| `microsoft/onnxruntime-genai` PR #2473 (`thpereir/onnxruntime-genai@gemma4-base-upstream`) | `603c36c57110e347a86ae87648f7dc1b1d8d8124`, vendored at `export/vendor/ortgenai_models_pr2473` |

## What the checkpoint is

- `backbone/` is the fine-tuned `Gemma4UnifiedTextModel` in fp32: 11.91B parameters, 47.6 GB.
- `unified/model.safetensors` is the whole `Gemma4UnifiedForConditionalGeneration` in bf16: 11.96B parameters,
  23.9 GB. It includes the vision embedder (52M) and the audio projection. Sampled Linear rows
  (`layers.0.q_proj`, `layers.20.down_proj`, `layers.47.o_proj`) are **bit-identical to the fp32 backbone rounded
  to bf16**, and differ from Google's stock weights, so this file is Jev-Omni, not the base model. `jev_omni.py` runs
  Linear weights in bf16 anyway, which makes `unified/` a faithful source that is half the size.
  Norms and `layer_scalar` are untouched by the fine-tune (they equal Google's).
- Gemma 4 12B "unified" is **encoder-free**. There is no SigLIP tower. An image becomes up to 280 merged 48 px
  patches (6912 values each) → LayerNorm → Dense(6912→3840) → LayerNorm → factorized 2-D position embedding →
  LayerNorm → RMSNorm → Linear. The decoder does the rest: on sliding layers, tokens within one image block attend
  to each other bidirectionally (global layers stay causal). Audio is also encoder-free: raw 640-sample frames go
  through one projection. So "Google's unmodified vision encoder" is a 52M-parameter embedder, and most of the vision
  compute happens in the fine-tuned decoder.

Decoder quirks the export must reproduce: attention scale 1.0 (q/k norms carry the scale), a scale-free `v_norm`,
K = V on the 8 global layers (`attention_k_eq_v`), head_dim 512 with one KV head on global layers vs 256 / 8 KV heads
on sliding ones, partial (0.25) "proportional" RoPE on global layers, `layer_scalar` on each layer output (layer 0:
0.053), a 1024-token sliding window, and embedding scale √3840.

## Exporter support

| Route | Gemma 4 unified |
|---|---|
| onnxruntime-genai 0.16.0 (latest release) | No. The builder stops at Gemma 3. |
| onnxruntime-genai PR #2473 (open, draft) | Yes, text only: a `Gemma4Model` that handles all the quirks above. |
| onnxruntime-genai PR #2286 (open) | Runtime/processor only (no builder); it confirms the encoder-free patch contract. |
| optimum-onnx 0.1.0 | No (Gemma 1–3 only). |
| Microsoft "mobius" (issue #2204) | Reportedly exports the 4-graph multimodal package with HF parity, but uses plain `Attention` with a float bias. Not checked. |

### Smoke test on a reduced model (`export/jev_omni_web_export/tiny.py`)

The test model is random-weight Gemma 4 unified with the real head dims, KV heads, K = V, RoPE, and vocab. It has
hidden size 256, 6 layers (s, s, global, s, s, global), and a 16-token window, so windowing is exercised.

Two adapters were needed to run PR #2473 against transformers 5.17 (`jev_omni_web_export/build.py`): lift the
per-layer attribute guard, and restore `global_head_dim` / `num_global_key_value_heads` from `per_layer_config`.
Post-processing (`postprocess.py`) handles two further issues:

- The CPU build sets `sliding_window_cache=1` on GQA, a ring-buffer mode that needs a preallocated past. The WebGPU
  kernel in onnxruntime-web 1.30 throws on it (`sliding_window_cache=1 is not implemented`). It is removed.
- The builder annotates the MLP intermediates with the hidden size. onnxruntime-web refuses to create the session
  (`Inferred=512 Declared=256`). The stale `value_info` entries are dropped.

Results (hidden state, last position unless stated):

| Check | max \|Δ\| (len 5 / 17 / 97) | relative |
|---|---|---|
| PyTorch fp32 vs PyTorch fp64 (noise floor of this model) | 2.2e-4 / 1.6e-3 / 3.6e-3 (all positions) | — |
| ONNX fp32 on ORT-CPU vs PyTorch fp64 | 1.3e-4 / 2.5e-3 / 3.7e-3 (all positions) | — |
| ONNX fp32 on **Chrome WebGPU** (Apple M-series, Metal) vs ORT-CPU | 1.0e-4 / 5.6e-4 / 7.3e-4 | ≤ 2.1e-4 |
| ONNX int8 (`MatMulNBits` 8-bit, block 32) on ORT-CPU or WebGPU vs PyTorch fp32 | 0.87–3.5 | 0.2–1.0 |

The fp32 graph is as close to the true value as PyTorch's own fp32 is, on CPU and on WebGPU. GQA with head_dim
512, the sliding window, K = V, `v_norm`, proportional RoPE and `layer_scalar` all run on WebGPU. The int8 row is
**not informative**: the random `layer_scalar` values (0.3–1.5 instead of ~0.05) make this model chaotic, so the
int8 error is amplified along with everything else. int8 quality can only be judged on the real weights.

The ~1e-4 fp32 parity target from the plan does not hold on this random model, because its own fp32 noise is
already 3.6e-3. On the real model, it needs a truncated fp32 graph (see the disk plan).

## Bundle size and browser limits

- Linear weights 10.90B → int8 block 32 with fp32 scales (as the builder emits them): **12.3 GB**. Embedding
  262144×3840 at int8 per row (kev's post-process): 1.0 GB. Vision + audio embedders in fp32: 0.2 GB.
  **Total ≈ 13.5 GB**, against 8.8 GB for Kev-9B. RoPE caches trimmed to 8,192 positions are negligible.
- Chrome's WebGPU adapter here reports `maxBufferSize` and `maxStorageBufferBindingSize` = 4 GiB − 4. The largest
  single tensor is the 1.0 GB embedding (which kev splits into column slices anyway), and the largest matrix is
  15360×3840 int8 = 59 MB. Weights need about 13.5 GB of GPU memory, which is fine on a 128 GB Mac.
- Attention activations: at 2k tokens the score matrix is 16×S×S×4 B ≈ 268 MB if the non-flash path is used. For
  video (16 frames × 280 tokens ≈ 4.5k tokens) it is ≈ 1.3 GB, which is still under the 4 GB limit. ORT's flash path
  applies when head_dim % 4 = 0, so this is only a fallback.
- Storage: Chrome's quota per origin is a share of total disk (hundreds of GB here), so the real limit is free
  space. kev.js ≥ 0.4's load-from-directory path avoids a second copy in Cache Storage.

## Images need a non-causal attention path (Phase 2 design constraint)

GroupQueryAttention is always causal. Its `attention_bias` input (supported on WebGPU in 1.30) can only mask
more, not unmask future image tokens. The sliding layers need `OR(causal, same-image-block)` within the window.
Options: (a) an image graph variant where the 40 sliding layers use `MultiHeadAttention` with an explicit additive
bias and K/V expanded from 8 to 16 heads, sharing weight files with the text graph; (b) one graph for everything,
with a bias built in JS per prompt. Text-only prompts would keep the GQA graph. This is design work, not a blocker.

## Disk budget (the blocker)

| Item | Size |
|---|---|
| `unified/model.safetensors` (source; `backbone/` fp32 at 47.6 GB does not fit at all) | 23.9 GB |
| int8 builder output, including untrimmed RoPE caches | ~14 GB |
| Sharded bundle (a copy, unless sharded in place) | ~13.5 GB |
| Truncated fp32 graph for the 1e-4 check (first 6 layers, including one global layer) | ~2 GB |
| Full fp32 ONNX reference as kev does it | ~48 GB (skip) |
| Free now | **35 GB** |

The lean path (bf16 source, in-place sharding, truncated fp32 parity, no full fp32 ONNX) peaks around 38–40 GB,
more than is free. Freeing to **≥ 80 GB** makes it comfortable, and ~130 GB would allow kev's full pipeline with an
fp32 ONNX reference. RAM (128 GB) is enough: loading the builder in fp32 takes ~48 GB, and the PyTorch reference in
bf16 on MPS takes ~24 GB.
