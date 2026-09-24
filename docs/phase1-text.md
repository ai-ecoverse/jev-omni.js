# Phase 1: text on WebGPU

Status: **done**. The int8 bundle runs the full Jev-Omni text path in Chrome on WebGPU. It predicts the same answer
as the fp32 PyTorch reference on all 293 DecisionBench medium questions.

## Bundle

`export/build_model.sh fetch evalset build package` builds it from the pinned snapshot
(`akhilaaa3/Jev-Omni@c050d51`, `unified/model.safetensors` bf16, SHA-256 `d78782f3…`).

| | |
|---|---|
| Graph | onnxruntime-genai builder (PR #2473), `exclude_lm_head`, output = final-norm hidden states, single pass (no KV cache inputs/outputs) |
| Weights | int8 `MatMulNBits` (block 32) for all projections; embedding int8 per row in 32 column slices (max abs error 1.6e-2 on the ×62 scaled table) |
| Activations | fp32, except q/k/v of the 8 global-attention layers, which are cast to fp16 (see below) |
| Head | `head.safetensors` (256-way decision head from `head.pt`), run in JS |
| Size | 13.34 GB in 436 shards of ≤32 MB, plus a manifest with sizes and SHA-256 |
| Runtime | `onnxruntime-web@1.31.0-dev.20260918-bc8e7ed75` (WebGPU EP) |

Two things are needed for prompts longer than about 2.8k tokens on Apple GPUs:

- The sliding-window flash-attention path from onnxruntime PR #32462, which is only in the 1.31 dev builds so far.
  With 1.30 the sliding layers fall back to an n² kernel whose buffer size overflows.
- fp16 attention on the global layers (`postprocess.py --global-attn-fp16`). With head_dim 512 in fp32, one
  flash-attention tile needs 64 KB of workgroup memory, which is over the 32 KB limit, so onnxruntime falls back to
  the same overflowing n² kernel. In fp16 the tile fits. On the reduced test model this cost a mean |Δp| of 0.003 and
  no flips. The table below includes its effect, together with int8, on the real model.

`embed_scale` is overridden to 62.0 (the value in `runtime_buffers.pt`, and what Jev-Omni runs with) instead of the
builder's √3840 = 61.97.

## Numerical parity (6-layer fp32 graph)

To check the graph without quantization, `build_model.sh truncated` exports the first 6 decoder layers
(5 sliding, 1 global) in fp32 and compares them with the same 6 layers in PyTorch fp32 on CPU. The comparison uses
8 DecisionBench prompts of 1.9k–3.7k tokens, so the 1024-token window and the global layer are both exercised.

| | worst over 8 prompts |
|---|---|
| last-position relative error (max abs / max ref) | 4.9e-6 |
| last-position max abs error | 6.3e-3 (values up to 1.5e3) |
| any-position max abs error | 2.6e-2 |

This is well within the 1e-4 relative-error target. Results are in `build/eval/parity-fp32-6l.json`.

## Accuracy on DecisionBench medium

The eval set is 293 questions over 80 states (`akhilaaa3/DecisionBench` @ `19334fe`, medium split). Prompts range
from 1044 to 8937 tokens (median 2321) and up to 120 options. The mapping from typed questions to option strings is
this repo's own (`decisionbench.py`): `True: …`/`False: …` for yes/no, `key: description` for choice, and the
criteria list for scores. So absolute accuracy is not directly comparable to Jev-Omni's published 87.57%, although
it comes close.

| | fp32 reference (PyTorch, MPS) | int8 bundle (Chrome, WebGPU) |
|---|---|---|
| accuracy (micro) | 86.35% | 86.35% |
| accuracy (state macro) | 87.92% | 87.92% |
| Brier | 0.1912 | 0.1915 |
| ECE (10 bins) | 0.041 | 0.046 |
| mean confidence | 0.880 | 0.880 |
| predictions that differ from the reference | – | **0 / 293** |
| mean \|Δp\| (all options) | – | 0.0019 |
| max \|Δp\| | – | 0.131 (`b3-m-0003/cm_4471_disposition`, 7 options, top-1 p 0.82 → 0.69, still top-1) |
| questions with max \|Δp\| > 0.05 / > 0.02 | – | 6 / 28 |
| hidden-state relative L2 error, median / max | – | 0.025 / 0.085 |

The drift comes from int8 weights and the fp16 global attention combined. These runs do not separate the two.

## Latency (Apple M4 Max, 128 GB, Playwright Chromium, headless)

| prompt tokens | n | WebGPU int8 median | PyTorch fp32 MPS median |
|---|---|---|---|
| < 2000 | 114 | 6.7 s | 4.1 s |
| 2000–3000 | 102 | 11.7 s | 7.6 s |
| 3000–5000 | 63 | 18.1 s | 12.5 s |
| 5000–8937 | 14 | 45.8 s | 37.7 s |
| all | 293 | 10.9 s | 6.6 s |

The longest prompt (8937 tokens) took 57 s. Loading the bundle from a local server into WebGPU, with SHA-256
checks, took 14–16 s. The first download is 13.3 GB. After that, the bundle comes from Cache Storage
(`jev-omni-v1`).

## Reproduce

```sh
cd export
./build_model.sh fetch evalset build truncated package
uv run python -m jev_omni_web_export.reference --model ../build/Jev-Omni/unified --head ../build/Jev-Omni/head.pt \
  --buffers ../build/Jev-Omni/runtime_buffers.pt --evalset ../build/eval/decisionbench-medium.json \
  --out ../build/eval/reference-fp32.jsonl --device mps --mode fp32
cd ..
JEV_BUNDLE=public/models/jev-omni node --import tsx scripts/browser-eval.ts --variant q8f32 \
  --evalset build/eval/decisionbench-medium.json --out build/eval/browser-q8f32.jsonl
cd export && uv run python -m jev_omni_web_export.compare --evalset ../build/eval/decisionbench-medium.json \
  --ref ../build/eval/reference-fp32.jsonl ../build/eval/browser-q8f32.jsonl
```

The fp32 reference needs about 50 GB of GPU-visible memory. It frees the MPS cache after each question, because
the caching allocator otherwise ran out of memory after about 160 prompts of varying length.
