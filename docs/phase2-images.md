# Phase 2: images on WebGPU

Status: **done**. The bundle now answers image questions in Chrome on WebGPU. On kev-vision's two image sets it
predicts the same answer as the fp32 PyTorch reference on 232 of 234 questions. It is as accurate as Kev-4B on the
easy set (vision-v1) and slightly less accurate on the hard set (vision-v2). Both gaps are within noise. It is not
better than Kev-4B at counting.

## What was added

| | |
|---|---|
| Vision embedder | `vision_export.py`: Gemma 4's vision embedder and projection, exported in fp32 from the vision weights only. It is a separate graph of 200 MB in 4 shards (`r-c050d51/vision/`), and it outputs `image_features [N, 3840]`. |
| LayerNorm | onnxruntime's fused `LayerNormalization` loses precision on flat patches (catastrophic cancellation, 5e-3 relative error against fp64). `two_pass_layernorm` rewrites it as mean, then centered variance, which gives 1e-5 to 7e-5. |
| Decoder graph | `image_graph.py`, run by `postprocess.py` (`--image-mask block`, the default): new inputs `image_features`, `image_positions`, `image_blocks`. The features are scattered into the scaled embeddings. Each attention layer gets `If(has_image)`: with an image, multi-head attention with repeated KV heads and an additive bias; without an image, the original GQA node. No weights changed. The text path runs the same kernels as in Phase 1. |
| Preprocessing | `src/vision.ts`: Gemma 4's resize to at most 280 soft tokens (48 px merged patches), bicubic antialiased resize (ported from cua-s1.js), rescale to [0, 1], 16 px patches with their positions. |
| Prompt | `<bos><\|turn>user\n<\|image>` + N × `<\|image\|>` + `<image\|>` + the usual text. `test/encode.test.ts` checks the token ids against the Hugging Face processor on all 234 image requests. |
| Bundle | 13.58 GB in 445 files (13.34 GB int8 decoder, 0.20 GB fp32 vision embedder, head, tokenizer). |

### The attention mask for image tokens

In transformers 5.17, `Gemma4Unified` lets image tokens attend to each other in both directions within one image,
and it does so in all 48 layers: in the 40 sliding-window layers, the window is OR'd with the image block, and in
the 8 global layers, the causal mask is OR'd with the image block. Jev-Omni's own `jev_omni.py` passes the
processor's `mm_token_type_ids` to that model, so its behavior is the reference. Some Gemma 4 documentation
describes the global layers as causal, so both variants were run in PyTorch (`predict.py --image-mask block|sliding`):

| fp32 PyTorch | vision-v1 acc / Brier | vision-v2 acc / Brier | both sets |
|---|---|---|---|
| block on all layers (transformers, used) | 0.991 / 0.021 | 0.852 / 0.178 | 216 / 234 |
| block on sliding layers only, global causal | 0.981 / 0.019 | 0.875 / 0.169 | 216 / 234 |

The two variants tie. The bundle uses the transformers behavior. `postprocess.py --image-mask causal` builds the
other variant.

## Numerical parity

| check | result |
|---|---|
| 6-layer fp32 decoder with image, ORT CPU vs PyTorch fp32 (8 images over 8 families, 1 text) | last-position relative error 2.3e-6 to 5.0e-6 (text: 5.0e-6, as in Phase 1) |
| vision graph, ORT CPU vs PyTorch fp32, same pixels | relative error 1.1e-5 to 7.6e-4 |
| browser pixels (`src/vision.ts` + `createImageBitmap`) vs the processor, 85 images | 0.022% of values differ, by at most 2 of 255 levels |
| browser image features (preprocessing + vision graph on WebGPU) vs PyTorch, 17 images (one per family) | relative error 5.8e-5 to 2.0e-3 (the photo), cosine ≥ 0.999993 |
| text questions through the new graph vs the Phase 1 graph, WebGPU, 15 DecisionBench questions | identical probabilities (max \|Δp\| 0.0) |

Results are in `build/eval/parity-image-fp32-6l.json`, `pixel-parity.json` and `browser-q8f32-imagegraph.jsonl`.

## Accuracy on kev-vision

The sets are `kev-vision/eval/vision-v1` (41 images, 106 questions: photos, charts, receipts, screens, shapes, signs,
traffic lights) and `vision-v2` (44 images, 128 questions: screenshots of inboxes, shops, orders, dashboards,
charts, departure boards, labels, fine print, plus dot counting). `vision_sets.py` converts them into Jev requests:

- the item's `context` becomes the state, and the question's `instructions` becomes the question;
- `noul` questions get the options `["Yes", "No"]`, and the label is flipped to match;
- `choice` questions get the option keys, or `key: description` when there is a description;
- `score` questions get the levels verbatim;
- captions are not used.

Brier is summed over options, as in kev-vision's reports (Kev's "raw" Brier). The 95% intervals come from
resampling images. "Counting" is the questions answered by counting objects: v1 `count` and `done`, and v2 `unread`,
`over50`, `above60`, `refunded`, dot `total`, `red` and `more`.

| | Jev PyTorch fp32 | Jev WebGPU int8 | Kev-4B PyTorch | Kev-4B WebGPU | Kev-0.8B |
|---|---|---|---|---|---|
| v1 accuracy | 0.991 [0.971, 1.000] | 0.981 [0.953, 1.000] | 0.981 | 0.972 | 0.906 |
| v1 Brier | 0.021 | 0.022 | | | |
| v1 counting (n = 11) | 0.909 | 0.818 | 0.818 | | |
| v2 accuracy | 0.852 [0.772, 0.919] | 0.852 [0.772, 0.919] | 0.875 | 0.875 | 0.672 / 0.664 |
| v2 Brier | 0.178 | 0.182 | 0.173 | | |
| v2 counting (n = 40) | 0.600 | 0.600 | 0.675 | | |
| v2 everything else (n = 88) | 0.966 | 0.966 | 0.977 | | |

WebGPU against PyTorch: 1 flip on each set (v1 `shapes-5/count`: 7 → 6 at p 0.39; v2 `dots-5/total`: wrong
before and after). Mean |Δp| is 0.0014 on v1 and 0.0037 on v2, and max |Δp| is 0.065.

Question by question against Kev-4B (PyTorch rows in `kev-vision/eval/*/results/kev-4b/rows.json`):

| | both right | only Jev | only Kev | both wrong |
|---|---|---|---|---|
| v1 (Jev PyTorch) | 103 | 2 | 1 | 0 |
| v2 (Jev PyTorch) | 103 | 6 | 9 | 10 |

Jev's v2 errors are almost all counting, as Kev's are. It gets 12 of the 24 dot questions wrong, nearly always by
counting too few. It is also off by one on unread emails, products over $50 and bars above 60%. The non-counting
errors are two "largest order total" questions and one order status question. The two models make different
mistakes: 18 questions are answered correctly by only one of them.

## Latency (Apple M4 Max, 128 GB, Playwright Chromium, headless)

These runs shared the GPU with other work: 10 training processes and the decision-vision-bench browser runs, with
GPU utilization at 97%. The numbers below are measured under that load.

| | median | p90 |
|---|---|---|
| image embed (preprocessing + vision graph, 256-280 soft tokens) | 54 ms (v1), 59 ms (v2) | 78-81 ms |
| decoder, image question (about 330-360 tokens) | 4.5 s | 5.6 s |
| PyTorch fp32 MPS decoder, image question (quieter machine) | 1.5-1.6 s | |

To separate the load from the graph change, the same 4 DecisionBench questions (2.9k-3.0k tokens) ran alternately
through the Phase 1 text graph and the new graph, twice each. Medians were 26.3 s and 26.7 s, with identical
probabilities, so the `If` branches cost nothing measurable. The same 4 questions took 14.4-15.2 s in Phase 1 on an
idle GPU, so the load slowed everything down by about 1.75×. Scaled by that factor, an image question should take
about 2.5 s on an idle M4 Max. That fits the 1.8-2.8 s decoder times in the first smoke test. A matched
measurement is left to the decision-vision-bench runs, which time all three models on the same machine.

## Verdict against Kev

On these two sets, Jev-Omni does not beat Kev-4B enough to justify its cost:

- accuracy is the same within noise: +0.9 points on v1 and -2.3 points on v2 (PyTorch), with overlapping intervals;
- it has the same weakness, counting, and is slightly worse at it on v2 (0.600 vs 0.675);
- the bundle is 13.6 GB against Kev-4B's about 5.4 GB;
- an image question takes about 2.5 s against Kev-4B's 0.7-1.6 s. A long text question takes about 10 s.

What Jev-Omni adds is a different model family whose mistakes only partly overlap Kev's (18 of 234 questions are
answered correctly by only one model), its text accuracy on DecisionBench (Phase 1), and its audio and video paths.
For images alone, Kev-4B is the better choice for the browser.

## Reproduce

```sh
cd export
./build_model.sh visionsets vision build truncated imageparity package
# PyTorch reference, both masks (about 48 GB of unified memory each)
for m in block sliding; do for s in vision-v1 vision-v2; do
  uv run python -m jev_omni_web_export.predict --requests ../build/eval/$s.jsonl \
    --out ../build/eval/torch-fp32-$m-$s.jsonl --mode fp32 --image-mask $m
done; done
cd ..
for s in vision-v1 vision-v2; do
  node --import tsx scripts/predict.ts --requests build/eval/$s.jsonl --out build/eval/web-q8f32-$s.jsonl --no-verify
done
cd export && for s in v1 v2; do
  uv run python -m jev_omni_web_export.vision_report --ref ../build/eval/torch-fp32-block-vision-$s.jsonl \
    ../build/eval/web-q8f32-vision-$s.jsonl --kev ../../kev-vision/eval/vision-$s/results/kev-4b/rows.json
done
```

`predict.py` and `scripts/predict.ts` also answer a single question:
`--image img.png --state "…" --question "…" --options '["Yes","No"]'`.
