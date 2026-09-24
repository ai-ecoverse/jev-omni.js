# @ai-ecoverse/jev-omni.js

[Jev-Omni](https://huggingface.co/akhilaaa3/Jev-Omni) decision classifier running in the browser on WebGPU, in the
style of [kev.js](https://github.com/ai-ecoverse/kev.js). Supply a state, a question and 2–256 options, and get one
probability per option from a single forward pass.

Work in progress. Text works: the 13.3 GB int8 bundle matches the fp32 reference on all 293 DecisionBench medium
questions in Chrome on WebGPU ([docs/phase1-text.md](docs/phase1-text.md)). Images work: with a 200 MB fp32 vision
embedder, the bundle matches the reference on 232 of 234 kev-vision questions, at Kev-4B's accuracy
([docs/phase2-images.md](docs/phase2-images.md)). Video and audio are not done yet.
Background: [docs/phase0-feasibility.md](docs/phase0-feasibility.md).

## Credits

- [Jev-Omni](https://huggingface.co/akhilaaa3/Jev-Omni) by akhilaaa3, Apache-2.0: the fine-tuned Gemma 4 12B
  backbone, the 256-way decision head, the prompt format and the published results. Jev-Omni states that it is
  independent of TypeSafe AI's Jev.
- [Gemma 4](https://huggingface.co/google/gemma-4-12B-it) by Google: the base model, its encoder-free vision and
  audio embedders and its processor.
- The image resize in `src/vision.ts` is ported from [cua-s1.js](https://github.com/ai-ecoverse/cua-s1.js), and the
  image eval sets come from kev.js (`eval/vision-v1`, `eval/vision-v2`).
- The ONNX model builder is vendored from
  [microsoft/onnxruntime-genai PR #2473](https://github.com/microsoft/onnxruntime-genai/pull/2473) (MIT) at
  `export/vendor/ortgenai_models_pr2473`.

## Related

- [kev.js](https://github.com/ai-ecoverse/kev.js): Kev decision models in the browser; this repo follows its export
  pipeline and layout. Its `-vision` bundles answer questions about images through Qwen3.5's vision tower.
- [cua-s1.js](https://github.com/ai-ecoverse/cua-s1.js): Cua's form-filling and next-action models in the browser.
- [decision-vision-bench](https://github.com/ai-ecoverse/decision-vision-bench): Jev-Omni, Kev vision and
  cua-s1-4b-0.2 multimodal on one mixed image decision set.

## License

Apache-2.0, following Jev-Omni and Gemma 4.
