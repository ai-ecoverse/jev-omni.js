# @ai-ecoverse/jev-omni.js

[Jev-Omni](https://huggingface.co/akhilaaa3/Jev-Omni) decision classifier running in the browser on WebGPU, in the
style of [kev.js](https://github.com/ai-ecoverse/kev.js). Supply a state, a question and 2–256 options, and get one
probability per option from a single forward pass.

Work in progress. Text works: the 13.3 GB int8 bundle matches the fp32 reference on all 293 DecisionBench medium
questions in Chrome on WebGPU ([docs/phase1-text.md](docs/phase1-text.md)). Images, video and audio are not done yet.
Background: [docs/phase0-feasibility.md](docs/phase0-feasibility.md).

## Credits

- [Jev-Omni](https://huggingface.co/akhilaaa3/Jev-Omni) by akhilaaa3, Apache-2.0: the fine-tuned Gemma 4 12B
  backbone, the 256-way decision head, the prompt format and the published results. Jev-Omni states that it is
  independent of TypeSafe AI's Jev.
- [Gemma 4](https://huggingface.co/google/gemma-4-12B-it) by Google: the base model, its encoder-free vision and
  audio embedders and its processor.
- The ONNX model builder is vendored from
  [microsoft/onnxruntime-genai PR #2473](https://github.com/microsoft/onnxruntime-genai/pull/2473) (MIT) at
  `export/vendor/ortgenai_models_pr2473`.

## License

Apache-2.0, following Jev-Omni and Gemma 4.
