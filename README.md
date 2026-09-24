# @ai-ecoverse/jev-omni.js

[Jev-Omni](https://huggingface.co/akhilaaa3/Jev-Omni) decision classifier running in the browser on WebGPU, in the
style of [kev.js](https://github.com/ai-ecoverse/kev.js). Supply a state, a question and 2–256 options, and get one
probability per option from a single forward pass.

Work in progress. Text works: the 13.3 GB int8 bundle matches the fp32 reference on all 293 DecisionBench medium
questions in Chrome on WebGPU ([docs/phase1-text.md](docs/phase1-text.md)). Images work: with a 200 MB fp32 vision
embedder, the bundle matches the reference on 232 of 234 kev-vision questions, at Kev-4B's accuracy
([docs/phase2-images.md](docs/phase2-images.md)). Video and audio are not done yet.
Background: [docs/phase0-feasibility.md](docs/phase0-feasibility.md).

**Demo:** <https://ai-ecoverse.github.io/jev-omni.js/>. It downloads 13.6 GB of weights on first use and needs
Chrome with WebGPU and plenty of memory; it has only been tested on an M4 Max with 128 GB. The weights are on
Hugging Face at [ai-ecoverse/jev-omni.js](https://huggingface.co/ai-ecoverse/jev-omni.js).

## Quick start

```js
import * as ort from "onnxruntime-web/webgpu"; // onnxruntime-web 1.30.0
import { loadJevOmni } from "@ai-ecoverse/jev-omni.js";

const jev = await loadJevOmni("https://huggingface.co/ai-ecoverse/jev-omni.js/resolve/main/jev-omni", { ort });
const res = await jev.predict({
  state: "The meeting starts at 10 AM. It is now 9 AM.",
  question: "Has the meeting started?",
  options: ["Yes", "No"],
  // image: { width, height, data }   // optional RGBA pixels
});
res.prediction; // "No"
res.probabilities; // { Yes: …, No: … }
```

- 2 to 256 options per question; quality is established up to 20.
- The loader checks each downloaded file's SHA-256 against the manifest and keeps the files in Cache Storage under
  the manifest's revision. Later loads read from disk; a new revision replaces the old files.
- Run it in a Web Worker, as the demo does (`site/worker.ts`).
- Prompt length: with onnxruntime-web 1.30, attention needs an n×n buffer per head, so prompts are capped at about
  8,000 tokens on a GPU with 4 GB buffers. `jev.maxTokens` holds the cap for the current GPU, and longer prompts are
  rejected with an error instead of failing inside the runtime.

To run the demo locally against a bundle in `public/models/jev-omni`: `npm run dev:demo`.

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
