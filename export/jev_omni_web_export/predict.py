"""PyTorch Jev-Omni over a JSONL file of requests, text or image, with timings: the reference for the browser runs.

Request lines: {"id", "state", "question", "options", "image"?} (other fields are passed through). Output lines:
{"id", "probs", "tokens", "image_tokens", "prep_ms", "embed_ms", "decoder_ms", "ms", ...passed-through fields}.
The output is appended to and resumed from, so a crashed run can be restarted with the same command.

  uv run python -m jev_omni_web_export.predict --requests ../build/eval/vision-v1.jsonl \
      --out ../build/eval/torch-fp32-vision-v1.jsonl [--mode fp32|bf16] [--image-mask block|sliding]

A single question without a requests file:

  uv run python -m jev_omni_web_export.predict --image cat.png --state "A photo." --question "Is it a cat?" \
      --options '["Yes", "No"]'

Semantics follow Jev-Omni's own code: text prompts go through the tokenizer's chat template as load_model.predict
does; image prompts go through the Gemma 4 processor's chat template with the image before the text, as
jev_omni.py does (one image, up to 280 soft tokens). The model is `unified/model.safetensors` with
`runtime_buffers.pt` applied (embed_scale 62.0), in fp32 by default, which is what the ONNX graphs compute.

Image attention mask (`--image-mask`): tokens of one image attend to each other in both directions. transformers
5.17's Gemma4Unified applies that on every layer (`block`, the default, what jev_omni.py gets with this
transformers); Gemma 4's documented design (`create_masks_for_vision_model`) keeps global layers causal and applies
it on sliding layers only (`sliding`).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from .decisionbench import encode, prompt
from .reference import Head256

HERE = Path(__file__).resolve().parents[2]
SNAP = HERE / "build" / "Jev-Omni"


def load_parts(unified: Path, buffers: Path | None, device: str, mode: str, layers: int | None = None):
    """The text model (runtime buffers applied) and the vision embedder; `layers` truncates the decoder."""
    config = AutoConfig.from_pretrained(unified)
    if layers:
        config.text_config.layer_types = config.text_config.layer_types[:layers]
        config.text_config.num_hidden_layers = layers
    model = AutoModelForImageTextToText.from_pretrained(unified, config=config, dtype=torch.float32 if mode == "fp32" else torch.bfloat16)
    text = model.model.language_model.eval()
    vision = model.model.embed_vision.float().eval()
    del model
    if mode == "bf16":
        for name, p in text.named_parameters():
            if not any(name.endswith(f"{proj}.weight") for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")):
                p.data = p.data.float()
    text.to(device)
    vision.to(device)
    if buffers is not None:
        for name, value in torch.load(buffers, map_location="cpu", weights_only=True).items():
            parent, _, attr = name.rpartition(".")
            if layers and parent.startswith("layers.") and int(parent.split(".")[1]) >= layers:
                continue
            setattr(text.get_submodule(parent) if parent else text, attr, value.to(device))
    return text, vision, config


class Jev:
    def __init__(self, text, vision, config, tokenizer, processor, head, device: str, mode: str, image_mask: str):
        self.text, self.vision, self.config, self.head = text, vision, config, head
        self.tokenizer, self.processor = tokenizer, processor
        self.device, self.mode, self.image_mask = device, mode, image_mask
        self.window = self.text.config.sliding_window

    @classmethod
    def load(cls, snapshot: Path, device: str, mode: str, image_mask: str):
        text, vision, config = load_parts(snapshot / "unified", snapshot / "runtime_buffers.pt", device, mode)
        head = Head256(text.config.hidden_size)
        head.load_state_dict(torch.load(snapshot / "head.pt", map_location="cpu", weights_only=True))
        return cls(text, vision, config, AutoTokenizer.from_pretrained(snapshot),
                   AutoProcessor.from_pretrained(snapshot / "unified"), head.to(device).eval(), device, mode, image_mask)

    def sync(self):
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device.startswith("cuda"):
            torch.cuda.synchronize()

    def masks(self, n, block_ids):
        i = torch.arange(n, device=self.device)[:, None]
        j = torch.arange(n, device=self.device)[None, :]
        causal = j <= i
        b = block_ids.to(self.device)
        same = (b[:, None] == b[None, :]) & (b[:, None] >= 0)
        # transformers ORs the image block in after the window; with a 1024 window and <= 280-token images the
        # order only matters for the tiny test model.
        sliding = (causal & (i - j < self.window)) | same
        full = (causal | same) if self.image_mask == "block" else causal
        return {"full_attention": full[None, None], "sliding_attention": sliding[None, None]}

    def encode(self, req):
        """Token ids, the image-token mask (None for text) and the processor's image tensors."""
        text = prompt(req["state"], req["question"], req["options"])
        if not req.get("image"):
            return torch.tensor(encode(self.tokenizer, text)), None, None
        image = Image.open(req["image"]).convert("RGB")
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False)
        return inputs["input_ids"][0], inputs["mm_token_type_ids"][0] == 1, inputs

    @torch.inference_mode()
    def image_features(self, inputs):
        """[N, hidden] vision embeddings of the image's real (non-padding) patches, in prompt order."""
        feats = self.vision(inputs["pixel_values"].to(self.device, torch.float32),
                            inputs["image_position_ids"].to(self.device)).pooler_output
        return feats[0][(inputs["image_position_ids"][0] != -1).all(-1).to(self.device)]

    @torch.inference_mode()
    def hidden_states(self, ids, is_image, feats):
        """[S, hidden] final-norm hidden states for token ids with image features merged in."""
        n = len(ids)
        x = ids.clone()
        if is_image is not None:
            x[is_image] = self.config.get_text_config().pad_token_id
        embeds = self.text.embed_tokens(x[None].to(self.device))
        block = torch.full((n,), -1)
        if is_image is not None:
            assert feats.shape[0] == int(is_image.sum()), (feats.shape[0], int(is_image.sum()))
            embeds = embeds.clone()
            embeds[0, is_image.to(self.device)] = feats.to(embeds.dtype)
            block = torch.where(is_image, 0, -1)
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.mode == "bf16"):
            out = self.text(inputs_embeds=embeds, attention_mask=self.masks(n, block),
                            position_ids=torch.arange(n, device=self.device)[None], use_cache=False)
        return out.last_hidden_state[0].float()

    @torch.inference_mode()
    def __call__(self, req):
        t0 = time.perf_counter()
        ids, is_image, inputs = self.encode(req)
        n = len(ids)
        t1 = time.perf_counter()
        self.sync()
        t2 = time.perf_counter()
        feats = self.image_features(inputs) if is_image is not None else None
        n_img = 0 if feats is None else feats.shape[0]
        self.sync()
        t3 = time.perf_counter()
        hidden = self.hidden_states(ids, is_image, feats)[-1:]
        probs = self.head(hidden, torch.tensor([len(req["options"])], device=self.device)).softmax(-1)[0, : len(req["options"])]
        probs = probs.cpu().tolist()
        t4 = time.perf_counter()
        if self.device == "mps":
            # The MPS caching allocator keeps a heap per distinct sequence length and OOMs after ~150 prompts.
            torch.mps.empty_cache()
        ms = lambda a, b: round((b - a) * 1000, 1)
        return {"probs": probs, "tokens": n, "image_tokens": n_img, "prep_ms": ms(t0, t1), "embed_ms": ms(t2, t3),
                "decoder_ms": ms(t3, t4), "ms": ms(t0, t4), "hidden": hidden[0].cpu().tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--requests", type=Path, help="JSONL requests; omit to answer one question from the flags below")
    ap.add_argument("--out", type=Path, help="JSONL output (appended, resumed); stdout if omitted")
    ap.add_argument("--image", help="single question: image path (omit for text)")
    ap.add_argument("--state")
    ap.add_argument("--question")
    ap.add_argument("--options", help="single question: JSON list of 2-256 strings")
    ap.add_argument("--snapshot", type=Path, default=SNAP)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--mode", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--image-mask", choices=["block", "sliding"], default="block")
    ap.add_argument("--hidden", action="store_true", help="also write the last-position hidden state")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    torch.set_float32_matmul_precision("highest")
    if a.requests:
        reqs = [json.loads(l) for l in a.requests.open() if l.strip()][: a.limit]
    else:
        if not (a.state is not None and a.question and a.options):
            ap.error("--state, --question and --options are required without --requests")
        reqs = [{"id": "q", "state": a.state, "question": a.question, "options": json.loads(a.options), "image": a.image}]
    done = set()
    if a.out and a.out.exists():
        done = {json.loads(l)["id"] for l in a.out.open() if l.strip()}
    todo = [r for r in reqs if r["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go", file=sys.stderr)
    if not todo:
        return
    t = time.perf_counter()
    jev = Jev.load(a.snapshot, a.device, a.mode, a.image_mask)
    print(f"loaded in {time.perf_counter() - t:.0f} s", file=sys.stderr)
    sink = a.out.open("a") if a.out else sys.stdout
    for k, r in enumerate(todo):
        res = jev(r)
        if not a.hidden:
            res.pop("hidden")
        row = {**r, **res, "mode": a.mode, "image_mask": a.image_mask if r.get("image") else None}
        sink.write(json.dumps(row) + "\n")
        sink.flush()
        top = max(range(len(res["probs"])), key=res["probs"].__getitem__)
        print(f"[{len(done) + k + 1}/{len(reqs)}] {r['id']} {res['tokens']} tok embed {res['embed_ms']:.0f} ms "
              f"decoder {res['decoder_ms']:.0f} ms -> {r['options'][top]!r} p={res['probs'][top]:.3f}"
              + (f" label {r['label']}" if "label" in r else ""), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
