"""Checks predict.Jev's image path (own embedding merge and masks) against transformers' own forward, on the tiny
random model: with --image-mask block the last hidden state must match Gemma4UnifiedModel.forward.

  uv run python -m jev_omni_web_export.check_predict --src ../build/Jev-Omni --image some.png
"""
import argparse
import shutil
import tempfile
from pathlib import Path

import torch
from PIL import Image

from .decisionbench import prompt
from .predict import Jev
from .reference import Head256
from .tiny import make


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--image", required=True)
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as d:
        snap = Path(d)
        model = make(a.src, snap / "unified")
        for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            shutil.copy(a.src / f, snap / f)
        torch.manual_seed(1)
        torch.save(Head256(256).state_dict(), snap / "head.pt")
        torch.save({}, snap / "runtime_buffers.pt")
        req = {"state": "A picture.", "question": "What is it?", "options": ["a", "b", "c"], "image": a.image}
        results = {}
        for mask in ("block", "sliding"):
            jev = Jev.load(snap, "cpu", "fp32", mask)
            results[mask] = torch.tensor(jev(req)["hidden"])
        inputs = jev.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": Image.open(a.image).convert("RGB")},
                                          {"type": "text", "text": prompt(req["state"], req["question"], req["options"])}]}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False)
        inputs.pop("num_soft_tokens_per_image", None)
        with torch.inference_mode():
            ref = model.model(**inputs, use_cache=False).last_hidden_state[0, -1]
        for mask, h in results.items():
            print(f"{mask:8s} vs transformers forward: max|Δ| {(h - ref).abs().max():.3e} (ref max {ref.abs().max():.2f})")


if __name__ == "__main__":
    main()
