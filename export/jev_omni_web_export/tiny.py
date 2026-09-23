"""A small random-weight Gemma 4 unified checkpoint with Jev-Omni's attention shapes.

Real head dims (256 sliding / 512 global, 1 global KV head, K=V on global layers), real vocab, partial
proportional RoPE and per-layer `layer_scalar`, but a narrow hidden size, 6 layers and a 16-token sliding window
so windowing is exercised by short prompts. Used to check an exporter without downloading the 12B weights.
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForImageTextToText


def make(src: Path, out: Path, seed: int = 0):
    cfg = json.loads((src / "unified" / "config.json").read_text())
    t = cfg["text_config"]
    t.update(hidden_size=256, intermediate_size=512, num_hidden_layers=6, sliding_window=16,
             layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2)
    t.pop("per_layer_config", None)
    cfg["vision_config"].update(mm_embed_dim=256, output_proj_dims=256)
    cfg["audio_config"].update(hidden_size=64, audio_embed_dim=64, output_proj_dims=64)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg))
    config = AutoConfig.from_pretrained(out)
    torch.manual_seed(seed)
    model = AutoModelForImageTextToText.from_config(config, dtype=torch.float32).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight") or "layernorm" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(0, 0.05)
        for layer in model.model.language_model.layers:
            layer.layer_scalar.fill_(float(torch.empty(1).uniform_(0.3, 1.5)))
    model.save_pretrained(out)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copy(src / f, out / f)
    for f in ("generation_config.json", "processor_config.json"):
        shutil.copy(src / "unified" / f, out / f)
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="directory with Jev-Omni's small files")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    make(a.src, a.out)
    print("wrote", a.out)
