"""Browser image features (scripts/vision-features.ts) against PyTorch's vision embedder on the processor's pixels.

  uv run python -m jev_omni_web_export.vision_parity --hf <unified dir> --features <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from .vision_export import load_embedder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    a = ap.parse_args()
    emb, _ = load_embedder(a.hf)
    proc = AutoProcessor.from_pretrained(a.hf)
    rows = []
    for item in json.loads((a.features / "index.json").read_text()):
        x = proc.image_processor(Image.open(item["image"]).convert("RGB"), return_tensors="pt")
        with torch.inference_mode():
            ref = emb(x["pixel_values"].float(), x["image_position_ids"]).pooler_output[0]
        ref = ref[(x["image_position_ids"][0] != -1).all(-1)].numpy()
        got = np.fromfile(a.features / item["file"], dtype=np.float32).reshape(ref.shape)
        d = np.abs(got - ref)
        cos = (got * ref).sum(-1) / np.linalg.norm(got, axis=-1) / np.linalg.norm(ref, axis=-1)
        row = {"image": item["image"], "max_abs": float(d.max()), "rel": float(d.max() / np.abs(ref).max()),
               "min_cos": float(cos.min())}
        rows.append(row)
        print(json.dumps(row))
    print(json.dumps({"images": len(rows), "worst_rel": max(r["rel"] for r in rows), "worst_min_cos": min(r["min_cos"] for r in rows)}))


if __name__ == "__main__":
    main()
