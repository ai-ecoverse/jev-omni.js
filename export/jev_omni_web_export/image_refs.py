"""Reference outputs of the Gemma 4 processor for the JS port's parity tests.

For every request with an image: the token ids of the image prompt (processor chat template, as jev_omni.py), and
per image the resized size, the merged-patch positions and the pixel values as uint8 levels (pixel * 255).

  uv run python -m jev_omni_web_export.image_refs --requests ../build/eval/vision-v1.jsonl ../build/eval/vision-v2.jsonl \
      --out ../build/eval/image-refs
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoProcessor

from .decisionbench import prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=Path, nargs="+", required=True)
    ap.add_argument("--processor", type=Path, default=Path(__file__).resolve().parents[2] / "build/Jev-Omni/unified")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    proc = AutoProcessor.from_pretrained(a.processor)
    a.out.mkdir(parents=True, exist_ok=True)
    index = {"requests": [], "images": {}}
    for f in a.requests:
        for line in f.open():
            r = json.loads(line)
            if not r.get("image"):
                continue
            image = Image.open(r["image"]).convert("RGB")
            x = proc.apply_chat_template(
                [{"role": "user", "content": [{"type": "image", "image": image},
                                              {"type": "text", "text": prompt(r["state"], r["question"], r["options"])}]}],
                add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="np", enable_thinking=False)
            index["requests"].append({"id": r["id"], "image": r["image"], "state": r["state"], "question": r["question"],
                                      "options": r["options"], "ids": x["input_ids"][0].tolist()})
            if r["image"] in index["images"]:
                continue
            pos = x["image_position_ids"][0]
            valid = (pos != -1).all(-1)
            n = int(valid.sum())
            levels = np.rint(x["pixel_values"][0][:n] * 255).astype(np.uint8)
            name = f"{len(index['images']):03d}.bin"
            levels.tofile(a.out / name)
            index["images"][r["image"]] = {"levels": name, "num_soft_tokens": n, "positions": pos[:n].tolist(),
                                           "size": list(image.size)}
    (a.out / "index.json").write_text(json.dumps(index))
    print(f"{a.out}: {len(index['requests'])} requests, {len(index['images'])} images")


if __name__ == "__main__":
    main()
