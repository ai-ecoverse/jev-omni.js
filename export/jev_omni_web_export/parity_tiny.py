"""Compare an exported hidden-state graph with the PyTorch text model on random prompts (no KV cache)."""
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForImageTextToText


def empty_past(sess, batch):
    feeds = {}
    for i in sess.get_inputs():
        if i.name.startswith("past_key_values."):
            shape = [batch if d == "batch_size" else 0 if isinstance(d, str) else d for d in i.shape]
            feeds[i.name] = np.zeros(shape, np.float32)
    return feeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[5, 16, 17, 40, 97])
    ap.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"])
    a = ap.parse_args()
    model = AutoModelForImageTextToText.from_pretrained(a.hf, dtype=torch.float32).eval()
    text = model.model.language_model
    sess = ort.InferenceSession(str(a.onnx / "model.onnx"), providers=a.providers)
    names = {i.name for i in sess.get_inputs()}
    g = torch.Generator().manual_seed(1)
    worst = 0.0
    for n in a.lengths:
        ids = torch.randint(3, 262000, (1, n), generator=g)
        with torch.inference_mode():
            ref = text(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state.numpy()
        feeds = {"input_ids": ids.numpy(), "attention_mask": np.ones((1, n), np.int64), **empty_past(sess, 1)}
        if "position_ids" in names:
            feeds["position_ids"] = np.arange(n, dtype=np.int64)[None]
        out = sess.run(["hidden_states"], feeds)[0]
        d = np.abs(out - ref)
        rel = d.max() / np.abs(ref).max()
        worst = max(worst, float(d.max()))
        print(f"len {n:4d}  max|Δ| {d.max():.3e}  mean|Δ| {d.mean():.3e}  max|ref| {np.abs(ref).max():.2f}  "
              f"rel {rel:.2e}  last-pos max|Δ| {d[0, -1].max():.3e}")
    print("worst", worst)


if __name__ == "__main__":
    main()
