"""ORT-CPU parity of an image-path graph (image_graph.py) with predict.Jev on the same image features.

Runs requests (image and text) through PyTorch (optionally truncated to the graph's layer count) and through the
ONNX graph fed the PyTorch vision features, and compares all hidden states. Also runs the vision embedder graph
(vision_export.py) when given and compares its output with PyTorch's.

  uv run python -m jev_omni_web_export.parity_image --hf <unified dir> --graph <model.onnx> --requests <jsonl> \
      [--buffers runtime_buffers.pt] [--layers 6] [--vision <vision.onnx>] [--image-mask block] [--limit 6]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoProcessor, AutoTokenizer

from .predict import Jev, load_parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=Path, required=True, help="unified/ dir (config, weights, processor)")
    ap.add_argument("--tokenizer", type=Path, help="dir with tokenizer.json + chat template (default: --hf)")
    ap.add_argument("--buffers", type=Path)
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--vision", type=Path)
    ap.add_argument("--layers", type=int)
    ap.add_argument("--requests", type=Path, required=True)
    ap.add_argument("--image-mask", choices=["block", "sliding"], default="block")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    text, vision, config = load_parts(a.hf, a.buffers, "cpu", "fp32", a.layers)
    jev = Jev(text, vision, config, AutoTokenizer.from_pretrained(a.tokenizer or a.hf), AutoProcessor.from_pretrained(a.hf),
              None, "cpu", "fp32", a.image_mask)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(a.graph), so, providers=["CPUExecutionProvider"])
    vsess = ort.InferenceSession(str(a.vision), so, providers=["CPUExecutionProvider"]) if a.vision else None
    reqs = [json.loads(l) for l in a.requests.open() if l.strip()][: a.limit]
    rows = []
    hidden_size = text.config.hidden_size
    for r in reqs:
        ids, is_image, inputs = jev.encode(r)
        feats = jev.image_features(inputs) if is_image is not None else None
        row = {"id": r["id"], "tokens": len(ids), "image": is_image is not None}
        if feats is not None and vsess is not None:
            vo = vsess.run(None, {"pixel_values": inputs["pixel_values"].numpy().astype(np.float32),
                                  "image_position_ids": inputs["image_position_ids"].numpy().astype(np.int64)})[0][0]
            valid = (inputs["image_position_ids"][0] != -1).all(-1).numpy()
            dv = np.abs(vo[valid] - feats.numpy())
            row.update(vision_max_abs=float(dv.max()), vision_rel=float(dv.max() / np.abs(feats.numpy()).max()))
        ref = jev.hidden_states(ids, is_image, feats).numpy()
        n = len(ids)
        if is_image is None:
            f = np.zeros((0, hidden_size), np.float32)
            pos = np.zeros((0,), np.int64)
        else:
            f = feats.numpy().astype(np.float32)
            pos = np.nonzero(is_image.numpy())[0].astype(np.int64)
        out = sess.run(["hidden_states"], {
            "input_ids": ids[None].numpy().astype(np.int64), "attention_mask": np.ones((1, n), np.int64),
            "position_ids": np.arange(n, dtype=np.int64)[None], "image_features": f, "image_positions": pos,
            "image_blocks": np.zeros_like(pos)})[0][0]
        d = np.abs(out - ref)
        row.update(max_abs=float(d.max()), max_abs_last=float(d[-1].max()), rel_last=float(d[-1].max() / np.abs(ref[-1]).max()))
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = {"graph": str(a.graph), "layers": a.layers, "image_mask": a.image_mask, "records": len(rows),
               "worst_rel_last": max(r["rel_last"] for r in rows), "worst_max_abs": max(r["max_abs"] for r in rows)}
    if any("vision_rel" in r for r in rows):
        summary["worst_vision_rel"] = max(r.get("vision_rel", 0) for r in rows)
    print(json.dumps(summary))
    if a.out:
        a.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
