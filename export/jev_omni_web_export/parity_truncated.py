"""fp32 parity of the export on the real weights, truncated to the first N layers.

A full fp32 ONNX export of Jev-Omni is 48 GB. The builder's `num_hidden_layers` option exports the first N layers
plus the final norm, which is compared here with the PyTorch text model truncated the same way (runtime buffers
applied as in load_model.py), on real eval prompts. N = 6 covers five sliding layers and the first global layer.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoConfig, AutoModelForImageTextToText


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=Path, required=True, help="unified/ dir")
    ap.add_argument("--buffers", type=Path, required=True)
    ap.add_argument("--onnx", type=Path, required=True, help="dir with the truncated model.onnx")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--evalset", type=Path, required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    cfg = AutoConfig.from_pretrained(a.hf)
    t = cfg.text_config
    t.layer_types = t.layer_types[: a.layers]
    t.num_hidden_layers = a.layers
    model = AutoModelForImageTextToText.from_pretrained(a.hf, config=cfg, dtype=torch.float32)
    text = model.model.language_model.eval()
    for name, value in torch.load(a.buffers, map_location="cpu", weights_only=True).items():
        parent, _, attr = name.rpartition(".")
        if parent.startswith("layers.") and int(parent.split(".")[1]) >= a.layers:
            continue
        setattr(text.get_submodule(parent) if parent else text, attr, value)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(a.onnx / "model.onnx"), so, providers=["CPUExecutionProvider"])
    records = json.loads(a.evalset.read_text())["records"]
    # spread over lengths: every k-th record by token count
    records = sorted(records, key=lambda r: len(r["ids"]))[:: max(1, len(records) // a.n)][: a.n]
    rows = []
    for r in records:
        ids = torch.tensor([r["ids"]])
        with torch.inference_mode():
            ref = text(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state[0].numpy()
        n = ids.shape[1]
        out = sess.run(["hidden_states"], {"input_ids": ids.numpy(), "attention_mask": np.ones((1, n), np.int64),
                                           "position_ids": np.arange(n, dtype=np.int64)[None]})[0][0]
        d = np.abs(out - ref)
        row = {"id": r["id"], "tokens": n, "max_abs": float(d.max()), "max_abs_last": float(d[-1].max()),
               "rel_last": float(d[-1].max() / np.abs(ref[-1]).max()), "max_ref_last": float(np.abs(ref[-1]).max())}
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = {"layers": a.layers, "records": len(rows), "worst_max_abs_last": max(r["max_abs_last"] for r in rows),
               "worst_rel_last": max(r["rel_last"] for r in rows), "worst_max_abs": max(r["max_abs"] for r in rows)}
    print(json.dumps(summary))
    if a.out:
        a.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
