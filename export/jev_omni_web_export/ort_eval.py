"""Run an ONNX hidden-state graph plus the decision head over an eval set; writes the same JSON lines as reference.py."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


def empty_past(sess):
    feeds = {}
    for i in sess.get_inputs():
        if i.name.startswith("past_key_values."):
            shape = [1 if d == "batch_size" else 0 if isinstance(d, str) else d for d in i.shape]
            feeds[i.name] = np.zeros(shape, np.float32)
    return feeds


class OrtJev:
    def __init__(self, model_dir: Path, head: Path, providers=("CPUExecutionProvider",)):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(model_dir / "model.onnx"), so, providers=list(providers))
        self.past = empty_past(self.sess)
        h = torch.load(head, map_location="cpu", weights_only=True)
        self.mu, self.sd = h["mu"].numpy()[0], h["sd"].numpy()[0]
        self.w, self.b = h["linear.weight"].numpy(), h["linear.bias"].numpy()

    def hidden(self, ids):
        n = len(ids)
        out = self.sess.run(["hidden_states"], {"input_ids": np.array([ids], np.int64), "attention_mask": np.ones((1, n), np.int64),
                                                "position_ids": np.arange(n, dtype=np.int64)[None], **self.past})[0]
        return out[0, -1].astype(np.float32)

    def probs(self, hidden, n_options):
        z = self.w[:n_options] @ ((hidden - self.mu) / self.sd) + self.b[:n_options]
        e = np.exp(z - z.max())
        return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--evalset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    records = json.loads(a.evalset.read_text())["records"][: a.limit]
    done = {json.loads(l)["id"] for l in a.out.open()} if a.out.exists() else set()
    todo = [r for r in records if r["id"] not in done]
    jev = OrtJev(a.model, a.head)
    with a.out.open("a") as f:
        for i, r in enumerate(todo):
            t0 = time.time()
            h = jev.hidden(r["ids"])
            p = jev.probs(h, len(r["options"]))
            ms = (time.time() - t0) * 1000
            f.write(json.dumps({"id": r["id"], "probs": p.tolist(), "hidden": h.tolist(), "ms": round(ms), "tokens": len(r["ids"])}) + "\n")
            f.flush()
            print(f"[{len(done) + i + 1}/{len(records)}] {r['id']} {len(r['ids'])} tok {ms:.0f} ms", flush=True)


if __name__ == "__main__":
    main()
