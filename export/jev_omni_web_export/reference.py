"""PyTorch reference probabilities for an eval set, with load_model.py's semantics.

The text model comes from `unified/model.safetensors` (Jev-Omni's backbone rounded to bf16, which is what
jev_omni.py runs its Linear layers in). `runtime_buffers.pt` is applied as load_model.py does, which sets
`embed_scale` to 62.0 and the RoPE inverse frequencies. Modes:

- fp32: every weight upcast to fp32, fp32 math. This is what the ONNX graphs compute (before int8), so it is the
  parity reference.
- bf16: Linear weights in bf16 under bf16 autocast, as jev_omni.py runs on CUDA; for accuracy only.

Writes one JSON line per record (probabilities and the last-position hidden state) and resumes from what it finds.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText


class Head256(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.register_buffer("mu", torch.zeros(1, hidden))
        self.register_buffer("sd", torch.ones(1, hidden))
        self.linear = torch.nn.Linear(hidden, 256, dtype=torch.float32)

    def forward(self, features, counts):
        z = self.linear((features.float() - self.mu) / self.sd)
        return z.masked_fill(torch.arange(256, device=z.device)[None] >= counts[:, None], -1e30)


def load(model_dir: Path, head: Path, buffers: Path | None, device: str, mode: str):
    model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.float32 if mode == "fp32" else torch.bfloat16)
    text = model.model.language_model.eval()
    del model
    if mode == "bf16":
        for name, p in text.named_parameters():
            if not any(name.endswith(f"{proj}.weight") for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")):
                p.data = p.data.float()
    text = text.to(device)
    if buffers is not None:
        for name, value in torch.load(buffers, map_location="cpu", weights_only=True).items():
            parent, _, attr = name.rpartition(".")
            module = text.get_submodule(parent) if parent else text
            setattr(module, attr, value.to(device))
    h = Head256(text.config.hidden_size)
    h.load_state_dict(torch.load(head, map_location="cpu", weights_only=True))
    return text, h.to(device).eval()


@torch.inference_mode()
def run(text, head, ids, n_options, device, mode):
    x = torch.tensor([ids], device=device)
    with torch.autocast(device, dtype=torch.bfloat16, enabled=mode == "bf16"):
        hidden = text(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).last_hidden_state[:, -1].float()
    probs = head(hidden, torch.tensor([n_options], device=device)).softmax(-1)[0, :n_options]
    return probs.cpu().tolist(), hidden[0].cpu().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help="dir with config.json + model.safetensors (unified/)")
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--buffers", type=Path)
    ap.add_argument("--evalset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="JSON lines, appended")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--mode", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    torch.set_float32_matmul_precision("highest")
    records = json.loads(a.evalset.read_text())["records"][: a.limit]
    done = {json.loads(l)["id"] for l in a.out.open()} if a.out.exists() else set()
    todo = [r for r in records if r["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go")
    if not todo:
        return
    text, head = load(a.model, a.head, a.buffers, a.device, a.mode)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("a") as f:
        for i, r in enumerate(todo):
            t0 = time.time()
            probs, hidden = run(text, head, r["ids"], len(r["options"]), a.device, a.mode)
            ms = (time.time() - t0) * 1000
            f.write(json.dumps({"id": r["id"], "probs": probs, "hidden": hidden, "ms": round(ms), "tokens": len(r["ids"])}) + "\n")
            f.flush()
            print(f"[{len(done) + i + 1}/{len(records)}] {r['id']} {len(r['ids'])} tok {ms:.0f} ms  p[label]={probs[r['label']]:.3f}", flush=True)


if __name__ == "__main__":
    main()
