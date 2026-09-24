"""Export Gemma 4 unified's vision embedder (52M parameters, encoder-free) to ONNX in fp32.

Inputs are the image processor's outputs, padded to 280 merged patches: `pixel_values` float32 [1, 280, 6912]
(48 x 48 x 3 patches, values in [0, 1]) and `image_position_ids` int64 [1, 280, 2] ((x, y) per patch, (-1, -1) for
padding). Output `image_features` float32 [1, 280, 3840]; the caller keeps the rows of real patches.

Only the embedder's tensors are read from the checkpoint (bf16, upcast exactly to fp32), so this needs no 24 GB load.
Weights go to external data files of at most --shard-mb, like the decoder.

  uv run python -m jev_omni_web_export.vision_export --hf ../build/Jev-Omni/unified --out ../build/onnx/vision
"""
import argparse
import os
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedVisionEmbedder

from .postprocess import save_sharded

# checkpoint name -> module name (transformers renames these on load)
RENAMES = {"model.vision_embedder.": "", "model.embed_vision.embedding_projection.": "multimodal_embedder.embedding_projection."}


def load_embedder(hf: Path):
    config = AutoConfig.from_pretrained(hf)
    module = Gemma4UnifiedVisionEmbedder(config.vision_config, config.text_config).float().eval()
    state = {}
    files = sorted(hf.glob("*.safetensors"))
    for path in files:
        with safe_open(path, "pt") as f:
            for key in f.keys():
                for prefix, repl in RENAMES.items():
                    if key.startswith(prefix):
                        state[repl + key[len(prefix):]] = f.get_tensor(key).float()
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"missing {missing}, unexpected {unexpected}")
    return module, config


def two_pass_layernorm(g) -> int:
    """Replace LayerNormalization with mean, centered variance, scale: onnxruntime's kernels take the variance as
    E[x^2] - E[x]^2, which cancels catastrophically on flat 48 x 48 patches (a uniform screenshot background) and put
    5e-3 relative error into the features where PyTorch has 5e-6. Mul/Reciprocal rather than Pow/Div keep the
    optimizer's LayerNormFusion from folding the pattern back."""
    n = 0
    for node in list(g.node):
        if node.op_type != "LayerNormalization":
            continue
        x, w, b = node.input
        eps = next(helper.get_attribute_value(a) for a in node.attribute if a.name == "epsilon")
        p = node.name
        g.initializer.append(numpy_helper.from_array(np.array(eps, np.float32), f"{p}/eps"))
        # opset 17: ReduceMean takes axes as an attribute
        new = [
            helper.make_node("ReduceMean", [x], [f"{p}/mean"], axes=[-1], keepdims=1),
            helper.make_node("Sub", [x, f"{p}/mean"], [f"{p}/d"]),
            helper.make_node("Mul", [f"{p}/d", f"{p}/d"], [f"{p}/d2"]),
            helper.make_node("ReduceMean", [f"{p}/d2"], [f"{p}/var"], axes=[-1], keepdims=1),
            helper.make_node("Add", [f"{p}/var", f"{p}/eps"], [f"{p}/ve"]),
            helper.make_node("Sqrt", [f"{p}/ve"], [f"{p}/std"]),
            helper.make_node("Reciprocal", [f"{p}/std"], [f"{p}/inv"]),
            helper.make_node("Mul", [f"{p}/d", f"{p}/inv"], [f"{p}/norm"]),
            helper.make_node("Mul", [f"{p}/norm", w], [f"{p}/scaled"]),
            helper.make_node("Add", [f"{p}/scaled", b], [node.output[0]]),
        ]
        i = list(g.node).index(node)
        g.node.remove(node)
        for k, nn in enumerate(new):
            g.node.insert(i + k, nn)
        n += 1
    return n


class Wrapper(torch.nn.Module):
    def __init__(self, embedder):
        super().__init__()
        self.embedder = embedder

    def forward(self, pixel_values, image_position_ids):
        return self.embedder(pixel_values, image_position_ids).pooler_output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard-mb", type=int, default=32)
    a = ap.parse_args()
    module, config = load_embedder(a.hf)
    n = config.vision_config.num_soft_tokens
    d = config.vision_config.model_patch_size ** 2 * 3
    pv = torch.rand(1, n, d)
    pos = torch.stack(torch.meshgrid(torch.arange(20), torch.arange(14), indexing="xy"), -1).reshape(-1, 2)[:n]
    pos = torch.cat([pos, torch.full((n - len(pos), 2), -1)])[None]
    a.out.mkdir(parents=True, exist_ok=True)
    tmp = a.out / "tmp.onnx"
    torch.onnx.export(Wrapper(module), (pv, pos), str(tmp), input_names=["pixel_values", "image_position_ids"],
                      output_names=["image_features"], opset_version=17, dynamo=False)
    m = onnx.load(str(tmp))
    m.ir_version = 10
    print("LayerNormalization -> two-pass:", two_pass_layernorm(m.graph))
    for f in a.out.iterdir():
        if f.name.startswith("vision.onnx"):
            f.unlink()
    files = save_sharded(m, a.out, a.shard_mb * 1_000_000)
    os.replace(a.out / "model.onnx", a.out / "vision.onnx")
    for k, f in enumerate(files):
        os.replace(a.out / f, a.out / f.replace("model.onnx", "vision.onnx"))
    tmp.unlink()
    # external data locations were written as model.onnx.data*; point them at the renamed files
    m = onnx.load(str(a.out / "vision.onnx"), load_external_data=False)
    for t in m.graph.initializer:
        for e in t.external_data:
            if e.key == "location":
                e.value = e.value.replace("model.onnx", "vision.onnx")
    onnx.save(m, str(a.out / "vision.onnx"))
    total = sum(os.path.getsize(a.out / f.replace("model.onnx", "vision.onnx")) for f in files)
    print(f"{a.out / 'vision.onnx'}: {len(files)} data files, {total / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
