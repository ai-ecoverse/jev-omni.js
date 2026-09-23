"""Graph fixes applied to the builder output before it is sharded.

`sliding_window_cache=1` makes GroupQueryAttention treat the sliding layers' past KV as a ring buffer that must
be preallocated to the window size. Jev-Omni runs one pass over the whole prompt with an empty cache, and the
WebGPU kernel in onnxruntime-web 1.30 rejects the attribute outright, so it is removed; `local_window_size` still
limits attention to the window.

The builder annotates the MLP's Gelu/Mul outputs with the hidden size instead of the intermediate size. Native
onnxruntime warns and merges leniently, onnxruntime-web refuses to create the session, so annotations whose
declared shape disagrees with inference are dropped.
"""
import argparse
from pathlib import Path

import onnx


def strip_sliding_window_cache(model: onnx.ModelProto) -> int:
    n = 0
    for node in model.graph.node:
        if node.op_type == "GroupQueryAttention":
            keep = [a for a in node.attribute if a.name != "sliding_window_cache"]
            n += len(node.attribute) - len(keep)
            del node.attribute[:]
            node.attribute.extend(keep)
    return n


def drop_mlp_value_info(model: onnx.ModelProto) -> int:
    before = len(model.graph.value_info)
    keep = [v for v in model.graph.value_info if "/mlp/" not in v.name]
    del model.graph.value_info[:]
    model.graph.value_info.extend(keep)
    return before - len(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    a = ap.parse_args()
    path = a.model_dir / "model.onnx"
    model = onnx.load(str(path), load_external_data=False)
    print("sliding_window_cache removed from", strip_sliding_window_cache(model), "nodes")
    print("MLP value_info dropped:", drop_mlp_value_info(model))
    onnx.save(model, str(path))


if __name__ == "__main__":
    main()
