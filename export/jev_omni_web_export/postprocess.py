"""Turn a builder export into the browser bundle's graph, without touching the transformer body.

- `sliding_window_cache=1` makes GroupQueryAttention treat the sliding layers' past KV as a ring buffer that must be
  preallocated to the window size. Jev-Omni runs one pass over the whole prompt with an empty cache, and the WebGPU
  kernel in onnxruntime-web 1.30 rejects the attribute outright, so it is removed; `local_window_size` still limits
  attention to the window.
- The builder annotates the MLP's Gelu/Mul outputs with the hidden size instead of the intermediate size. Native
  onnxruntime warns and merges leniently, onnxruntime-web refuses to create the session, so those annotations are
  dropped.
- The builder scales embeddings by round(sqrt(hidden), 2) = 61.97. Jev-Omni's `runtime_buffers.pt` sets
  `embed_tokens.embed_scale` to 62.0 (sqrt(3840) in bf16), and `load_model.py` applies it, so --embed-scale
  overrides the constant.
- --embed int8 stores the 262k x 3840 embedding table as int8 with one scale per row, dequantized after the lookup
  with standard ops (Gather, Cast, Mul), in column slices so no file exceeds --shard-mb (ported from kev.js).
- The rotary caches are sized for 262k positions; --rope-positions trims them.
- --global-attn-fp16 runs the 8 full-attention layers' attention in fp16 (see fp16_global_attention); WebGPU
  cannot run them past ~2.8k tokens otherwise.
- --image-mask adds the image inputs and the per-layer image attention path (image_graph.py).
- Jev-Omni answers in one pass, so the graph is made single-pass: the past KV inputs are removed (optional GQA
  inputs) and `hidden_states` is the only output. The presents are then intermediates that onnxruntime frees
  layer by layer; as outputs they would hold every layer's KV at once (about 6 GB at 9k tokens).
"""
import argparse
import os
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from . import image_graph


def strip_sliding_window_cache(g) -> int:
    n = 0
    for node in g.node:
        if node.op_type == "GroupQueryAttention":
            keep = [a for a in node.attribute if a.name != "sliding_window_cache"]
            n += len(node.attribute) - len(keep)
            del node.attribute[:]
            node.attribute.extend(keep)
    return n


def drop_mlp_value_info(g) -> int:
    before = len(g.value_info)
    keep = [v for v in g.value_info if "/mlp/" not in v.name]
    del g.value_info[:]
    g.value_info.extend(keep)
    return before - len(keep)


def single_pass(g) -> int:
    past = {i.name for i in g.input if i.name.startswith("past_key_values.")}
    for node in g.node:
        if node.op_type == "GroupQueryAttention":
            for k in (3, 4):
                if node.input[k] in past:
                    node.input[k] = ""
    used = {x for n in g.node for x in n.input}
    if past & used:
        raise ValueError(f"past inputs still used outside GQA: {sorted(past & used)[:3]}")
    keep_in = [i for i in g.input if i.name not in past]
    del g.input[:]
    g.input.extend(keep_in)
    keep_out = [o for o in g.output if o.name == "hidden_states"]
    del g.output[:]
    g.output.extend(keep_out)
    return len(past)


def fp16_global_attention(g) -> int:
    """Cast q/k/v of the full-attention GQA nodes (head_dim 512) to fp16 and the output back to fp32.

    onnxruntime-web's flash-attention prefill keeps 2 x head_dim x 16 K/V elements in workgroup memory: 64 KB for
    head_dim 512 in fp32, twice the 32 KB WebGPU guarantees (and Apple GPUs have). It then falls back to a path whose
    buffers grow with seq^2, which overflows past ~2.8k tokens. In fp16 the tile is exactly 32 KB. q/k/v are
    RMS-normalized just before attention, so their range suits fp16.
    """
    n = 0
    for node in list(g.node):
        if node.op_type != "GroupQueryAttention":
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if attrs.get("local_window_size", -1) != -1:
            continue
        i = list(g.node).index(node)
        casts = []
        for k in range(3):
            src = node.input[k]
            dst = f"{node.name}/Cast_fp16_{k}/output_0"
            casts.append(helper.make_node("Cast", [src], [dst], name=f"{node.name}/Cast_fp16_{k}", to=TensorProto.FLOAT16))
            node.input[k] = dst
        out = node.output[0]
        node.output[0] = f"{node.name}/output_fp16"
        back = helper.make_node("Cast", [node.output[0]], [out], name=f"{node.name}/Cast_fp32", to=TensorProto.FLOAT)
        for k, c in enumerate(casts):
            g.node.insert(i + k, c)
        g.node.insert(i + len(casts) + 1, back)
        n += 1
    return n


def set_embed_scale(g, value: float):
    (mul,) = [n for n in g.node if n.name == "/model/embed_tokens/Mul"]
    name = "/model/constants/FLOAT/embed_scale"
    for t in g.initializer:
        if t.name == name:
            g.initializer.remove(t)
            break
    g.initializer.append(numpy_helper.from_array(np.array(value, np.float32), name))
    old, mul.input[1] = mul.input[1], name
    if old != name and not any(old in n.input for n in g.node):
        for n in list(g.node):
            if n.op_type == "Constant" and list(n.output) == [old]:
                g.node.remove(n)


def trim_rope(g, inits, positions: int):
    for name, t in inits.items():
        if name.startswith(("cos_cache", "sin_cache")):
            arr = numpy_helper.to_array(t)
            if arr.shape[0] > positions:
                t.CopyFrom(numpy_helper.from_array(np.ascontiguousarray(arr[:positions]), name))


def embed_int8(g, inits, shard_bytes: int):
    (node,) = [n for n in g.node if n.op_type == "Gather" and n.input[0] == "model.embed_tokens.weight"]
    w = numpy_helper.to_array(inits["model.embed_tokens.weight"])
    wf = w.astype(np.float32)
    scale = np.maximum(np.abs(wf).max(axis=1), 1e-12) / 127.0
    q = np.clip(np.rint(wf / scale[:, None]), -127, 127).astype(np.int8)
    err = float(np.abs(q.astype(np.float32) * scale[:, None] - wf).max())
    g.initializer.remove(inits["model.embed_tokens.weight"])
    del wf
    # An ONNX initializer cannot be split across external-data files, so the table is stored as column slices that
    # are gathered with the same ids and concatenated: bit-identical, and every file stays small.
    vocab, hidden = q.shape
    slices = max(1, -(-q.nbytes // shard_bytes))
    width = -(-hidden // slices)
    out, ids, p = node.output[0], node.input[1], "/model/embed_tokens"
    io = helper.np_dtype_to_tensor_dtype(w.dtype)
    new_nodes, cast_outputs = [], []
    for k, start in enumerate(range(0, hidden, width)):
        part = np.ascontiguousarray(q[:, start : start + width])
        g.initializer.append(numpy_helper.from_array(part, f"model.embed_tokens.weight_Q8_{k}"))
        new_nodes += [helper.make_node("Gather", [f"model.embed_tokens.weight_Q8_{k}", ids], [f"{p}/GatherQ8_{k}/output_0"], name=f"{p}/GatherQ8_{k}"),
                      helper.make_node("Cast", [f"{p}/GatherQ8_{k}/output_0"], [f"{p}/CastQ8_{k}/output_0"], name=f"{p}/CastQ8_{k}", to=io)]
        cast_outputs.append(f"{p}/CastQ8_{k}/output_0")
    g.initializer.append(numpy_helper.from_array(scale.astype(w.dtype), "model.embed_tokens.weight_scales"))
    joined = cast_outputs[0] if len(cast_outputs) == 1 else f"{p}/Concat/output_0"
    if len(cast_outputs) > 1:
        new_nodes.append(helper.make_node("Concat", cast_outputs, [joined], name=f"{p}/Concat", axis=-1))
    new_nodes += [helper.make_node("Gather", ["model.embed_tokens.weight_scales", ids], [f"{p}/GatherScale/output_0"], name=f"{p}/GatherScale"),
                  helper.make_node("Unsqueeze", [f"{p}/GatherScale/output_0", f"{p}/axes_last"], [f"{p}/Unsqueeze/output_0"], name=f"{p}/Unsqueeze"),
                  helper.make_node("Mul", [joined, f"{p}/Unsqueeze/output_0"], [out], name=f"{p}/MulScale")]
    g.initializer.append(numpy_helper.from_array(np.array([-1], np.int64), f"{p}/axes_last"))
    i = list(g.node).index(node)
    g.node.remove(node)
    for k, n in enumerate(new_nodes):
        g.node.insert(i + k, n)
    return len(cast_outputs), err


def save_sharded(m, out, max_bytes, threshold=1024):
    """Write initializers over `threshold` bytes to model.onnx.data, model.onnx.data_1, ... (each <= max_bytes unless a
    single tensor is larger), then the graph itself to model.onnx."""
    from onnx.external_data_helper import set_external_data
    files, fh, off = [], None, 0
    for t in m.graph.initializer:
        raw = t.raw_data if t.HasField("raw_data") else numpy_helper.from_array(numpy_helper.to_array(t), t.name).raw_data
        if len(raw) <= threshold:
            continue
        if fh is None or (off and off + len(raw) > max_bytes):
            if fh:
                fh.close()
            files.append("model.onnx.data" if not files else f"model.onnx.data_{len(files)}")
            fh, off = open(f"{out}/{files[-1]}", "wb"), 0
        fh.write(raw)
        nt = TensorProto(name=t.name, data_type=t.data_type, dims=list(t.dims), raw_data=raw)
        set_external_data(nt, location=files[-1], offset=off, length=len(raw))
        nt.data_location = TensorProto.EXTERNAL
        nt.ClearField("raw_data")
        t.CopyFrom(nt)
        off += len(raw)
    if fh:
        fh.close()
    with open(f"{out}/model.onnx", "wb") as f:
        f.write(m.SerializeToString())
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="builder output dir")
    ap.add_argument("--out", help="bundle dir; omit to only rewrite src/model.onnx (graph fixes, no weight changes)")
    ap.add_argument("--embed", choices=["keep", "int8"], default="int8")
    ap.add_argument("--embed-scale", type=float, default=62.0)
    ap.add_argument("--rope-positions", type=int, default=16384)
    ap.add_argument("--shard-mb", type=int, default=32)
    ap.add_argument("--global-attn-fp16", action="store_true", help="needed for WebGPU; see fp16_global_attention")
    ap.add_argument("--image-mask", choices=["none", "block", "causal"], default="block",
                    help="add the image path (image_graph.py); block/causal: the full-attention layers' image mask")
    ap.add_argument("--window", type=int, default=1024, help="sliding window, for the image mask")
    ap.add_argument("--delete-src-data", action="store_true",
                    help="remove the builder's model.onnx.data once it is loaded, so src and out never coexist on disk")
    a = ap.parse_args()
    in_place = a.out is None
    m = onnx.load(f"{a.src}/model.onnx", load_external_data=not in_place)
    g = m.graph
    print("sliding_window_cache removed from", strip_sliding_window_cache(g), "nodes")
    print("MLP value_info dropped:", drop_mlp_value_info(g))
    set_embed_scale(g, a.embed_scale)
    print("past inputs removed:", single_pass(g))
    if a.global_attn_fp16:
        print("global attention layers in fp16:", fp16_global_attention(g))
    if a.image_mask != "none":
        counts = image_graph.rewrite(m, a.image_mask, a.window)
        print(f"image path: {counts['sliding']} sliding + {counts['full']} full attention layers with an image mask")
    if in_place:
        onnx.save(m, f"{a.src}/model.onnx")
        return
    if a.delete_src_data:
        for f in os.listdir(a.src):
            if f.startswith("model.onnx.data"):
                os.remove(f"{a.src}/{f}")
    inits = {t.name: t for t in g.initializer}
    trim_rope(g, inits, a.rope_positions)
    if a.embed == "int8":
        k, err = embed_int8(g, inits, a.shard_mb * 1_000_000)
        print(f"embedding -> int8 per-row in {k} column slices (max abs error {err:.2e})")
    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(a.src):
        if f.endswith((".json", ".jinja")):
            shutil.copy(f"{a.src}/{f}", a.out)
    for f in os.listdir(a.out):
        if f.startswith("model.onnx.data"):
            os.remove(f"{a.out}/{f}")
    files = save_sharded(m, a.out, a.shard_mb * 1_000_000)
    total = sum(os.path.getsize(f"{a.out}/{f}") for f in files)
    print(f"{a.out}: {len(files)} files, {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
