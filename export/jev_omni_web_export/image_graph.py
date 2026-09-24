"""Give the text graph an image path without touching its weights: only model.onnx is rewritten.

Three new inputs, all empty ([0]) for a text prompt:

- `image_features` float32 [N, hidden]: the vision embedder's output for the image tokens, in prompt order.
- `image_positions` int64 [N]: the prompt position of each image token.
- `image_blocks` int64 [N]: which image each token belongs to (0, 1, ...); tokens of one image attend to each other in
  both directions.

With N > 0 (one top-level `If` computes the flag):

- the embeddings at `image_positions` are replaced by `image_features` (ScatterND), as Gemma4UnifiedModel's
  masked_scatter does;
- every sliding-window attention runs as MultiHeadAttention with an additive mask
  `(causal and within window) or same image`, transformers' mask for Gemma 4 unified; GroupQueryAttention, which
  onnxruntime only runs causally, is kept for text prompts. Each attention node becomes `If(has_image)`, so the
  text path runs the same kernels as before and both paths share one copy of the weights;
- with --global-image-mask block the full-attention layers get the same treatment (mask `causal or same image`),
  which is what transformers 5.17 computes; with `causal` they stay as they are (Gemma 4's documented design).

  uv run python -m jev_omni_web_export.image_graph --model <dir>/model.onnx [--out <file>] [--global-image-mask block]
"""
import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MS = "com.microsoft"
NEG = -3.0e38


def c64(name, values):
    return numpy_helper.from_array(np.array(values, np.int64), name)


def const_node(name, arr):
    return helper.make_node("Constant", [], [name], value=numpy_helper.from_array(arr, name + "_value"))


def mask_branch(window):
    """Then-branch of the mask If: additive masks [1,1,S,S] (float32) for sliding and full attention."""
    p = "/image_mask"
    n = [
        helper.make_node("Shape", ["input_ids"], [f"{p}/ids_shape"]),
        helper.make_node("Gather", [f"{p}/ids_shape", f"{p}/one"], [f"{p}/S"], axis=0),
        helper.make_node("Range", [f"{p}/zero", f"{p}/S", f"{p}/one"], [f"{p}/ar64"]),
        helper.make_node("Cast", [f"{p}/ar64"], [f"{p}/ar"], to=TensorProto.INT32),
        helper.make_node("Unsqueeze", [f"{p}/S", f"{p}/ax0"], [f"{p}/S1"]),
        helper.make_node("ConstantOfShape", [f"{p}/S1"], [f"{p}/neg1"],
                         value=numpy_helper.from_array(np.array([-1], np.int64), f"{p}/neg1_value")),
        helper.make_node("Unsqueeze", ["image_positions", f"{p}/ax1"], [f"{p}/pos_idx"]),
        helper.make_node("ScatterND", [f"{p}/neg1", f"{p}/pos_idx", "image_blocks"], [f"{p}/block64"]),
        helper.make_node("Cast", [f"{p}/block64"], [f"{p}/block"], to=TensorProto.INT32),
        helper.make_node("Unsqueeze", [f"{p}/ar", f"{p}/ax1"], [f"{p}/i"]),
        helper.make_node("Unsqueeze", [f"{p}/ar", f"{p}/ax0"], [f"{p}/j"]),
        helper.make_node("LessOrEqual", [f"{p}/j", f"{p}/i"], [f"{p}/causal"]),
        helper.make_node("Sub", [f"{p}/i", f"{p}/j"], [f"{p}/diff"]),
        helper.make_node("Less", [f"{p}/diff", f"{p}/window"], [f"{p}/inwin"]),
        helper.make_node("Unsqueeze", [f"{p}/block", f"{p}/ax1"], [f"{p}/bi"]),
        helper.make_node("Unsqueeze", [f"{p}/block", f"{p}/ax0"], [f"{p}/bj"]),
        helper.make_node("Equal", [f"{p}/bi", f"{p}/bj"], [f"{p}/eq"]),
        helper.make_node("GreaterOrEqual", [f"{p}/bi", f"{p}/zero32"], [f"{p}/isimg"]),
        helper.make_node("And", [f"{p}/eq", f"{p}/isimg"], [f"{p}/same"]),
        helper.make_node("And", [f"{p}/causal", f"{p}/inwin"], [f"{p}/cw"]),
        helper.make_node("Or", [f"{p}/cw", f"{p}/same"], [f"{p}/ok_sliding"]),
        helper.make_node("Or", [f"{p}/causal", f"{p}/same"], [f"{p}/ok_full"]),
        helper.make_node("Where", [f"{p}/ok_sliding", f"{p}/f0", f"{p}/fneg"], [f"{p}/b_sliding"]),
        helper.make_node("Where", [f"{p}/ok_full", f"{p}/f0", f"{p}/fneg"], [f"{p}/b_full"]),
        helper.make_node("Unsqueeze", [f"{p}/b_sliding", f"{p}/ax01"], ["image_bias_sliding_then"]),
        helper.make_node("Unsqueeze", [f"{p}/b_full", f"{p}/ax01"], ["image_bias_full_then"]),
    ]
    inits = [c64(f"{p}/zero", 0), c64(f"{p}/one", 1), c64(f"{p}/ax0", [0]), c64(f"{p}/ax1", [1]), c64(f"{p}/ax01", [0, 1]),
             numpy_helper.from_array(np.array(window, np.int32), f"{p}/window"),
             numpy_helper.from_array(np.array(0, np.int32), f"{p}/zero32"),
             numpy_helper.from_array(np.array(0.0, np.float32), f"{p}/f0"),
             numpy_helper.from_array(np.array(NEG, np.float32), f"{p}/fneg")]
    f = TensorProto.FLOAT
    return helper.make_graph(n, "image_mask_then", [], [
        helper.make_tensor_value_info("image_bias_sliding_then", f, None),
        helper.make_tensor_value_info("image_bias_full_then", f, None)], inits)


def empty_mask_branch():
    z = np.zeros((1, 1, 1, 1), np.float32)
    return helper.make_graph([const_node("image_bias_sliding_else", z), const_node("image_bias_full_else", z)],
                             "image_mask_else", [], [
                                 helper.make_tensor_value_info("image_bias_sliding_else", TensorProto.FLOAT, None),
                                 helper.make_tensor_value_info("image_bias_full_else", TensorProto.FLOAT, None)])


def mha_branch(gqa, bias, dtype):
    """GQA's q/k/v -> K/V repeated to all heads -> MultiHeadAttention with the additive mask."""
    a = {x.name: helper.get_attribute_value(x) for x in gqa.attribute}
    heads, kv = a["num_heads"], a["kv_num_heads"]
    p = f"{gqa.name}/image"
    q, k, v = gqa.input[:3]
    nodes = [
        helper.make_node("Reshape", [k, f"{p}/kv5"], [f"{p}/k5"]),
        helper.make_node("Reshape", [v, f"{p}/kv5"], [f"{p}/v5"]),
        helper.make_node("Expand", [f"{p}/k5", f"{p}/rep"], [f"{p}/ke"]),
        helper.make_node("Expand", [f"{p}/v5", f"{p}/rep"], [f"{p}/ve"]),
        helper.make_node("Reshape", [f"{p}/ke", f"{p}/flat"], [f"{p}/kf"]),
        helper.make_node("Reshape", [f"{p}/ve", f"{p}/flat"], [f"{p}/vf"]),
    ]
    b = bias
    if dtype != TensorProto.FLOAT:
        nodes.append(helper.make_node("Cast", [bias], [f"{p}/bias_cast"], to=dtype))
        b = f"{p}/bias_cast"
    out = f"{p}/out"
    nodes.append(helper.make_node("MultiHeadAttention", [q, f"{p}/kf", f"{p}/vf", "", "", b], [out], domain=MS,
                                  num_heads=heads, scale=a.get("scale", 0.0)))
    inits = [c64(f"{p}/kv5", [0, 0, kv, 1, -1]), c64(f"{p}/rep", [1, 1, 1, heads // kv, 1]), c64(f"{p}/flat", [0, 0, -1])]
    return helper.make_graph(nodes, f"{gqa.name}/then", [], [helper.make_tensor_value_info(out, dtype, None)], inits)


def producer_dtype(g, name):
    for n in g.node:
        if name in n.output and n.op_type == "Cast":
            return helper.get_attribute_value([x for x in n.attribute if x.name == "to"][0])
    return TensorProto.FLOAT


def rewrite(m, global_mask: str, window: int):
    g = m.graph
    if any(i.name == "image_features" for i in g.input):
        raise ValueError("graph already has an image path")
    ln0 = next(t for t in g.initializer if t.name == "model.layers.0.input_layernorm.weight")
    hidden = int(ln0.dims[0])
    g.input.extend([
        helper.make_tensor_value_info("image_features", TensorProto.FLOAT, ["num_image_tokens", hidden]),
        helper.make_tensor_value_info("image_positions", TensorProto.INT64, ["num_image_tokens"]),
        helper.make_tensor_value_info("image_blocks", TensorProto.INT64, ["num_image_tokens"]),
    ])
    g.initializer.extend([c64("/image/zero", 0), c64("/image/ax1", [1])])
    head = [
        helper.make_node("Size", ["image_positions"], ["/image/n"], name="/image/Size"),
        helper.make_node("Greater", ["/image/n", "/image/zero"], ["has_image"], name="/image/Greater"),
        helper.make_node("If", ["has_image"], ["image_bias_sliding", "image_bias_full"], name="/image/MaskIf",
                         then_branch=mask_branch(window), else_branch=empty_mask_branch()),
    ]
    # embeddings: the output of the scale Mul feeds layer 0
    (mul,) = [n for n in g.node if n.name == "/model/embed_tokens/Mul"]
    emb = mul.output[0]
    merged = "/image/embeds"
    then_emb = helper.make_graph([
        helper.make_node("Unsqueeze", ["image_positions", "/image/ax1"], ["/image/pos1"]),
        helper.make_node("Mul", ["/image/pos1", "/image/zero"], ["/image/b0"]),
        helper.make_node("Concat", ["/image/b0", "/image/pos1"], ["/image/idx"], axis=1),
        helper.make_node("ScatterND", [emb, "/image/idx", "image_features"], ["/image/embeds_then"]),
    ], "embeds_then", [], [helper.make_tensor_value_info("/image/embeds_then", TensorProto.FLOAT, None)])
    else_emb = helper.make_graph([helper.make_node("Identity", [emb], ["/image/embeds_else"])], "embeds_else", [],
                                 [helper.make_tensor_value_info("/image/embeds_else", TensorProto.FLOAT, None)])
    emb_if = helper.make_node("If", ["has_image"], [merged], name="/image/EmbedIf", then_branch=then_emb, else_branch=else_emb)
    for n in g.node:
        for k, x in enumerate(n.input):
            if x == emb:
                n.input[k] = merged
    nodes = list(g.node)
    at = nodes.index(mul) + 1
    for k, n in enumerate(head + [emb_if]):
        g.node.insert(at + k, n)

    counts = {"sliding": 0, "full": 0}
    for gqa in [n for n in g.node if n.op_type == "GroupQueryAttention"]:
        a = {x.name: helper.get_attribute_value(x) for x in gqa.attribute}
        sliding = a.get("local_window_size", -1) != -1
        if not sliding and global_mask == "causal":
            continue
        dtype = producer_dtype(g, gqa.input[0])
        out = gqa.output[0]
        i = list(g.node).index(gqa)
        g.node.remove(gqa)
        else_out = f"{gqa.name}/causal_out"
        orig = onnx.NodeProto()
        orig.CopyFrom(gqa)
        orig.output[0] = else_out
        else_g = helper.make_graph([orig], f"{gqa.name}/else", [], [helper.make_tensor_value_info(else_out, dtype, None)])
        then_g = mha_branch(gqa, "image_bias_sliding" if sliding else "image_bias_full", dtype)
        g.node.insert(i, helper.make_node("If", ["has_image"], [out], name=f"{gqa.name}/ImageIf",
                                          then_branch=then_g, else_branch=else_g))
        counts["sliding" if sliding else "full"] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model.onnx (external data is left where it is)")
    ap.add_argument("--out", help="output graph path; default: overwrite --model")
    ap.add_argument("--global-image-mask", choices=["block", "causal"], default="block")
    ap.add_argument("--window", type=int, default=1024)
    a = ap.parse_args()
    m = onnx.load(a.model, load_external_data=False)
    counts = rewrite(m, a.global_image_mask, a.window)
    with open(a.out or a.model, "wb") as f:
        f.write(m.SerializeToString())
    print(f"image path added: {counts['sliding']} sliding and {counts['full']} full attention layers switch to "
          f"MultiHeadAttention with an image mask -> {a.out or a.model}")


if __name__ == "__main__":
    main()
