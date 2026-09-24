"""Single-op attention graphs for checking how onnxruntime-web's WebGPU EP handles image-block masks.

Writes <out>/<name>.onnx plus, for the small check length, raw float32 inputs and a numpy reference output.

  uv run python -m jev_omni_web_export.attn_probe --out /tmp/jev-attn
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

MS = "com.microsoft"


def mha_graph(name, heads, kv_heads, head_dim, fp16):
    """q/k/v in GQA layout -> K/V repeated to all heads -> MultiHeadAttention with an additive [1,1,S,S] bias."""
    f = TensorProto.FLOAT
    rep = heads // kv_heads
    nodes = [
        helper.make_node("Shape", ["q"], ["qshape"]),
        helper.make_node("Slice", ["qshape", "c0", "c2"], ["bs"]),
        helper.make_node("Concat", ["bs", "kvshape_tail"], ["kv5"], axis=0),
        helper.make_node("Reshape", ["k", "kv5"], ["k5"]),
        helper.make_node("Reshape", ["v", "kv5"], ["v5"]),
        helper.make_node("Concat", ["bs", "rep_tail"], ["rep5"], axis=0),
        helper.make_node("Expand", ["k5", "rep5"], ["ke"]),
        helper.make_node("Expand", ["v5", "rep5"], ["ve"]),
        helper.make_node("Concat", ["bs", "flat_tail"], ["flat3"], axis=0),
        helper.make_node("Reshape", ["ke", "flat3"], ["kf"]),
        helper.make_node("Reshape", ["ve", "flat3"], ["vf"]),
    ]
    q, k, v, bias = "q", "kf", "vf", "bias"
    if fp16:
        nodes += [helper.make_node("Cast", [x], [x + "16"], to=TensorProto.FLOAT16) for x in ("q", "kf", "vf", "bias")]
        q, k, v, bias = "q16", "kf16", "vf16", "bias16"
    nodes.append(helper.make_node("MultiHeadAttention", [q, k, v, "", "", bias], ["o16" if fp16 else "o"],
                                  domain=MS, num_heads=heads, scale=1.0))
    if fp16:
        nodes.append(helper.make_node("Cast", ["o16"], ["o"], to=f))
    inits = [
        helper.make_tensor("c0", TensorProto.INT64, [1], [0]),
        helper.make_tensor("c2", TensorProto.INT64, [1], [2]),
        helper.make_tensor("kvshape_tail", TensorProto.INT64, [3], [kv_heads, 1, head_dim]),
        helper.make_tensor("rep_tail", TensorProto.INT64, [3], [kv_heads, rep, head_dim]),
        helper.make_tensor("flat_tail", TensorProto.INT64, [1], [heads * head_dim]),
    ]
    g = helper.make_graph(nodes, name, [
        helper.make_tensor_value_info("q", f, [1, "S", heads * head_dim]),
        helper.make_tensor_value_info("k", f, [1, "S", kv_heads * head_dim]),
        helper.make_tensor_value_info("v", f, [1, "S", kv_heads * head_dim]),
        helper.make_tensor_value_info("bias", f, [1, 1, "S", "S"]),
    ], [helper.make_tensor_value_info("o", f, [1, "S", heads * head_dim])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid(MS, 1)])
    m.ir_version = 10
    return m


def if_graph(name, heads, kv_heads, head_dim, window):
    """If(use_bias) { MHA with bias } else { GQA }, both reading q/k/v from the outer scope."""
    f = TensorProto.FLOAT
    mha = mha_graph("m", heads, kv_heads, head_dim, False).graph
    for n in mha.node:
        n.output[:] = [o if o != "o" else "o_mha" for o in n.output]
    then_g = helper.make_graph(list(mha.node), "then", [], [helper.make_tensor_value_info("o_mha", f, None)],
                               list(mha.initializer))
    gqa = helper.make_node("GroupQueryAttention", ["q", "k", "v", "", "", "seqlens_k", "total"], ["o_gqa", "pk", "pv"],
                           domain=MS, num_heads=heads, kv_num_heads=kv_heads, scale=1.0,
                           local_window_size=window or -1)
    else_g = helper.make_graph([gqa], "else", [], [helper.make_tensor_value_info("o_gqa", f, None)])
    node = helper.make_node("If", ["use_bias"], ["o"], then_branch=then_g, else_branch=else_g)
    g = helper.make_graph([node], name, [
        helper.make_tensor_value_info("q", f, [1, "S", heads * head_dim]),
        helper.make_tensor_value_info("k", f, [1, "S", kv_heads * head_dim]),
        helper.make_tensor_value_info("v", f, [1, "S", kv_heads * head_dim]),
        helper.make_tensor_value_info("bias", f, [1, 1, "S", "S"]),
        helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [1]),
        helper.make_tensor_value_info("total", TensorProto.INT32, []),
        helper.make_tensor_value_info("use_bias", TensorProto.BOOL, []),
    ], [helper.make_tensor_value_info("o", f, [1, "S", heads * head_dim])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid(MS, 1)])
    m.ir_version = 10
    return m


def mask(n, window, block):
    i, j = np.arange(n)[:, None], np.arange(n)[None, :]
    ok = j <= i
    b0, b1 = block
    inb = (i >= b0) & (i < b1) & (j >= b0) & (j < b1)
    ok = ok | inb
    if window:
        ok = ok & (np.abs(i - j) < window)
    return np.where(ok, 0.0, -3.4e38).astype(np.float32)


def reference(q, k, v, bias, heads, kv_heads, d):
    n = q.shape[1]
    qh = q.reshape(n, heads, d).transpose(1, 0, 2).astype(np.float64)
    kh = np.repeat(k.reshape(n, kv_heads, d).transpose(1, 0, 2), heads // kv_heads, 0).astype(np.float64)
    vh = np.repeat(v.reshape(n, kv_heads, d).transpose(1, 0, 2), heads // kv_heads, 0).astype(np.float64)
    s = qh @ kh.transpose(0, 2, 1) + bias[0, 0]
    s -= s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    return (p @ vh).transpose(1, 0, 2).reshape(1, n, heads * d).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check-len", type=int, default=600)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cfgs = {"sliding": (16, 8, 256, 1024), "global": (16, 1, 512, 0)}
    meta = {}
    rng = np.random.default_rng(0)
    for kind, (h, kvh, d, win) in cfgs.items():
        for fp16 in (False, True):
            name = f"mha_{kind}_{'fp16' if fp16 else 'fp32'}"
            onnx.save(mha_graph(name, h, kvh, d, fp16), a.out / f"{name}.onnx")
        ifm = if_graph(f"if_{kind}", h, kvh, d, win)
        onnx.save(ifm, a.out / f"if_{kind}.onnx")
        gq = onnx.ModelProto()
        gq.CopyFrom(ifm)
        else_g = gq.graph.node[0].attribute[1].g if gq.graph.node[0].attribute[1].name == "else_branch" else gq.graph.node[0].attribute[0].g
        gq.graph.ClearField("node")
        gq.graph.node.extend(else_g.node)
        gq.graph.node[0].output[0] = "o"
        onnx.save(gq, a.out / f"gqa_{kind}.onnx")
        n = a.check_len
        block = (40, 40 + 273)
        # q/k are post-RMSNorm in the model, so unit-scale values with scale 1.0 are realistic.
        q = rng.standard_normal((1, n, h * d), dtype=np.float32) / float(np.sqrt(np.sqrt(d)))
        k = rng.standard_normal((1, n, kvh * d), dtype=np.float32) / float(np.sqrt(np.sqrt(d)))
        v = rng.standard_normal((1, n, kvh * d), dtype=np.float32)
        bias = mask(n, win, block)[None, None]
        ref = reference(q, k, v, bias, h, kvh, d)
        for nm, arr in (("q", q), ("k", k), ("v", v), ("bias", bias), ("ref", ref)):
            arr.tofile(a.out / f"{kind}_{nm}.bin")
        meta[kind] = {"n": n, "heads": h, "kv_heads": kvh, "head_dim": d, "window": win, "block": block}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
