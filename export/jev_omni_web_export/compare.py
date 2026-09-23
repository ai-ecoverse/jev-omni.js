"""Score probability files (JSON lines: id, probs, ms) against the eval set's labels and against each other.

Accuracy is reported both per question (micro) and state-macro (DecisionBench's headline aggregation). ECE uses ten
equal-width bins over the top probability, as the model card does. |Δp| is taken over every option of every
question; a flip is a question whose argmax differs from the reference's.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path: Path):
    return {r["id"]: r for r in map(json.loads, path.open())}


def scores(records, run):
    rs = [r for r in records if r["id"] in run]
    correct = [int(np.argmax(run[r["id"]]["probs"]) == r["label"]) for r in rs]
    brier = [float(np.sum((np.array(run[r["id"]]["probs"]) - np.eye(len(r["options"]))[r["label"]]) ** 2)) for r in rs]
    conf = np.array([max(run[r["id"]]["probs"]) for r in rs])
    ok = np.array(correct)
    ece = 0.0
    for lo in np.arange(0, 1, 0.1):
        m = (conf > lo) & (conf <= lo + 0.1) if lo > 0 else (conf <= 0.1)
        if m.any():
            ece += m.mean() * abs(ok[m].mean() - conf[m].mean())
    by_state = defaultdict(list)
    for r, c in zip(rs, correct):
        by_state[r["state_id"]].append(c)
    ms = [run[r["id"]]["ms"] for r in rs if "ms" in run[r["id"]]]
    return {"n": len(rs), "accuracy_micro": float(np.mean(correct)),
            "accuracy_state_macro": float(np.mean([np.mean(v) for v in by_state.values()])),
            "brier": float(np.mean(brier)), "ece": float(ece), "mean_confidence": float(conf.mean()),
            "median_ms": float(np.median(ms)) if ms else None}


def diff(records, run, ref):
    ids = [r["id"] for r in records if r["id"] in run and r["id"] in ref]
    d = [np.abs(np.array(run[i]["probs"]) - np.array(ref[i]["probs"])) for i in ids]
    flips = [i for i in ids if np.argmax(run[i]["probs"]) != np.argmax(ref[i]["probs"])]
    return {"n": len(ids), "mean_abs_dp": float(np.mean(np.concatenate(d))), "max_abs_dp": float(max(x.max() for x in d)),
            "flips": len(flips), "flipped": flips}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evalset", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("runs", type=Path, nargs="*")
    a = ap.parse_args()
    records = json.loads(a.evalset.read_text())["records"]
    ref = load(a.ref)
    out = {"reference": {"file": str(a.ref), **scores(records, ref)}}
    for p in a.runs:
        run = load(p)
        out[p.stem] = {"file": str(p), **scores(records, run), "vs_reference": diff(records, run, ref)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
