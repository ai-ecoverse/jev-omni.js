"""Score image-set runs (predict.py / scripts/predict.ts output: JSONL with label, family, type, probs, timings) the
way kev-vision reports its sets: accuracy with a 95% interval from resampling images, Brier summed over options,
per type and per family, plus flips and |Δp| against a reference run and median latencies.

  uv run python -m jev_omni_web_export.vision_report --ref torch.jsonl [run.jsonl ...] [--out report.json]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path: Path):
    return {r["id"]: r for r in map(json.loads, path.open()) if r}


def score(rows, seed=0, draws=2000):
    correct = np.array([int(np.argmax(r["probs"]) == r["label"]) for r in rows])
    brier = np.array([float(np.sum((np.array(r["probs"]) - np.eye(len(r["options"]))[r["label"]]) ** 2)) for r in rows])
    images = sorted({r["image"] for r in rows})
    by_image = defaultdict(list)
    for k, r in enumerate(rows):
        by_image[r["image"]].append(k)
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(draws):
        idx = np.concatenate([by_image[images[i]] for i in rng.integers(0, len(images), len(images))])
        accs.append(correct[idx].mean())
    out = {"n": len(rows), "images": len(images), "accuracy": float(correct.mean()),
           "accuracy_ci95": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
           "brier": float(brier.mean()), "mean_top_p": float(np.mean([max(r["probs"]) for r in rows]))}
    for key in ("type", "family"):
        groups = defaultdict(list)
        for r, c in zip(rows, correct):
            groups[r[key]].append(int(c))
        out[f"by_{key}"] = {g: {"n": len(v), "accuracy": float(np.mean(v))} for g, v in sorted(groups.items())}
    for t in ("embed_ms", "decoder_ms", "ms"):
        vals = [r[t] for r in rows if t in r]
        if vals:
            out[f"median_{t}"] = float(np.median(vals))
    out["median_tokens"] = float(np.median([r["tokens"] for r in rows]))
    out["errors"] = [{"id": r["id"], "answer": r["options"][int(np.argmax(r["probs"]))], "label": r["options"][r["label"]],
                      "p": round(float(max(r["probs"])), 3)} for r, c in zip(rows, correct) if not c]
    return out


def diff(run, ref):
    ids = [i for i in run if i in ref]
    d = [np.abs(np.array(run[i]["probs"]) - np.array(ref[i]["probs"])) for i in ids]
    flips = [i for i in ids if np.argmax(run[i]["probs"]) != np.argmax(ref[i]["probs"])]
    return {"n": len(ids), "mean_abs_dp": float(np.mean(np.concatenate(d))), "max_abs_dp": float(max(x.max() for x in d)),
            "flips": len(flips), "flipped": flips}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("runs", type=Path, nargs="*")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    ref = load(a.ref)
    report = {a.ref.stem: {"file": str(a.ref), **score(list(ref.values()))}}
    for p in a.runs:
        run = load(p)
        report[p.stem] = {"file": str(p), **score(list(run.values())), "vs_reference": diff(run, ref)}
    text = json.dumps(report, indent=1)
    if a.out:
        a.out.write_text(text)
    for name, r in report.items():
        line = (f"{name}: acc {r['accuracy']:.3f} [{r['accuracy_ci95'][0]:.3f}, {r['accuracy_ci95'][1]:.3f}] "
                f"brier {r['brier']:.3f} n={r['n']}")
        if "vs_reference" in r:
            v = r["vs_reference"]
            line += f" | flips {v['flips']} mean|dp| {v['mean_abs_dp']:.4f} max|dp| {v['max_abs_dp']:.3f}"
        line += f" | median embed {r.get('median_embed_ms', 0):.0f} ms decoder {r.get('median_decoder_ms', 0):.0f} ms"
        print(line)
        print("   by type:", {k: round(v["accuracy"], 3) for k, v in r["by_type"].items()})
        print("   by family:", {k: round(v["accuracy"], 3) for k, v in r["by_family"].items()})


if __name__ == "__main__":
    main()
