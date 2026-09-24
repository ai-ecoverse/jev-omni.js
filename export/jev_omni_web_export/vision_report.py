"""Score image-set runs (predict.py / scripts/predict.ts output: JSONL with label, family, type, probs, timings) the
way kev-vision reports its sets: accuracy with a 95% interval from resampling images, Brier summed over options,
per type, per family and counting vs the rest, plus flips and |Δp| against a reference run, median latencies and,
with --kev, a per-question comparison with a Kev run on the same set (kev-vision results/<model>/rows.json).

  uv run python -m jev_omni_web_export.vision_report --ref torch.jsonl [run.jsonl ...] [--kev rows.json] [--out report.json]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


# questions that are answered by counting objects in the image ("how many ...", "more blue than green dots?");
# the rest read a value, a label or a scene. v1's "total" is a receipt amount, v2's "total" counts dots.
COUNTING = {"vision-v1": {"count", "done"}, "vision-v2": {"above60", "over50", "red", "refunded", "total", "unread", "more"}}


def skill(r):
    return "counting" if r["id"].split("/")[1] in COUNTING.get(r.get("set"), ()) else "other"


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
    for key in ("type", "family", "skill"):
        groups = defaultdict(list)
        for r, c in zip(rows, correct):
            groups[skill(r) if key == "skill" else r[key]].append(int(c))
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


def vs_kev(run, kev_rows):
    """Paired correctness against Kev's image condition. Option lists differ for noul (see vision_sets.py), so only
    correctness is compared, not probabilities."""
    kev = {f"{k['item']}/{k['question']}": int(np.argmax(k["p"]) == k["label"]) for k in kev_rows if k["condition"] == "image"}
    ids = [i for i in run if i in kev]
    jev = {i: int(np.argmax(run[i]["probs"]) == run[i]["label"]) for i in ids}
    out = {"n": len(ids), "kev_accuracy": float(np.mean([kev[i] for i in ids])),
           "both": sum(jev[i] and kev[i] for i in ids), "only_jev": sum(jev[i] and not kev[i] for i in ids),
           "only_kev": sum(kev[i] and not jev[i] for i in ids), "neither": sum(not jev[i] and not kev[i] for i in ids)}
    counting = [i for i in ids if skill(run[i]) == "counting"]
    if counting:
        out["counting"] = {"n": len(counting), "jev": float(np.mean([jev[i] for i in counting])),
                           "kev": float(np.mean([kev[i] for i in counting]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("runs", type=Path, nargs="*")
    ap.add_argument("--kev", type=Path, help="kev-vision results/<model>/rows.json for the same set")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    kev_rows = json.loads(a.kev.read_text()) if a.kev else None
    ref = load(a.ref)
    report = {a.ref.stem: {"file": str(a.ref), **score(list(ref.values()))}}
    for p in a.runs:
        run = load(p)
        report[p.stem] = {"file": str(p), **score(list(run.values())), "vs_reference": diff(run, ref)}
    if kev_rows:
        for name, p in [(a.ref.stem, a.ref), *((p.stem, p) for p in a.runs)]:
            report[name]["vs_kev"] = vs_kev(load(p), kev_rows)
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
        print("   by skill:", {k: f"{v['accuracy']:.3f} (n={v['n']})" for k, v in r["by_skill"].items()})
        if "vs_kev" in r:
            print("   vs kev:", r["vs_kev"])


if __name__ == "__main__":
    main()
