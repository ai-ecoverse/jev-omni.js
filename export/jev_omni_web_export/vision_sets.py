"""kev-vision's image sets (vision-v1, vision-v2) as Jev-Omni requests.

Each kev item has an image, a neutral `context` and typed questions in kev.serve's shape with a `label`. The
conversion to Jev-Omni's (state, question, options) form is ours:

- image: the item's image, passed as Jev-Omni's `media` (one image before the text, as jev_omni.py does).
- state: the item's `context` (kev's `image` condition shows the image, then the context).
- question: the question's `instructions`.
- noul: options ["Yes", "No"]. kev labels noul 0 = no, 1 = yes, so the Jev label is 1 - kev label.
- choice: one option per criteria key in kev's order; "<key>: <description>" when the description is not null,
  else the key alone. The label is kev's (an index in criteria order).
- score: the criteria levels in kev's order, verbatim. The label is kev's level index.

The `caption` is not used. Output is one JSON request per line, the format predict.py and scripts/predict.ts read:
{id, set, family, type, image, state, question, options, label}.

  uv run python -m jev_omni_web_export.vision_sets --src ../../kev-vision/eval/vision-v1 --out ../build/eval/vision-v1.jsonl
"""
import argparse
import json
from pathlib import Path


def convert(src: Path):
    data = json.loads((src / "questions.json").read_text())
    for item in data["items"]:
        for key, q in item["questions"].items():
            if q["type"] == "noul":
                options, label = ["Yes", "No"], 1 - q["label"]
            elif q["type"] == "choice":
                options, label = [k if v is None else f"{k}: {v}" for k, v in q["criteria"].items()], q["label"]
            elif q["type"] == "score":
                options, label = list(q["criteria"]), q["label"]
            else:
                raise ValueError(q["type"])
            yield {"id": f"{item['id']}/{key}", "set": src.name, "family": item["family"], "type": q["type"],
                   "image": str((src / item["image"]).resolve()), "state": item["context"],
                   "question": q["instructions"], "options": options, "label": label}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="kev-vision eval set dir (questions.json, images/)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = list(convert(a.src))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{a.out}: {len(rows)} questions, {len({r['image'] for r in rows})} images")


if __name__ == "__main__":
    main()
