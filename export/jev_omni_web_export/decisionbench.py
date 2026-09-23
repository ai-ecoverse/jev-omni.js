"""DecisionBench (akhilaaa3/decision-bench) as Jev-Omni (state, question, options) records with token ids.

DecisionBench stores typed questions (`noul`, `choice`, `score`, each with criteria). Jev-Omni's model card does
not say how they were turned into options, so this mapping is ours and is fixed here:

- state: the row's `state` field as stored (a JSON-encoded list of events), unchanged.
- question: the question's `instructions`.
- noul: options "True: <criteria.true>", "False: <criteria.false>"; the answer is True/False.
- choice: one option per criteria key, "<key>: <description>", in the stored order; the answer is a key.
- score: the criteria list in order, verbatim; the answer is the 0-based index.

Parity between the browser and the reference does not depend on the mapping (both see the same token ids), but
accuracy is only comparable to the published 87.57% if it matches the authors' mapping.
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

DATASET = "akhilaaa3/decision-bench"
DATASET_REVISION = "19334fec40b54b693a63e1ffd91636651d39e847"


def prompt(state, question, options):
    """load_model.predict's prompt, verbatim."""
    choices = "\n".join(f"{i + 1}. {v}" for i, v in enumerate(options))
    return (f"{state}\n\n---\n\nQUESTION: {question}\n\nOPTIONS:\n{choices}\n\n"
            f"Reply with only the number of the correct option (1-{len(options)}).\n"
            "Output a single number and nothing else.")


def encode(tokenizer, text):
    """load_model.predict's chat-template call (the first variant it tries, which Jev-Omni's template accepts)."""
    ids = tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                        tokenize=True, enable_thinking=False)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return list(ids)


def to_records(row):
    questions, answers = json.loads(row["questions"]), json.loads(row["answers"])
    for key, q in questions.items():
        c, y = q["criteria"], answers[key]
        if q["type"] == "noul":
            options, label = [f"True: {c['true']}", f"False: {c['false']}"], 0 if y is True else 1
        elif q["type"] == "choice":
            keys = list(c)
            options, label = [f"{k}: {v}" for k, v in c.items()], keys.index(y)
        elif q["type"] == "score":
            options, label = list(c), int(y)
        else:
            raise ValueError(q["type"])
        yield {"id": f"{row['id']}/{key}", "state_id": row["id"], "type": q["type"], "state": row["state"],
               "question": q["instructions"], "options": options, "label": label}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="Jev-Omni snapshot dir (tokenizer.json, chat_template.jinja)")
    ap.add_argument("--subset", default="medium", choices=["medium", "hard"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    path = hf_hub_download(DATASET, f"data/{a.subset}.jsonl", repo_type="dataset", revision=DATASET_REVISION)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    records = []
    for line in open(path):
        for r in to_records(json.loads(line)):
            r["ids"] = encode(tok, prompt(r["state"], r["question"], r["options"]))
            records.append(r)
    lengths = sorted(len(r["ids"]) for r in records)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"dataset": DATASET, "revision": DATASET_REVISION, "subset": a.subset,
                                 "records": records}))
    print(f"{len(records)} questions from {len({r['state_id'] for r in records})} states; tokens "
          f"min {lengths[0]} median {lengths[len(lengths) // 2]} p90 {lengths[int(len(lengths) * .9)]} max {lengths[-1]}; "
          f"options max {max(len(r['options']) for r in records)}")


if __name__ == "__main__":
    main()
