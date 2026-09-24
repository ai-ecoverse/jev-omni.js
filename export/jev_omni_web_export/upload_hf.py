"""Publish the packaged bundle to a Hugging Face model repo (default ai-ecoverse/jev-omni.js, folder jev-omni/).

    HF_TOKEN=... uv run python -m jev_omni_web_export.upload_hf [--bundle ../public/models/jev-omni] [--dry-run]

The browser runtime reads <base>/jev-omni/manifest.json and the files it names. Upload order makes the switch atomic
for clients: first the files under r-<rev>/ (nothing a live manifest points at is replaced by different bytes of a
different size mid-download), then manifest.json alone, then whatever the new manifest no longer names is deleted.
Finally every remote file is checked against the manifest's size and the remote manifest against the local one.
A model card (README.md) is generated from the manifest."""
import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

CARD = """---
license: apache-2.0
library_name: jev-omni.js
pipeline_tag: image-text-to-text
base_model:
- {base_repo}
- {jev_repo}
tags: [jev-omni, gemma4, decision-model, classifier, onnx, onnxruntime-web, webgpu, int8]
---

# jev-omni.js weights

A browser-ready export of [Jev-Omni]({jev_url}), akhilaaa3's Gemma 4 12B decision classifier: a state, a question and
2-256 options in, one probability per option out, from a single forward pass. It runs in Chrome on WebGPU through
[jev-omni.js](https://github.com/ai-ecoverse/jev-omni.js), for text and images.
[Live demo](https://ai-ecoverse.github.io/jev-omni.js/).

This repo only holds converted weights. The model, its training, its decision head and its prompt format are
Jev-Omni's. Jev-Omni states that it is independent of TypeSafe AI's Jev, and so is this repo.

```js
import * as ort from "onnxruntime-web/webgpu";   // {ort_version}: see Requirements
import {{ loadJevOmni }} from "@ai-ecoverse/jev-omni.js";

const jev = await loadJevOmni("https://huggingface.co/{repo}/resolve/main/jev-omni", {{ ort }});
const res = await jev.predict({{
  state: "The meeting starts at 10 AM. It is now 9 AM.",
  question: "Has the meeting started?",
  options: ["Yes", "No"],
}});
res.probabilities;   // {{ Yes: ..., No: ... }}
```

## Contents

| Path | What | Size |
|---|---|---|
| `jev-omni/manifest.json` | file list with sizes and SHA-256, graph inputs, parity numbers; `revision` {revision} | |
| `jev-omni/{rev_dir}/q8f32/` | decoder: int8 weights (MatMulNBits, block 32), fp32 activations, fp16 attention in the 8 global layers; {n_decoder} files | {decoder_gb:.2f} GB |
| `jev-omni/{rev_dir}/vision/` | Gemma 4's vision embedder and projection, fp32 | {vision_gb:.2f} GB |
| `jev-omni/{rev_dir}/head.safetensors`, `tokenizer*.json` | the 256-way decision head, the tokenizer | {small_mb:.0f} MB |

Total download: {total_gb:.2f} GB in {n_files} files of at most 32 MB.

## Provenance

- Model: [{jev_repo}]({jev_url}) at `{jev_rev}` (Apache-2.0), `unified/model.safetensors` (bf16) and `head.pt`.
- Base: [{base_repo}](https://huggingface.co/{base_repo}) at `{base_rev}` (Apache-2.0).
- Export: a vendored draft of the onnxruntime-genai model builder with Gemma 4 support
  ([microsoft/onnxruntime-genai#{pull}](https://github.com/microsoft/onnxruntime-genai/pull/{pull}), `{builder_commit}`),
  without the LM head, then graph rewrites for WebGPU and the image path. Details are in the
  [jev-omni.js docs](https://github.com/ai-ecoverse/jev-omni.js/tree/main/docs).

## Parity with the fp32 PyTorch model

Chrome on WebGPU (Apple M4 Max) against Jev-Omni in fp32 PyTorch:

| Set | Accuracy, browser / reference | Answers changed | Mean / max abs. probability difference |
|---|---|---|---|
| DecisionBench medium, {db_n} text questions | {db_acc:.2%} / {db_ref:.2%} | {db_flips} | {db_mean:.4f} / {db_max:.3f} |
| kev.js vision-v1, {v1_n} image questions | {v1_acc:.2%} / {v1_ref:.2%} | {v1_flips} | {v1_mean:.4f} / {v1_max:.3f} |
| kev.js vision-v2, {v2_n} image questions | {v2_acc:.2%} / {v2_ref:.2%} | {v2_flips} | {v2_mean:.4f} / {v2_max:.3f} |

Without quantization, a 6-layer fp32 export matches PyTorch to a relative error of {rel6:.1e} at the last position.

## Requirements

- **Memory:** about 16 GB of GPU memory for the WebGPU session. While loading, the tab also holds the 13.6 GB of
  files in memory until they are on the GPU. On Apple Silicon that means 32 GB of unified memory at the very least,
  and 64 GB or more to be comfortable. It has been tested on an Apple M4 Max with 128 GB only.
- **Download:** 13.6 GB on the first load. jev-omni.js keeps it in Cache Storage, so later loads read from disk.
- **onnxruntime-web `{ort_version}`:** its attention needs an n×n fp32 buffer per head, so prompts are capped at
  about 8,000 tokens on GPUs with 4 GB buffers (Apple Silicon) and fewer on GPUs with smaller ones. jev-omni.js
  rejects longer prompts with a clear error. The parity numbers above were measured with a 1.31 development build,
  whose sliding-window attention runs prompts up to 9k tokens; on prompts of 1-8k tokens, 1.30 gives the same
  answers (max. probability difference to 1.31: 0.019).
- **Latency** (M4 Max): about 1.2-1.6 s per image question, about 7 s for a text question under 2k tokens.
"""


def published_files(manifest):
    f = manifest["files"]
    paths = ["manifest.json", f["tokenizer"], f["tokenizer_config"], f["head"]]
    for v in manifest["variants"].values():
        paths += [v["model"], *v["data"]]
    if "vision" in manifest:
        paths += [manifest["vision"]["model"], *manifest["vision"]["data"]]
    return paths


def card(manifest, repo, ort_version):
    sizes = manifest["sizes"]
    v = manifest["variants"]["q8f32"]
    vis = manifest["vision"]
    decoder = [v["model"], *v["data"]]
    vision = [vis["model"], *vis["data"]]
    small = [manifest["files"][k] for k in ("tokenizer", "tokenizer_config", "head")]
    db = v["parity"]["browser-q8f32"]
    vp = vis["parity"]
    files = published_files(manifest)[1:]
    return CARD.format(
        repo=repo, revision=f"`{manifest['revision']}`", rev_dir=v["model"].split("/")[0], ort_version=ort_version,
        jev_repo=manifest["model"]["repo"], jev_rev=manifest["model"]["revision"], jev_url=f"https://huggingface.co/{manifest['model']['repo']}",
        base_repo=manifest["base"]["repo"], base_rev=manifest["base"]["revision"],
        pull=manifest["builder"]["pull"], builder_commit=manifest["builder"]["commit"][:7],
        n_decoder=len(decoder), decoder_gb=sum(sizes[p] for p in decoder) / 1e9, vision_gb=sum(sizes[p] for p in vision) / 1e9,
        small_mb=sum(sizes[p] for p in small) / 1e6, total_gb=sum(sizes[p] for p in files) / 1e9, n_files=len(files),
        db_n=db["n"], db_acc=db["accuracy_micro"], db_ref=v["parity"]["reference"]["accuracy_micro"], db_flips=db["vs_reference"]["flips"],
        db_mean=db["vs_reference"]["mean_abs_dp"], db_max=db["vs_reference"]["max_abs_dp"],
        **{f"{k}_{f}": vp[f"kev-vision-{k}"][g] for k in ("v1", "v2")
           for f, g in (("n", "n"), ("acc", "accuracy"), ("ref", "reference_accuracy"), ("flips", "flips"), ("mean", "mean_abs_dp"), ("max", "max_abs_dp"))},
        rel6=vp["fp32_6_layer"]["worst_rel_last"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parents[2] / "public/models/jev-omni")
    ap.add_argument("--repo", default="ai-ecoverse/jev-omni.js")
    ap.add_argument("--folder", default="jev-omni")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    manifest = json.loads((a.bundle / "manifest.json").read_text())
    pkg = json.loads((Path(__file__).resolve().parents[2] / "package.json").read_text())
    text = card(manifest, a.repo, pkg["devDependencies"]["onnxruntime-web"])
    files = published_files(manifest)
    missing = [p for p in files[1:] if not (a.bundle / p).is_file() or (a.bundle / p).stat().st_size != manifest["sizes"][p]]
    if missing:
        raise SystemExit(f"{len(missing)} files missing or of the wrong size locally, e.g. {missing[:3]}")
    total = sum(manifest["sizes"][p] for p in files[1:])
    print(f"{a.repo}/{a.folder}: {len(files)} files, {total / 1e9:.2f} GB, revision {manifest['revision']}")
    if a.dry_run:
        print(text)
        return

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(a.repo, repo_type="model", private=a.private, exist_ok=True)
    api.upload_file(path_or_fileobj=text.encode(), path_in_repo="README.md", repo_id=a.repo, repo_type="model",
                    commit_message="Model card")
    remote = set(api.list_repo_files(a.repo))
    # 1. the revision's files; upload_large_folder hashes, resumes and parallelises, and skips files already there.
    #    It uploads a folder's paths as they are, so the bundle's parent is the root and the folder name the prefix.
    payload = [f"{a.folder}/{p}" for p in files[1:]]
    if a.bundle.name != a.folder:
        raise SystemExit(f"--bundle must be a directory named {a.folder}")
    api.upload_large_folder(repo_id=a.repo, repo_type="model", folder_path=a.bundle.parent, allow_patterns=payload,
                            num_workers=a.workers)
    # 2. the manifest alone: this commit switches clients to the new revision
    api.upload_file(path_or_fileobj=str(a.bundle / "manifest.json"), path_in_repo=f"{a.folder}/manifest.json", repo_id=a.repo,
                    repo_type="model", commit_message=f"{a.folder}: revision {manifest['revision']}")
    # 3. what the manifest no longer names
    keep = {f"{a.folder}/{p}" for p in files}
    stale = sorted(f for f in remote if f.startswith(f"{a.folder}/") and f not in keep)
    if stale:
        api.delete_files(repo_id=a.repo, repo_type="model", delete_patterns=stale,
                         commit_message=f"{a.folder}: drop {len(stale)} superseded files")
        print(f"removed {len(stale)} superseded files")

    # 4. verify: every file present at its size, and the remote manifest identical to the local one
    info = {s.path: s.size for s in api.list_repo_tree(a.repo, path_in_repo=a.folder, recursive=True) if hasattr(s, "size")}
    bad = [p for p in files[1:] if info.get(f"{a.folder}/{p}") != manifest["sizes"][p]]
    got = Path(hf_hub_download(a.repo, f"{a.folder}/manifest.json", token=os.environ["HF_TOKEN"], force_download=True)).read_bytes()
    same = got == (a.bundle / "manifest.json").read_bytes()
    print(f"verify: {len(files) - 1 - len(bad)}/{len(files) - 1} files at their size, manifest {'identical' if same else 'DIFFERENT'}")
    if bad or not same:
        raise SystemExit(f"verification failed: {bad[:5]}")
    print(f"https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
