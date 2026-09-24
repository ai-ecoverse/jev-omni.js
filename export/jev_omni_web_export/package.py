"""Lay out a browser bundle: manifest.json at the root, then r-<rev>/{tokenizer.json, tokenizer_config.json,
head.safetensors}, r-<rev>/<variant>/{model.onnx, model.onnx.data*} and, with --vision-dir,
r-<rev>/vision/{vision.onnx, vision.onnx.data*} (shared by all variants).

The manifest pins the Jev-Omni and base revisions, lists every file with its size and SHA-256 (the loader checks
both), records the graph's inputs and the head's shape, and carries parity numbers when given. Variants are moved
into place (not copied), so a 13 GB bundle is never on disk twice; running it again for another variant merges.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import onnx
import torch
from safetensors.torch import save_file

JEV = {"repo": "akhilaaa3/Jev-Omni", "revision": "c050d51354147985d13286cf4acf90f562f2c631"}
BASE = {"repo": "google/gemma-4-12B-it", "revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"}
BUILDER = {"repo": "microsoft/onnxruntime-genai", "pull": 2473, "commit": "603c36c57110e347a86ae87648f7dc1b1d8d8124"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True, help="Jev-Omni snapshot (head.pt, tokenizer files)")
    ap.add_argument("--variant-dir", type=Path, required=True, help="postprocess output")
    ap.add_argument("--variant", required=True, help="e.g. q8f32")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parity", type=Path, help="compare.py output to record")
    ap.add_argument("--revision", default=JEV["revision"])
    ap.add_argument("--head", type=Path, help="head.pt, default <snapshot>/head.pt")
    ap.add_argument("--vision-dir", type=Path, help="vision_export output (vision.onnx, vision.onnx.data*)")
    ap.add_argument("--vision-parity", type=Path, help="parity_image.py output to record for the image path")
    a = ap.parse_args()
    rev = f"r-{a.revision[:7]}"
    root = a.out / rev
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = a.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "name": "jev-omni", "model": {**JEV, "revision": a.revision}, "base": BASE, "builder": BUILDER,
        "files": {}, "sizes": {}, "sha256": {}, "variants": {}}

    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(a.snapshot / f, root / f)
    head = torch.load(a.head or a.snapshot / "head.pt", map_location="cpu", weights_only=True)
    save_file({k: v.float().contiguous() for k, v in head.items()}, root / "head.safetensors")
    manifest["files"] = {"tokenizer": f"{rev}/tokenizer.json", "tokenizer_config": f"{rev}/tokenizer_config.json",
                         "head": f"{rev}/head.safetensors"}
    manifest["head"] = {"hidden": int(head["linear.weight"].shape[1]), "classes": int(head["linear.weight"].shape[0])}

    vdir = root / a.variant
    data = sorted((p for p in a.variant_dir.iterdir() if p.name.startswith("model.onnx.data")),
                  key=lambda p: int(p.name.rpartition("_")[2]) if "_" in p.name else 0)
    if a.variant_dir.resolve() != vdir.resolve():   # else: refresh the manifest of a variant already in place
        # the source is moved, not copied: a rerun against an emptied source must not wipe the bundle
        if not (a.variant_dir / "model.onnx").exists():
            raise SystemExit(f"{a.variant_dir}/model.onnx missing (already packaged? pass --variant-dir {vdir})")
        if vdir.exists():
            shutil.rmtree(vdir)
        vdir.mkdir()
        for p in [a.variant_dir / "model.onnx", *data]:
            shutil.move(str(p), vdir / p.name)
    if a.vision_dir:
        vis = root / "vision"
        if a.vision_dir.resolve() != vis.resolve():
            if not (a.vision_dir / "vision.onnx").exists():
                raise SystemExit(f"{a.vision_dir}/vision.onnx missing (already packaged? pass --vision-dir {vis})")
            if vis.exists():
                shutil.rmtree(vis)
            vis.mkdir()
            for p in a.vision_dir.iterdir():
                if p.name.startswith("vision.onnx"):
                    shutil.move(str(p), vis / p.name)
        vdata = sorted((p for p in vis.iterdir() if p.name.startswith("vision.onnx.data")),
                       key=lambda p: int(p.name.rpartition("_")[2]) if "_" in p.name else 0)
        vg = onnx.load(str(vis / "vision.onnx"), load_external_data=False).graph
        manifest["vision"] = {"model": f"{rev}/vision/vision.onnx", "data": [f"{rev}/vision/{p.name}" for p in vdata],
                              "inputs": [i.name for i in vg.input], "outputs": [o.name for o in vg.output]}
        if a.vision_parity:
            manifest["vision"]["parity"] = json.loads(a.vision_parity.read_text())
    g = onnx.load(str(vdir / "model.onnx"), load_external_data=False).graph
    manifest["variants"][a.variant] = {
        "model": f"{rev}/{a.variant}/model.onnx",
        "data": [f"{rev}/{a.variant}/{p.name}" for p in data],
        "inputs": [i.name for i in g.input], "outputs": [o.name for o in g.output],
    }
    if a.parity:
        manifest["variants"][a.variant]["parity"] = json.loads(a.parity.read_text())

    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(a.out))
            manifest["sizes"][rel] = p.stat().st_size
            manifest["sha256"][rel] = sha256(p)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    total = sum(manifest["sizes"][f] for f in [manifest["variants"][a.variant]["model"], *manifest["variants"][a.variant]["data"]])
    print(f"{manifest_path}: {a.variant} {len(data)} data files, {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
