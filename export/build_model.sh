#!/usr/bin/env bash
# Build the Jev-Omni browser bundle and its parity checks from the pinned snapshot.
#
#   ./build_model.sh            # all steps
#   ./build_model.sh <step>...  # fetch | evalset | visionsets | vision | build | truncated | imageparity | package
#
# Disk: the bf16 source is 24 GB and the builder output ~14 GB; postprocess loads it and deletes it before writing
# the ~13.5 GB bundle, so the peak is ~40 GB. RAM: the builder peaks around 50-60 GB.
set -euo pipefail
cd "$(dirname "$0")"

REV=c050d51354147985d13286cf4acf90f562f2c631
SNAP=../build/Jev-Omni
ONNX=../build/onnx
EVAL=../build/eval
PUBLIC=../public/models/jev-omni
PY="uv run python"
KEV_VISION=../../kev-vision/eval
PARITY_IDS="shapes-0/count chart-5/compare astronaut/job inbox-0/unread dashboard-1/down line-2/march departures-0/gate terms-2/refund"

fetch() {
  mkdir -p "$SNAP/unified"
  $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('akhilaaa3/Jev-Omni', revision='$REV', local_dir='$SNAP',
  allow_patterns=['unified/*.json', 'unified/*.jinja', 'head.pt', 'runtime_buffers.pt', 'decision_config.json',
                  'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'load_model.py', 'jev_omni.py', 'sha256.json'])"
  # the 24 GB file through curl: resumable, and the Hub's parallel client stalled on this link
  until curl -fsSL --retry 20 --retry-all-errors -C - -o "$SNAP/unified/model.safetensors" \
      "https://huggingface.co/akhilaaa3/Jev-Omni/resolve/$REV/unified/model.safetensors"; do sleep 10; done
  cp "$SNAP/tokenizer.json" "$SNAP/unified/tokenizer.json"
  echo "d78782f3b9c3302353a7d1a1b6277fa5e05d692f4a0cb2b862797026c65eabe6  $SNAP/unified/model.safetensors" | shasum -a 256 -c
}

evalset() {
  $PY -m jev_omni_web_export.decisionbench --tokenizer "$SNAP" --out "$EVAL/decisionbench-medium.json"
}

visionsets() {
  # kev-vision's two image sets, converted to Jev requests (mapping documented in vision_sets.py)
  for v in vision-v1 vision-v2; do $PY -m jev_omni_web_export.vision_sets --src "$KEV_VISION/$v" --out "$EVAL/$v.jsonl"; done
}

vision() {
  # fp32 vision embedder (SigLIP-style encoder + projection), exported from the vision keys only
  rm -rf "$ONNX/vision"
  $PY -m jev_omni_web_export.vision_export --hf "$SNAP/unified" --out "$ONNX/vision"
}

build() {
  # On macOS the builder can abort with `recursive_mutex lock failed` after writing everything (as in kev.js),
  # so success is judged by genai_config.json.
  rm -rf "$ONNX/q8f32-raw"
  $PY -m jev_omni_web_export.build -i "$SNAP/unified" -o "$ONNX/q8f32-raw" -p int8 -e webgpu -c "$ONNX/cache" \
    --extra_options exclude_lm_head=true use_webgpu_fp32=true || true
  test -f "$ONNX/q8f32-raw/genai_config.json"
  $PY -m jev_omni_web_export.postprocess --src "$ONNX/q8f32-raw" --out "$ONNX/q8f32" --global-attn-fp16 --delete-src-data
}

truncated() {
  rm -rf "$ONNX/fp32-6l"
  $PY -m jev_omni_web_export.build -i "$SNAP/unified" -o "$ONNX/fp32-6l" -p fp32 -e cpu -c "$ONNX/cache" \
    --extra_options exclude_lm_head=true num_hidden_layers=6 || true
  test -f "$ONNX/fp32-6l/genai_config.json"
  $PY -m jev_omni_web_export.postprocess --src "$ONNX/fp32-6l"
  $PY -m jev_omni_web_export.parity_truncated --hf "$SNAP/unified" --buffers "$SNAP/runtime_buffers.pt" \
    --onnx "$ONNX/fp32-6l" --layers 6 --evalset "$EVAL/decisionbench-medium.json" --out "$EVAL/parity-fp32-6l.json"
}

imageparity() {
  # 8 image questions across families + one text-only copy, through the truncated fp32 graph and the vision graph
  $PY - "$EVAL" $PARITY_IDS <<'PYEOF'
import json, sys
ev, ids = sys.argv[1], sys.argv[2:]
rows = {r["id"]: r for v in ("vision-v1", "vision-v2") for r in map(json.loads, open(f"{ev}/{v}.jsonl"))}
reqs = [rows[i] for i in ids]
text = {k: v for k, v in rows["inbox-1/top"].items() if k != "image"}
reqs.append(text | {"id": "inbox-1/top/text"})
open(f"{ev}/parity-requests.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in reqs))
PYEOF
  # package moves the vision graph into the bundle, so after a package step read it from there
  local vis="$ONNX/vision/vision.onnx"
  [ -f "$vis" ] || vis="$PUBLIC/r-${REV:0:7}/vision/vision.onnx"
  $PY -m jev_omni_web_export.parity_image --hf "$SNAP/unified" --tokenizer "$SNAP" --buffers "$SNAP/runtime_buffers.pt" \
    --graph "$ONNX/fp32-6l/model.onnx" --vision "$vis" --layers 6 \
    --requests "$EVAL/parity-requests.jsonl" --limit 9 --out "$EVAL/parity-image-fp32-6l.json"
}

package() {
  # rerunning after a package: VARIANT_DIR=$PUBLIC/r-<rev>/q8f32 VISION_DIR=$PUBLIC/r-<rev>/vision refresh in place
  $PY -m jev_omni_web_export.package --snapshot "$SNAP" --variant-dir "${VARIANT_DIR:-$ONNX/q8f32}" --variant q8f32 \
    --out "$PUBLIC" --vision-dir "${VISION_DIR:-$ONNX/vision}" ${PARITY:+--parity "$PARITY"} \
    ${VISION_PARITY:+--vision-parity "$VISION_PARITY"}
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(fetch evalset visionsets vision build truncated imageparity package)
for s in "${steps[@]}"; do echo "== $s"; "$s"; done
