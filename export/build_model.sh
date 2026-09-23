#!/usr/bin/env bash
# Build the Jev-Omni browser bundle and its parity checks from the pinned snapshot.
#
#   ./build_model.sh            # all steps
#   ./build_model.sh <step>...  # fetch | evalset | build | truncated | package
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

package() {
  $PY -m jev_omni_web_export.package --snapshot "$SNAP" --variant-dir "$ONNX/q8f32" --variant q8f32 --out "$PUBLIC" \
    ${PARITY:+--parity "$PARITY"}
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(fetch evalset build truncated package)
for s in "${steps[@]}"; do echo "== $s"; "$s"; done
