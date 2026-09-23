"""Run the vendored onnxruntime-genai model builder (PR #2473) against transformers 5.17.

transformers 5.17 guards per-layer attributes such as `head_dim` on heterogeneous configs and folds
`global_head_dim` / `num_global_key_value_heads` into `per_layer_config`. The builder reads the global
(sliding-layer) values and the two `global_*` attributes, so the guard is lifted and those are restored from the
first full-attention layer. Arguments are passed through to `builder.py`.
"""
import runpy
import sys
from pathlib import Path

import transformers

BUILDER = Path(__file__).resolve().parents[1] / "vendor" / "ortgenai_models_pr2473"

_from_pretrained = transformers.AutoConfig.from_pretrained.__func__


def _allow(config):
    for c in (config, *(getattr(config, k, None) for k in ("text_config", "vision_config", "audio_config"))):
        if c is None:
            continue
        object.__setattr__(c, "allow_global_per_layer_attribute_access", True)
        per_layer = getattr(c, "per_layer_config", None)
        if per_layer is not None and "full_attention" in getattr(c, "layer_types", []):
            full = per_layer[c.layer_types.index("full_attention")]
            c.global_head_dim = full.head_dim
            c.num_global_key_value_heads = full.num_key_value_heads
    return config


transformers.AutoConfig.from_pretrained = classmethod(lambda cls, *a, **k: _allow(_from_pretrained(cls, *a, **k)))

if __name__ == "__main__":
    sys.path.insert(0, str(BUILDER))
    sys.argv[0] = str(BUILDER / "builder.py")
    runpy.run_path(str(BUILDER / "builder.py"), run_name="__main__")
