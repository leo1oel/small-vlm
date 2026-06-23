"""探查任意 HF VLM 的加载/结构,决定如何接入 sufmeanabl 探针(标准 kind 分支 vs 自定义 dir)。
Usage: python devtools/probe_arch.py <model_id> [trc]   (任意第2参数 => trust_remote_code=True)"""
import inspect
import sys

import torch
from transformers import AutoConfig

mid = sys.argv[1]
trc = len(sys.argv) > 2
cfg = AutoConfig.from_pretrained(mid, trust_remote_code=trc)
print("model_type:", getattr(cfg, "model_type", None))
print("architectures:", getattr(cfg, "architectures", None))
print("auto_map:", getattr(cfg, "auto_map", None))
for k in ("image_token_id", "image_token_index", "img_context_token_id", "image_seq_length"):
    print("  cfg.%s = %s" % (k, getattr(cfg, k, None)))
tc = getattr(cfg, "text_config", None)
print("text layers:", getattr(tc, "num_hidden_layers", None) if tc else None)
try:
    from transformers import AutoModel
    m = AutoModel.from_pretrained(mid, trust_remote_code=trc, torch_dtype=torch.bfloat16,
                                  attn_implementation="eager")
    print("LOADED class:", type(m).__name__)
    print("has lm_head:", hasattr(m, "lm_head"),
          "| language_model.lm_head:", hasattr(getattr(m, "language_model", None), "lm_head"))
    for path in ("language_model.model.layers", "model.language_model.layers",
                 "model.layers", "language_model.layers"):
        obj = m
        ok = True
        for p in path.split("."):
            obj = getattr(obj, p, None)
            if obj is None:
                ok = False
                break
        if ok and hasattr(obj, "__len__"):
            print("LAYERS at: %s  N=%d" % (path, len(obj)))
            break
    print("forward params:", list(inspect.signature(m.forward).parameters)[:14])
except Exception:
    import traceback
    traceback.print_exc()
