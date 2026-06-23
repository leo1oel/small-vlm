"""Diagnostic: for each VLM, count how many image tokens the extractor's img_id matches
on one real ImageNet image, and where vision features sit. Tells us if the low/flat probe
(ov15 ~0.15) is a token-identification bug vs a genuine representation issue.

Usage: python devtools/diag_imgtok.py <kind> <model_id>
"""
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor

DEV = "cuda"
ROOT = Path(__file__).resolve().parents[1] / "neo_analysis"


def load_one_image():
    import glob

    import pyarrow as pa
    import pyarrow.ipc as ipc
    base = "/gscratch/krishna/leoym/hf_cache/datasets/mrm8488___image_net1_k-val"
    p = sorted(glob.glob(f"{base}/**/*.arrow", recursive=True))[0]
    with pa.memory_map(p, "r") as s:
        try:
            t = ipc.open_stream(s).read_all()
        except pa.lib.ArrowInvalid:
            s.seek(0); t = ipc.open_file(s).read_all()
    return t.to_pylist()[0]


def main():
    kind, model_id = sys.argv[1], sys.argv[2]
    trc = kind == "onevision15"
    if trc:
        import transformers.modeling_rope_utils as _R
        if "default" not in _R.ROPE_INIT_FUNCTIONS:
            def _d(config, device=None, seq_len=None, **kw):
                base = getattr(config, "rope_theta", 1e4)
                dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
                inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
                return inv, 1.0
            _R.ROPE_INIT_FUNCTIONS["default"] = _d
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=trc)
    lk = {}
    if kind == "onevision15":
        from transformers import AutoConfig, AutoModel as M
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        tc = getattr(cfg, "text_config", None)
        if tc is not None and not hasattr(tc, "pad_token_id"):
            tc.pad_token_id = None
        lk = {"config": cfg, "attn_implementation": "sdpa", "trust_remote_code": True}
    elif kind == "gemma":
        from transformers import Gemma4UnifiedForConditionalGeneration as M
    elif kind == "internvl":
        from transformers import InternVLForConditionalGeneration as M
    model = M.from_pretrained(model_id, dtype=torch.bfloat16, **lk).to(DEV).eval()
    img_id = getattr(model.config, "image_token_index", None) or getattr(model.config, "image_token_id", None)
    print(f"[diag] kind={kind} img_id={img_id} model_type={model.config.model_type if hasattr(model.config,'model_type') else '?'}", flush=True)

    ex = load_one_image()
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(ex["image"]["bytes"])).convert("RGB")
    q = "What is the main object in this image?"
    if kind == "gemma":
        msg = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]}]
        b = proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        b = {k: (v.to(DEV) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
    else:
        msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        pr = proc.apply_chat_template(msg, add_generation_prompt=True)
        b = proc(images=[img], text=[pr], return_tensors="pt").to(DEV)
    ids = b["input_ids"][0]
    vm = (ids == img_id)
    print(f"[diag] seq_len={len(ids)} n_image_tokens(vm.sum)={int(vm.sum())} "
          f"unique_ids_near_imgid={sorted(set(ids[ max(0,int(vm.nonzero()[0,0])-1) : int(vm.nonzero()[-1,0])+2 ].tolist()))[:6] if vm.any() else 'NONE'}", flush=True)
    keys = list(b.keys())
    print(f"[diag] batch keys={keys}", flush=True)
    import inspect
    params = inspect.signature(model.forward).parameters
    if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
        b = {k: v for k, v in b.items() if k in set(params)}
    with torch.no_grad():
        out = model(**b, output_hidden_states=True)
    hs = out.hidden_states
    h0 = hs[0][0]
    if vm.any():
        img_norm = h0[vm].float().norm(dim=-1).mean().item()
        txt_norm = h0[~vm].float().norm(dim=-1).mean().item()
        # cosine spread among image tokens (are they diverse or all identical?)
        v = h0[vm].float()
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        offdiag = (v @ v.T)
        n = offdiag.shape[0]
        mean_cos = (offdiag.sum() - n) / max(n * (n - 1), 1)
        print(f"[diag] L0 img-token norm={img_norm:.2f} txt-token norm={txt_norm:.2f} "
              f"img-token mean pairwise cos={mean_cos.item():.3f} (1.0=all identical)", flush=True)
    print("[diag] DONE", flush=True)


if __name__ == "__main__":
    main()
