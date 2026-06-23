"""Extract per-layer mean-pooled IMAGE-TOKEN representations for CKA analysis.
Tests where (at what decoder depth) an encoder-free VLM's image-token stream
becomes aligned with a real frozen visual encoder (DINOv2 / CLIP / SigLIP) —
i.e. where the early layers finish doing the "vision encoder's job".

For VLM kinds (llava/qwen/gemma): output_hidden_states, mean-pool over image
tokens at each layer -> (L+1, N, H). For reference encoders (dino/clip/siglip):
mean-pool patch tokens of the last hidden state -> (1, N, H).

Uses a FIXED stratified image subset (same indices across all models) so CKA
pairs row-for-row. Saves neo_analysis/cka_<tag>.npz {feats, idx}.

Usage: python devtools/cka_extract.py <kind> <model_id> <tag> [N]
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "neo_analysis"))
ROOT = Path(__file__).resolve().parents[1] / "neo_analysis"
DEV = "cuda"
POST = "Answer with the option's letter from the given choices directly.\n"


def vmc():
    import datasets
    return datasets.load_dataset("suyc21/VMCBench", split="dev")


def prompt(doc):
    op = "Options:\n" + "".join(f"{k}. {doc[k]}\n" for k in "ABCD")
    return f"Question: {doc['question']}\n{op}{POST}"


def main():
    kind, model_id, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    ds = vmc()
    stride = max(1, len(ds) // N)
    idx = list(range(0, len(ds), stride))[:N]
    from transformers import AutoProcessor
    feats = []  # list over images of (L+1, H)
    if kind in ("dino", "clip", "siglip"):
        from transformers import AutoModel
        proc = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16).to(DEV).eval()
        for i in idx:
            img = ds[i]["image"].convert("RGB")
            inp = proc(images=img, return_tensors="pt").to(DEV)
            pv = inp["pixel_values"].to(torch.bfloat16)
            with torch.no_grad():
                vt = getattr(model, "vision_model", model)  # CLIP/SigLIP tower or DINO itself
                hs = vt(pixel_values=pv).last_hidden_state[0]  # (P,H)
                out = hs.float().mean(0, keepdim=True)  # (1,H)
            feats.append(out.cpu().numpy())
        arr = np.stack(feats, axis=1)  # (1,N,H)
    else:
        if kind == "llava":
            from transformers import LlavaForConditionalGeneration as M
        elif kind == "qwen":
            from transformers import Qwen2_5_VLForConditionalGeneration as M
        elif kind == "gemma":
            from transformers import Gemma4UnifiedForConditionalGeneration as M
        elif kind == "internvl":
            from transformers import InternVLForConditionalGeneration as M
        elif kind == "janus":
            from transformers import JanusForConditionalGeneration as M
        elif kind == "gemma4moe":
            from transformers import Gemma4ForConditionalGeneration as M
        else:
            raise ValueError(kind)
        proc = AutoProcessor.from_pretrained(model_id)
        import os as _os
        # big MoE (30B/26B) won't fit one a40 -> device_map=auto shards it.
        big = _os.environ.get("DEVMAP")
        if _os.environ.get("RANDINIT"):
            # causal control: identical architecture, RANDOM (untrained) weights.
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_id)
            model = M(cfg).to(torch.bfloat16).to(DEV).eval()
            print(f"[cka] RANDOM-INIT {kind} ({sum(p.numel() for p in model.parameters())/1e9:.2f}B)", flush=True)
        elif big:
            model = M.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto").eval()
        else:
            model = M.from_pretrained(model_id, dtype=torch.bfloat16).to(DEV).eval()
        img_id = (getattr(model.config, "image_token_index", None) or
                  getattr(model.config, "image_token_id", None))
        for i in idx:
            img = ds[i]["image"].convert("RGB")
            q = prompt(ds[i])
            if kind in ("gemma", "gemma4moe"):
                msg = [{"role": "user", "content": [{"type": "image", "image": img},
                        {"type": "text", "text": q}]}]
                b = proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=True,
                                             return_dict=True, return_tensors="pt")
                b = {k: (v.to(DEV) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            else:
                msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
                pr = proc.apply_chat_template(msg, add_generation_prompt=True)
                b = proc(images=[img], text=[pr], return_tensors="pt").to(DEV)
            vm = (b["input_ids"][0] == img_id)
            with torch.no_grad():
                out = model(**b, output_hidden_states=True)
            hs = out.hidden_states  # (L+1) of (1,S,H)
            pooled = torch.stack([h[0][vm].float().mean(0) for h in hs])  # (L+1,H)
            feats.append(pooled.cpu().numpy())
        arr = np.stack(feats, axis=1)  # (L+1,N,H)
    np.savez(ROOT / f"cka_{tag}.npz", feats=arr.astype(np.float16), idx=np.array(idx))
    print(f"[cka] {tag}: saved {arr.shape} (L+1,N,H)", flush=True)


if __name__ == "__main__":
    main()
