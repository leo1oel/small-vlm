"""ImageNet linear probing (P2, Claim B — gold-standard functional test).

Question: does a native VLM's internal image representation reach ImageNet top-1
accuracy comparable to a real frozen encoder's (and to an encoder-VLM's internal
layers), rather than to raw pixels? If a native model's peak internal layer probes
near the encoder ceiling and far above the layer-0/pixel floor, its internal params
function as a trained vision encoder.

Two subcommands:
  extract  <kind> <model_id> <tag> [fracs] [N]
      Mean-pool the image-token rep at each requested layer-fraction over the
      ImageNet-1k val set; save imagenet_<tag>.npz {reps (Lf,N,H), labels (N,),
      layers (Lf,), fracs (Lf,)}. Native models (sail/neo/mono) use their own
      analysis-dir extractor (mirror this format).
  probe    <tag1> [tag2 ...]
      For each tag & layer, train a multinomial logistic head on a FIXED per-class
      stratified split (same indices for every source) and report test top-1.

Reps are extracted into neo_analysis/ (alongside cka_*). CPU probe.

Usage:
  python devtools/imagenet_probe.py extract clip openai/clip-vit-large-patch14-336 clip
  python devtools/imagenet_probe.py probe clip dino siglip llava gemma sail mono
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1] / "neo_analysis"
DEV = "cuda"
DEFAULT_FRACS = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
POST = "Answer with the option's letter from the given choices directly.\n"


def load_imagenet(N=None, seed=0):
    import datasets
    ds = datasets.load_dataset("mrm8488/ImageNet1K-val", split="train")
    if N is not None and N < len(ds):
        # stratified per-class subset: take first k per class deterministically
        per = {}
        keep = []
        cap = max(1, N // 1000)
        labels = ds["label"]
        for i, y in enumerate(labels):
            if per.get(y, 0) < cap:
                per[y] = per.get(y, 0) + 1
                keep.append(i)
        ds = ds.select(keep)
    return ds


def frac_to_layers(L, fracs):
    return sorted(set(int(round(f * (L - 1))) for f in fracs))


# ---------------- extraction (main-venv kinds) ----------------
def pool_tokens(V):
    # V: (n_img_tokens, H). Default mean-pool. POOL=l2mean L2-normalizes each token first, which
    # removes a shared high-norm DC/bias direction (e.g. OneVision-1.5's pixel-shuffle merger makes
    # all image tokens collinear at high norm; plain mean then averages the discriminative residuals
    # to ~zero -> flat probe). L2-norm-then-mean keeps the angular/per-token signal.
    import os as _os
    if _os.environ.get("POOL") == "l2mean":
        V = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return V.mean(0)


def extract_main(kind, model_id, tag, fracs, N):
    from transformers import AutoProcessor
    ds = load_imagenet(N)
    labels = np.array(ds["label"], dtype=np.int64)
    reps, layers = [], None
    if kind in ("dino", "clip", "siglip"):
        from transformers import AutoModel
        proc = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16).to(DEV).eval()
        layers = [0]
        for ex in ds:
            img = ex["image"].convert("RGB")
            inp = proc(images=img, return_tensors="pt").to(DEV)
            pv = inp["pixel_values"].to(torch.bfloat16)
            with torch.no_grad():
                vt = getattr(model, "vision_model", model)
                hs = vt(pixel_values=pv).last_hidden_state[0]
                reps.append(hs.float().mean(0, keepdim=True).cpu().numpy())  # (1,H)
        arr = np.stack(reps, axis=1)  # (1,N,H)
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
        elif kind == "onevision15":  # LLaVA-OneVision-1.5 (self-contained TRC; repo auto_map -> *ForCG)
            from transformers import AutoModel as M
        else:
            raise ValueError(kind)
        trc = kind == "onevision15"
        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=trc)
        load_kwargs = {}
        if trc:
            # transformers 5.10 dropped the "default" key from ROPE_INIT_FUNCTIONS (the OneVision-1.5
            # custom code still indexes it for rope_scaling=None). Re-register the standard computation.
            import transformers.modeling_rope_utils as _R
            if "default" not in _R.ROPE_INIT_FUNCTIONS:
                def _default_rope(config, device=None, seq_len=None, **kw):
                    base = getattr(config, "rope_theta", 10000.0)
                    dim = getattr(config, "head_dim", None) or (
                        config.hidden_size // config.num_attention_heads)
                    inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
                    return inv, 1.0
                _R.ROPE_INIT_FUNCTIONS["default"] = _default_rope
        if trc:  # custom text_config lacks pad_token_id (transformers 5.10 incompat). Use sdpa, NOT
            # eager: the custom eager attention path (self.num_heads) is broken; we only need
            # output_hidden_states here (no attention hooks), so sdpa is fine and avoids that path.
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            tc = getattr(cfg, "text_config", None)
            if tc is not None and not hasattr(tc, "pad_token_id"):
                tc.pad_token_id = None
            load_kwargs = {"config": cfg, "attn_implementation": "sdpa", "trust_remote_code": True}
        if os.environ.get("DEVMAP"):  # big MoE (30B/26B) -> shard across GPU
            model = M.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto", **load_kwargs).eval()
        else:
            model = M.from_pretrained(model_id, dtype=torch.bfloat16, **load_kwargs).to(DEV).eval()
        img_id = (getattr(model.config, "image_token_index", None) or
                  getattr(model.config, "image_token_id", None))
        # TRC processors (OneVision-1.5) emit extra keys (mm_token_type_ids) the custom forward
        # rejects; restrict the batch to the forward signature when there's no **kwargs catch-all.
        fwd_keys = None
        if trc:
            import inspect
            params = inspect.signature(model.forward).parameters
            if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
                fwd_keys = set(params)
        q = "What is the main object in this image?"

        def ckpt_save(done):  # incremental save so a SLURM timeout still yields a usable npz
            if not done:
                return
            arr_p = np.stack(reps, axis=1)  # (Lf,done,H)
            np.savez(ROOT / f"imagenet_{tag}.npz", reps=arr_p.astype(np.float16),
                     labels=labels[:done], layers=np.array(layers), fracs=np.array(fracs))
            print(f"[in-extract] {tag}: checkpoint saved {done}/{len(ds)} reps {arr_p.shape}", flush=True)

        for n, ex in enumerate(ds):
            img = ex["image"].convert("RGB")
            if kind == "qwen":
                img = img.resize((448, 448))
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
            if fwd_keys is not None:
                b = {k: v for k, v in b.items() if k in fwd_keys}
            with torch.no_grad():
                out = model(**b, output_hidden_states=True)
            hs = out.hidden_states
            if layers is None:
                layers = frac_to_layers(len(hs), fracs)
            pooled = np.stack([pool_tokens(hs[L][0][vm].float()).cpu().numpy() for L in layers])  # (Lf,H)
            reps.append(pooled)
            del out
            torch.cuda.empty_cache()
            if (n + 1) % 1000 == 0:
                print(f"[in-extract] {tag}: {n + 1}/{len(ds)}", flush=True)
            if (n + 1) % 5000 == 0:
                ckpt_save(n + 1)
        arr = np.stack(reps, axis=1)  # (Lf,N,H)
    np.savez(ROOT / f"imagenet_{tag}.npz", reps=arr.astype(np.float16), labels=labels,
             layers=np.array(layers), fracs=np.array(fracs if kind not in ("dino", "clip", "siglip") else [1.0]))
    print(f"[in-extract] {tag}: reps {arr.shape} labels {labels.shape} layers {layers}", flush=True)


# ---------------- probe (model-agnostic) ----------------
def fixed_split(labels, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for y in np.unique(labels):
        idx = np.where(labels == y)[0]
        rng.shuffle(idx)
        k = max(1, int(round(len(idx) * test_frac)))
        te.extend(idx[:k]); tr.extend(idx[k:])
    return np.array(sorted(tr)), np.array(sorted(te))


def probe(tags):
    import json
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    results = {}
    for tag in tags:
        f = ROOT / f"imagenet_{tag}.npz"
        if not f.exists():
            print(f"[probe] SKIP {tag} (no npz)", flush=True); continue
        d = np.load(f)
        reps, labels, layers = d["reps"].astype(np.float32), d["labels"], d["layers"]
        tr, te = fixed_split(labels)
        per_layer = {}
        for li in range(reps.shape[0]):
            X = reps[li]
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=300, C=1.0, n_jobs=-1)  # multinomial by default in sklearn>=1.7
            clf.fit(sc.transform(X[tr]), labels[tr])
            acc = float((clf.predict(sc.transform(X[te])) == labels[te]).mean())
            per_layer[int(layers[li])] = round(acc, 4)
            print(f"[probe] {tag:10s} L{int(layers[li]):3d} top1={acc:.4f}", flush=True)
        best = max(per_layer.items(), key=lambda kv: kv[1])
        results[tag] = {"per_layer": per_layer, "peak_layer": best[0], "peak_top1": best[1],
                        "n_test": int(len(te)), "n_classes": int(len(np.unique(labels)))}
    (ROOT / "imagenet_probe_results.json").write_text(json.dumps(results, indent=2))
    print(f"[probe] saved imagenet_probe_results.json", flush=True)


def main():
    mode = sys.argv[1]
    if mode == "extract":
        kind, model_id, tag = sys.argv[2], sys.argv[3], sys.argv[4]
        fracs = [float(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 and sys.argv[5] else DEFAULT_FRACS
        N = int(sys.argv[6]) if len(sys.argv) > 6 else None
        extract_main(kind, model_id, tag, fracs, N)
    elif mode == "probe":
        probe(sys.argv[2:])
    else:
        raise SystemExit("mode must be extract|probe")


if __name__ == "__main__":
    main()
