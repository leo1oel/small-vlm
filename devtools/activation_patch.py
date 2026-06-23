"""ACTIVATION-PATCH probe: orthogonal causal triangulation of attention-knockout.

The fusion-window probe localizes fusion by blocking text->image ATTENTION EDGES.
This probe instead intervenes on the IMAGE-TOKEN RESIDUAL STREAM (the value pathway),
so it is mechanistically independent of attention: self-repair / Hydra and attention-
sink artifacts cannot reproduce its signal. If it reproduces the knockout fusion depth,
the two most serious objections to the knockout result fall at once.

  denoise@d : run the DONOR image, but overwrite image-position layer OUTPUTS in
              layers [0..d) with the CLEAN image's cached outputs. retained_denoise(d)/R0
              rises from ~0 (d=0 == swap) to ~1 (d=N == intact); the rise marks where
              injecting the clean image early becomes SUFFICIENT.
  meanabl@d : run the CLEAN image, replace image-position outputs in [0..d) with their
              mean over image tokens (removes image-specific info, keeps norm/scale).
              retained_meanabl(d)/R0 falls; the fall marks where the image-specific
              signal is NECESSARY. (knockout robustness vs the -inf OOD shock.)

Same VMCBench dev, letter scoring, donor (i+37)%n pairing, stratified causal subset,
resume-safe jsonl, find_layers, eager attention as freeze_probe.py / fusion_window.py.

Usage: python devtools/activation_patch.py <model_id> <kind> <out.jsonl> [n] [n_causal] [mode]
  kind in {llava, qwen, gemma}; mode in {denoise, meanabl, both} (default both)
"""

import json
import os
import sys
from pathlib import Path

import torch

DEV = "cuda"
POST = "Answer with the option's letter from the given choices directly.\n"
# fixed square (multiple of 14*2 merge) => deterministic image-token count for Qwen.
# 448 => 256 tokens (enough for coarse VQA); raise via env for dense-OCR benches (sufmeanabl is
# self-aligned so a higher count is fine; only `denoise` needs cross-run alignment). 896 => 1024 tok.
QWEN_SIDE = int(os.environ.get("QWEN_SIDE", "448"))


def load_vmcbench():
    import datasets
    return datasets.load_dataset("suyc21/VMCBench", split="dev")


def load_dataset_bench(bench):
    import datasets
    if bench == "mmstar":
        return datasets.load_dataset("Lin-Chen/MMStar")["val"]  # 1500 MCQ, options embedded in `question`
    if bench == "worldbench":
        # 2000 MCQ, 7 visually-diverse domains; option_a..d + bare-letter answer + `domain` label.
        # Dataset is stored ORDERED BY DOMAIN -> shuffle(seed=0) so any contiguous slice / strided
        # causal subset is domain-representative (reproducible).
        ds = datasets.load_dataset("zlab-princeton/WorldBench", split="train").shuffle(seed=0)
        return ds.cast_column("image", datasets.Image(decode=True))  # stored decode=false
    return load_vmcbench()


def doc_to_prompt(doc):
    if "option_a" in doc:  # WorldBench: option_a..option_d fields
        op = "Options:\n" + "".join(f"{k}. {doc['option_' + k.lower()]}\n" for k in "ABCD")
        return f"Question: {doc['question']}\n{op}{POST}"
    if "A" in doc:  # VMCBench: separate A/B/C/D option fields
        op = "Options:\n" + "".join(f"{k}. {doc[k]}\n" for k in "ABCD")
        return f"Question: {doc['question']}\n{op}{POST}"
    return f"{doc['question']}\n{POST}"  # MMStar: options already embedded in `question`


def find_layers(model):
    for path in ("model.language_model.layers", "language_model.model.layers",
                 "model.layers", "language_model.layers"):
        obj = model
        ok = True
        for p in path.split("."):
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False
                break
        if ok and hasattr(obj, "__len__"):
            return obj
    raise RuntimeError("could not locate decoder layers")


class PatchProbe:
    def __init__(self, model_id, kind):
        from transformers import AutoProcessor
        self.kind = kind
        self.trc = kind in ("onevision15", "onevision2")  # self-contained trust_remote_code models
        # Qwen is native-resolution => image-token count varies per image, which would make
        # the (i+37) donor mismatch the clean image's image-token positions and force a skip.
        # A pixel-budget kwarg alone does NOT fix this (smart_resize rounds H and W to multiples
        # of 28 independently, so the token count still drifts with aspect ratio). The real fix is
        # to pre-resize every image to a FIXED 28-multiple square in _build (QWEN_SIDE below) so the
        # patch grid is identical for every image; min/max_pixels is pinned to match as a guard.
        if kind == "qwen":
            px = QWEN_SIDE * QWEN_SIDE
            self.proc = AutoProcessor.from_pretrained(model_id, min_pixels=px, max_pixels=px)
        else:
            self.proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=self.trc)
        if kind == "llava":
            from transformers import LlavaForConditionalGeneration as M
        elif kind == "qwen":
            from transformers import Qwen2_5_VLForConditionalGeneration as M
        elif kind == "gemma":
            from transformers import Gemma4UnifiedForConditionalGeneration as M
        elif kind == "internvl":   # InternVL3.5-{8B-HF dense, 30B-A3B-HF MoE}; native-res (variable n_vis)
            from transformers import InternVLForConditionalGeneration as M
        elif kind == "janus":      # Janus-Pro-7B (probe understanding path)
            from transformers import JanusForConditionalGeneration as M
        elif kind in ("onevision15", "onevision2"):  # LLaVA-OneVision-1.5 / -2 (self-contained TRC)
            from transformers import AutoModel as M  # repo auto_map maps AutoModel -> its *ForConditionalGeneration
        elif kind == "gemma4moe":  # Gemma-4-26B-A4B (sparse MoE, encoder-free); same image handling as gemma
            from transformers import Gemma4ForConditionalGeneration as M
        else:
            raise ValueError(kind)
        load_kwargs = {}
        if self.trc:  # onevision TRC: its custom text_config lacks pad_token_id (transformers 5.10 incompat)
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            tc = getattr(cfg, "text_config", None)
            if tc is not None and not hasattr(tc, "pad_token_id"):
                tc.pad_token_id = None
            load_kwargs["config"] = cfg
        self.model = M.from_pretrained(model_id, dtype=torch.bfloat16,
                                       attn_implementation="eager",
                                       trust_remote_code=self.trc, **load_kwargs).to(DEV).eval()
        self.img_id = (getattr(self.model.config, "image_token_index", None) or
                       getattr(self.model.config, "image_token_id", None))
        self.tok = self.proc.tokenizer
        self.letter_ids = {c: self.tok(c, add_special_tokens=False).input_ids[0] for c in "ABCD"}
        self.layers = find_layers(self.model)
        self.N = len(self.layers)
        self.mode = "none"   # none | capture | denoise | meanabl
        self.depth = 0       # patch layers [0..depth)
        self.pos = None      # bool (S,) image positions of the CURRENT run
        self.cache = {}      # layer i -> clean image-position outputs [n_vis, D]
        for i, lyr in enumerate(self.layers):
            lyr.register_forward_hook(self._post(i), with_kwargs=True)

    def _post(self, i):
        def f(_m, _args, _kwargs, out):
            if self.mode == "none":
                return None
            h = out[0] if isinstance(out, tuple) else out
            if self.mode == "capture":
                self.cache[i] = h[0, self.pos].detach().clone()
            elif self.mode == "denoise" and i < self.depth and i in self.cache:
                h[0, self.pos] = self.cache[i].to(h.dtype)
            elif self.mode == "meanabl" and i < self.depth:
                h[0, self.pos] = h[0, self.pos].mean(0, keepdim=True)
            elif self.mode == "sufmeanabl" and i >= self.depth:
                h[0, self.pos] = h[0, self.pos].mean(0, keepdim=True)
            return None
        return f

    def _build(self, image, question):
        if self.kind == "qwen":
            image = image.resize((QWEN_SIDE, QWEN_SIDE))   # fixed grid => aligned image-token positions
        if self.kind in ("gemma", "gemma4moe"):
            msg = [{"role": "user", "content": [{"type": "image", "image": image},
                    {"type": "text", "text": question}]}]
            b = self.proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=True,
                                              return_dict=True, return_tensors="pt")
            b = {k: (v.to(DEV) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        else:
            msg = [{"role": "user", "content": [{"type": "image"},
                    {"type": "text", "text": question}]}]
            prompt = self.proc.apply_chat_template(msg, add_generation_prompt=True)
            b = self.proc(images=[image], text=[prompt], return_tensors="pt").to(DEV)
        vm = (b["input_ids"][0] == self.img_id)
        return b, vm

    @torch.no_grad()
    def _logits(self, b):
        return self.model(**b).logits[0, -1].float()

    def _score(self, lg, gt):
        s = {c: lg[self.letter_ids[c]].item() for c in "ABCD"}
        return dict(pred=max(s, key=s.get), nll=-lg.log_softmax(-1)[self.letter_ids[gt]].item())

    @torch.no_grad()
    def run(self, doc, donor, do_causal, depths, modes):
        gt = str(doc["answer"]).strip()
        if gt not in "ABCD":
            return dict(skip="bad_gt")
        q = doc_to_prompt(doc)
        bI, vmI = self._build(doc["image"].convert("RGB"), q)
        bIp, vmIp = self._build(donor["image"].convert("RGB"), q)
        rec = dict(gt=gt, category=doc.get("category"), domain=doc.get("domain"),
                   n_vis=int(vmI.sum()), seq_len=int(bI["input_ids"].shape[1]))
        self.mode = "none"
        rec["intact"] = self._score(self._logits(bI), gt)
        rec["swap"] = self._score(self._logits(bIp), gt)
        if not do_causal:
            return rec
        rec["depths"] = depths
        # denoise injects CLEAN image residuals into the DONOR run, so it needs the two runs'
        # image-token positions to line up (== count, since the image block is contiguous). For
        # native-resolution models (NEO/SAIL/InternVL/OV) the donor often differs => skip denoise
        # only. meanabl is self-aligned (each run ablates its OWN image tokens) => always runs.
        aligned = int(vmI.sum()) == int(vmIp.sum())
        if "denoise" in modes and aligned:
            self.mode, self.pos, self.cache = "capture", vmI, {}
            _ = self._logits(bI)                      # cache clean image-position outputs
            out = []
            for d in depths:
                self.mode, self.depth, self.pos = "denoise", d, vmIp
                out.append(self._score(self._logits(bIp), gt))
            rec["denoise"] = out
        elif "denoise" in modes:
            rec["denoise_skip"] = "nvis_mismatch"
        if "meanabl" in modes:
            out, out_null = [], []
            for d in depths:
                self.mode, self.depth, self.pos = "meanabl", d, vmI
                out.append(self._score(self._logits(bI), gt))
                self.pos = vmIp
                out_null.append(self._score(self._logits(bIp), gt))
            rec["meanabl"], rec["meanabl_null"] = out, out_null
        if "sufmeanabl" in modes:
            # PRIMARY discriminative metric: flatten image residuals in [d..N) (suffix). The image
            # content is intact for [0..d) and destroyed after. retained rises from 0 (d=0, all
            # flattened) to 1 (d=N, none flattened); the RISE marks the depth up to which content
            # must be preserved = the read/fusion onset. Self-aligned (no donor matching needed).
            out, out_null = [], []
            for d in depths:
                self.mode, self.depth, self.pos = "sufmeanabl", d, vmI
                out.append(self._score(self._logits(bI), gt))
                self.pos = vmIp
                out_null.append(self._score(self._logits(bIp), gt))
            rec["sufmeanabl"], rec["sufmeanabl_null"] = out, out_null
        self.mode, self.cache = "none", {}
        return rec


@torch.no_grad()
def main():
    model_id, kind, out_path = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    n_causal = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    mode = sys.argv[6] if len(sys.argv) > 6 else "sufmeanabl"
    modes = {"both": ("denoise", "meanabl"), "all": ("denoise", "meanabl", "sufmeanabl")}.get(mode, (mode,))
    bench = sys.argv[7] if len(sys.argv) > 7 else "vmcbench"
    ds = load_dataset_bench(bench)
    n = min(n, len(ds))
    stride = max(1, n // max(n_causal, 1))
    causal_set = set(range(0, n, stride))
    done = sum(1 for _ in open(out_path)) if out_path.exists() else 0
    pr = PatchProbe(model_id, kind)
    depths = list(range(0, pr.N + 1))                 # [0..N]
    print(f"[patch] {model_id} kind={kind} N={pr.N} modes={modes} n={n} "
          f"nc={len(causal_set)} resume={done}", flush=True)
    print(f"[patch] img_id={pr.img_id} letter_ids={pr.letter_ids}", flush=True)
    f = open(out_path, "a")
    for i in range(done, n):
        doc, donor = ds[i], ds[(i + 37) % n]
        try:
            rec = pr.run(doc, donor, do_causal=(i in causal_set), depths=depths, modes=modes)
            rec["i"] = i
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rec = dict(i=i, skip="oom")
        except Exception as e:  # noqa: BLE001
            rec = dict(i=i, skip=f"{type(e).__name__}: {e}")
        f.write(json.dumps(rec) + "\n")
        if (i + 1) % 25 == 0:
            f.flush(); torch.cuda.empty_cache()
            print(f"[patch] {i + 1}/{n}", flush=True)
    f.close()
    print("[patch] done", flush=True)


if __name__ == "__main__":
    main()
