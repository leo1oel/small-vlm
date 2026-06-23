"""Functional fusion depth (cost/retained/FDI) + rho for ENCODER-BASED VLMs
(LLaVA-1.5, Qwen2.5-VL), same metric as the encoder-free probes. Tests whether
having a vision encoder lets the model fuse EARLIER (lower FDI) than native
models, or whether mid-stack fusion (FDI~0.5) is an LLM-depth property.

cost(d): block text-query->image-key in LLM decoder layers [0..d), real image
vs swap-image null. rho(l): answer-position hidden-state image-content
sensitivity. Letter scoring on VMCBench dev.

Runs in MAIN .venv, eager attention. cost on first N_CAUSAL, rho on all n.

Usage: python devtools/encoder_vlm_fusion.py <model_id> <kind> <out.jsonl> [n] [n_causal]
  kind in {llava, qwen}
"""

import json
import sys
from pathlib import Path

import torch

DEV = "cuda"
POST = "Answer with the option's letter from the given choices directly.\n"


def load_vmcbench():
    import datasets
    return datasets.load_dataset("suyc21/VMCBench", split="dev")


def doc_to_prompt(doc):
    op = "Options:\n" + "".join(f"{k}. {doc[k]}\n" for k in "ABCD")
    return f"Question: {doc['question']}\n{op}{POST}"


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


class EncProbe:
    def __init__(self, model_id, kind):
        from transformers import AutoProcessor
        self.kind = kind
        self.proc = AutoProcessor.from_pretrained(model_id)
        if kind == "llava":
            from transformers import LlavaForConditionalGeneration as M
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration as M
        self.model = M.from_pretrained(model_id, dtype=torch.bfloat16,
                                       attn_implementation="eager").to(DEV).eval()
        self.img_id = (getattr(self.model.config, "image_token_index", None) or
                       getattr(self.model.config, "image_token_id", None))
        self.tok = self.proc.tokenizer
        self.letter_ids = {c: self.tok(c, add_special_tokens=False).input_ids[0] for c in "ABCD"}
        self.layers = find_layers(self.model)
        self.N = len(self.layers)
        self.mode, self.depth, self.capture = "none", 0, False
        self.vm = None
        self.cap = {}
        self._mc = {}
        for i, lyr in enumerate(self.layers):
            lyr.register_forward_pre_hook(self._pre(i), with_kwargs=True)
            lyr.register_forward_hook(self._post(i + 1))

    def _isolated(self, mask):
        key = id(mask)
        if key not in self._mc:
            m = mask.clone()
            tp = (~self.vm).nonzero().squeeze(-1)
            vp = self.vm.nonzero().squeeze(-1)
            neg = False if m.dtype == torch.bool else torch.finfo(m.dtype).min
            m[0, :, tp[:, None], vp[None, :]] = neg
            self._mc[key] = m
        return self._mc[key]

    def _pre(self, i):
        def f(_m, args, kwargs):
            if self.mode == "cumulative" and i < self.depth and self.vm is not None:
                am = kwargs.get("attention_mask")
                if am is not None and am.dim() == 4:
                    kwargs = dict(kwargs)
                    kwargs["attention_mask"] = self._isolated(am)
                    return args, kwargs
            return None
        return f

    def _post(self, k):
        def f(_m, _a, out):
            if self.capture:
                h = (out[0] if isinstance(out, tuple) else out)[0]
                self.cap[k] = h[-1].float()
        return f

    def _build(self, image, question):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        b = self.proc(images=[image], text=[prompt], return_tensors="pt").to(DEV)
        vm = (b["input_ids"][0] == self.img_id)
        return b, vm

    def _build_text(self, question):
        msgs = [{"role": "user", "content": [{"type": "text", "text": question}]}]
        prompt = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        return self.proc(text=[prompt], return_tensors="pt").to(DEV)

    @torch.no_grad()
    def _logits(self, b):
        return self.model(**b).logits[0, -1].float()

    def _score(self, lg, gt):
        s = {c: lg[self.letter_ids[c]].item() for c in "ABCD"}
        return dict(pred=max(s, key=s.get), nll=-lg.log_softmax(-1)[self.letter_ids[gt]].item())

    @torch.no_grad()
    def _cap_all(self, b, vm):
        self.mode, self.capture, self.vm, self.cap = "none", True, vm, {}
        self._mc.clear()
        self.model(**b)
        self.capture = False
        return dict(self.cap)

    @torch.no_grad()
    def run(self, doc, donor, do_causal):
        gt = str(doc["answer"]).strip()
        if gt not in "ABCD":
            return dict(skip="bad_gt")
        q = doc_to_prompt(doc)
        bI, vmI = self._build(doc["image"].convert("RGB"), q)
        bIp, vmIp = self._build(donor["image"].convert("RGB"), q)
        bt = self._build_text(q)
        rec = dict(gt=gt, category=doc["category"], n_vis=int(vmI.sum()),
                   seq_len=int(bI["input_ids"].shape[1]))
        # rho
        cI = self._cap_all(bI, vmI)
        cIp = self._cap_all(bIp, vmIp)
        self.mode, self.capture, self.vm, self.cap = "none", True, None, {}
        self._mc.clear()
        self.model(**bt)
        c0 = dict(self.cap)
        self.capture = False
        dsw, dno = [], []
        for k in range(1, self.N + 1):
            if k in cI and k in cIp and k in c0:
                dsw.append((cI[k] - cIp[k]).norm().item())
                dno.append((cI[k] - c0[k]).norm().item())
            else:
                dsw.append(float("nan")); dno.append(float("nan"))
        rec["rho"] = dict(dswap=dsw, dnoimg=dno)
        # intact / swap
        self.mode, self.vm = "none", vmI
        self._mc.clear()
        rec["intact"] = self._score(self._logits(bI), gt)
        self.vm = vmIp
        self._mc.clear()
        rec["swap"] = self._score(self._logits(bIp), gt)
        if not do_causal:
            return rec
        cost, cost_null = [], []
        self.mode = "cumulative"
        for d in range(1, self.N + 1):
            self.depth = d
            self.vm = vmI
            self._mc.clear()
            cost.append(self._score(self._logits(bI), gt))
            self.vm = vmIp
            self._mc.clear()
            cost_null.append(self._score(self._logits(bIp), gt))
        rec["cost"], rec["cost_null"] = cost, cost_null
        self.mode = "none"
        return rec


@torch.no_grad()
def main():
    model_id, kind, out_path = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    n_causal = int(sys.argv[5]) if len(sys.argv) > 5 else 150
    ds = load_vmcbench()
    n = min(n, len(ds))
    # STRATIFIED causal subset: stride across the (category-ordered) dataset so
    # the cost/FDI sweep spans all categories, not just the hard first few.
    stride = max(1, n // max(n_causal, 1))
    causal_set = set(range(0, n, stride))
    done = sum(1 for _ in open(out_path)) if out_path.exists() else 0
    print(f"[encfuse] {model_id} kind={kind} n={n} n_causal={len(causal_set)} "
          f"(strided/{stride}) resume={done}", flush=True)
    pr = EncProbe(model_id, kind)
    print(f"[encfuse] N_layers={pr.N} img_id={pr.img_id}", flush=True)
    f = open(out_path, "a")
    for i in range(done, n):
        doc, donor = ds[i], ds[(i + 37) % n]
        try:
            rec = pr.run(doc, donor, do_causal=(i in causal_set))
            rec["i"] = i
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rec = dict(i=i, skip="oom")
        except Exception as e:  # noqa: BLE001
            rec = dict(i=i, skip=f"{type(e).__name__}: {e}")
        f.write(json.dumps(rec) + "\n")
        if (i + 1) % 25 == 0:
            f.flush(); torch.cuda.empty_cache()
            print(f"[encfuse] {i + 1}/{n}", flush=True)
    f.close()
    print("[encfuse] done", flush=True)


if __name__ == "__main__":
    main()
