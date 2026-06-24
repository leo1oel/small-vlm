"""FUSION WINDOW probe: prefix- AND suffix-blocking sweeps for HF VLMs.

Motivation (user critique): the prefix sweep cost(d) = block text-q->image-k in
layers [0..d) measures where fusion becomes IRRECOVERABLE (the latest the model
can still extract the image if the early read is destroyed) — NOT where fusion
normally starts. The complementary SUFFIX sweep blocks layers [d..N):

  suf(d)  = acc(real img, block text->image in [d..N))
  suf_null(d) = same with swap image
  retained_suf(d)/R0 = fraction of usable image signal ALREADY transferred into
  the text stream by depth d (it can still propagate text->text afterwards).

The rise of retained_suf marks the true ONSET of functional fusion; the fall of
the prefix curve marks its COMPLETION/irrecoverability. Together: the window.

Kinds: llava, qwen, gemma, onevision, llavanext. Sweeps selectable so models
that already have prefix data only re-run suffix. Letter scoring, stratified
causal subset, resume-safe jsonl (same conventions as encoder_vlm_fusion.py).

Usage: python devtools/fusion_window.py <model_id> <kind> <out.jsonl> [n] [n_causal] [sweeps]
  sweeps in {suffix, prefix, both}; default suffix
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
    for path in (
        "model.language_model.layers",
        "language_model.model.layers",
        "model.layers",
        "language_model.layers",
    ):
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


class WinProbe:
    def __init__(self, model_id, kind):
        from transformers import AutoProcessor

        self.kind = kind
        self.proc = AutoProcessor.from_pretrained(model_id)
        if kind == "llava":
            from transformers import LlavaForConditionalGeneration as M
        elif kind == "qwen":
            from transformers import Qwen2_5_VLForConditionalGeneration as M
        elif kind == "gemma":
            from transformers import Gemma4UnifiedForConditionalGeneration as M
        elif kind == "onevision":
            from transformers import LlavaOnevisionForConditionalGeneration as M
        elif kind == "llavanext":
            from transformers import LlavaNextForConditionalGeneration as M
        elif kind == "internvl":
            from transformers import InternVLForConditionalGeneration as M
        elif kind == "janus":
            from transformers import JanusForConditionalGeneration as M
        else:
            raise ValueError(kind)
        self.model = (
            M.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="eager")
            .to(DEV)
            .eval()
        )
        self.img_id = getattr(self.model.config, "image_token_index", None) or getattr(
            self.model.config, "image_token_id", None
        )
        self.tok = self.proc.tokenizer
        self.letter_ids = {c: self.tok(c, add_special_tokens=False).input_ids[0] for c in "ABCD"}
        self.layers = find_layers(self.model)
        self.N = len(self.layers)
        # mode: none | prefix (block i < depth) | suffix (block i >= depth)
        self.mode, self.depth = "none", 0
        self.rowscope = "alltext"  # alltext | lastrow (answer position only)
        self.vm = None
        self._mc = {}
        for i, lyr in enumerate(self.layers):
            lyr.register_forward_pre_hook(self._pre(i), with_kwargs=True)

    def _isolated(self, mask):
        key = id(mask)
        if key not in self._mc:
            m = mask.clone()
            if self.rowscope == "lastrow":
                # block only the ANSWER position's reads of image keys — tests the
                # direct image->answer pathway, leaving image->question flow intact
                tp = torch.tensor([m.shape[-2] - 1], device=m.device)
            else:
                tp = (~self.vm).nonzero().squeeze(-1)
            vp = self.vm.nonzero().squeeze(-1)
            neg = False if m.dtype == torch.bool else torch.finfo(m.dtype).min
            m[0, :, tp[:, None], vp[None, :]] = neg
            self._mc[key] = m
        return self._mc[key]

    def _pre(self, i):
        def f(_m, args, kwargs):
            active = (self.mode == "prefix" and i < self.depth) or (
                self.mode == "suffix" and i >= self.depth
            )
            if active and self.vm is not None:
                am = kwargs.get("attention_mask")
                if am is not None and am.dim() == 4:
                    kwargs = dict(kwargs)
                    kwargs["attention_mask"] = self._isolated(am)
                    return args, kwargs
            return None

        return f

    def _build(self, image, question):
        if self.kind == "gemma":
            msg = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            b = self.proc.apply_chat_template(
                msg,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            b = {k: (v.to(DEV) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        else:
            msg = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
            ]
            prompt = self.proc.apply_chat_template(msg, add_generation_prompt=True)
            b = self.proc(images=[image], text=[prompt], return_tensors="pt").to(DEV)
        vm = b["input_ids"][0] == self.img_id
        return b, vm

    @torch.no_grad()
    def _logits(self, b):
        return self.model(**b).logits[0, -1].float()

    def _score(self, lg, gt):
        s = {c: lg[self.letter_ids[c]].item() for c in "ABCD"}
        return dict(pred=max(s, key=s.get), nll=-lg.log_softmax(-1)[self.letter_ids[gt]].item())

    def _sweep(self, bI, bIp, vmI, vmIp, gt, mode, depths):
        out, out_null = [], []
        self.mode = mode
        for d in depths:
            self.depth = d
            self.vm = vmI
            self._mc.clear()
            out.append(self._score(self._logits(bI), gt))
            self.vm = vmIp
            self._mc.clear()
            out_null.append(self._score(self._logits(bIp), gt))
        self.mode = "none"
        return out, out_null

    @torch.no_grad()
    def run(self, doc, donor, do_causal, sweeps):
        gt = str(doc["answer"]).strip()
        if gt not in "ABCD":
            return dict(skip="bad_gt")
        q = doc_to_prompt(doc)
        bI, vmI = self._build(doc["image"].convert("RGB"), q)
        bIp, vmIp = self._build(donor["image"].convert("RGB"), q)
        rec = dict(
            gt=gt,
            category=doc.get("category"),
            n_vis=int(vmI.sum()),
            seq_len=int(bI["input_ids"].shape[1]),
        )
        self.mode, self.vm = "none", None
        rec["intact"] = self._score(self._logits(bI), gt)
        rec["swap"] = self._score(self._logits(bIp), gt)
        if not do_causal:
            return rec
        if "prefix" in sweeps:
            # block [0..d) for d=1..N  (cost[k] = depth k+1, matches old cost files)
            c, cn = self._sweep(bI, bIp, vmI, vmIp, gt, "prefix", range(1, self.N + 1))
            rec["cost"], rec["cost_null"] = c, cn
        if "suffix" in sweeps:
            # block [d..N) for d=0..N-1  (suf[k] = first k layers free)
            s, sn = self._sweep(bI, bIp, vmI, vmIp, gt, "suffix", range(0, self.N))
            rec["suf"], rec["suf_null"] = s, sn
        return rec


@torch.no_grad()
def main():
    model_id, kind, out_path = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    n_causal = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    sweeps = sys.argv[6] if len(sys.argv) > 6 else "suffix"
    sweeps = ("prefix", "suffix") if sweeps == "both" else (sweeps,)
    rowscope = sys.argv[7] if len(sys.argv) > 7 else "alltext"
    ds = load_vmcbench()
    n = min(n, len(ds))
    stride = max(1, n // max(n_causal, 1))
    causal_set = set(range(0, n, stride))
    done = sum(1 for _ in open(out_path)) if out_path.exists() else 0
    print(
        f"[win] {model_id} kind={kind} n={n} nc={len(causal_set)} sweeps={sweeps} "
        f"rowscope={rowscope} resume={done}",
        flush=True,
    )
    pr = WinProbe(model_id, kind)
    pr.rowscope = rowscope
    print(f"[win] N_layers={pr.N} img_id={pr.img_id}", flush=True)
    f = open(out_path, "a")
    for i in range(done, n):
        doc, donor = ds[i], ds[(i + 37) % n]
        try:
            rec = pr.run(doc, donor, do_causal=(i in causal_set), sweeps=sweeps)
            rec["i"] = i
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rec = dict(i=i, skip="oom")
        except Exception as e:  # noqa: BLE001
            rec = dict(i=i, skip=f"{type(e).__name__}: {e}")
        f.write(json.dumps(rec) + "\n")
        if (i + 1) % 25 == 0:
            f.flush()
            torch.cuda.empty_cache()
            print(f"[win] {i + 1}/{n}", flush=True)
    f.close()
    print("[win] done", flush=True)


if __name__ == "__main__":
    main()
