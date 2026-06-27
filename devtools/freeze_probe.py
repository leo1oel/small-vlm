"""FREEZE probe: causally test what the PRE-FUSION layers do, per stream.

freeze_img@k: for layers [0..k), overwrite the IMAGE positions' layer output
with their layer input — the image residual stream stays at its layer-0
embedding until layer k, cancelling all in-LLM image processing before k
(text stream and all attention reads proceed normally; at k the image tokens
"enter" the model DeepInsert-style).

freeze_txt@k: same for TEXT positions (excluding position 0, so the BOS/sink
token develops normally and we don't destroy attention for trivial reasons).

Predictions: if encoder-model early layers do nothing essential to the image,
freeze_img is harmless up to fusion onset (and harmful for natives whose early
layers encode the image). freeze_txt should be harmful early everywhere EXCEPT
regions where text is architecturally idle (NEO pre-Buffer).

Measures intact + swap letter accuracy at each k (image-signal R0(k) =
intact-swap separates "model still works" from "image still used").

Usage: python devtools/freeze_probe.py <model_id> <kind> <out.jsonl> [n] [n_causal]
  kind in {llava, qwen, gemma}
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


class FreezeProbe:
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
        self.mode, self.k = "none", 0  # mode: none | img | txt
        self.pos = None  # bool mask (S,) of positions to freeze
        self._in = {}
        for i, lyr in enumerate(self.layers):
            lyr.register_forward_pre_hook(self._pre(i), with_kwargs=True)
            lyr.register_forward_hook(self._post(i), with_kwargs=True)

    # freeze window is [1..k): layer 0 is SPARED so the universal embedding->
    # residual norm jump (u_txt(L0) is ~100-200x the per-layer average in every
    # model) still happens; freezing it would destroy any stream trivially and
    # mask the real division-of-labor signal.
    def _pre(self, i):
        def f(_m, args, kwargs):
            if self.mode != "none" and 1 <= i < self.k:
                h = args[0] if args else kwargs.get("hidden_states")
                self._in[i] = h
            return None

        return f

    def _post(self, i):
        def f(_m, _args, _kwargs, out):
            if self.mode != "none" and 1 <= i < self.k and i in self._in:
                h = out[0] if isinstance(out, tuple) else out
                h[0, self.pos] = self._in[i][0, self.pos]
                del self._in[i]
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

    @torch.no_grad()
    def run(self, doc, donor, do_causal, ks):
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
        self.mode = "none"
        rec["intact"] = self._score(self._logits(bI), gt)
        rec["swap"] = self._score(self._logits(bIp), gt)
        if not do_causal:
            return rec
        rec["ks"] = ks
        for mode in ("img", "txt"):
            res, res_null = [], []
            for k in ks:
                self.mode, self.k = mode, k
                for b, vm, acc_list in ((bI, vmI, res), (bIp, vmIp, res_null)):
                    if mode == "img":
                        self.pos = vm
                    else:
                        self.pos = ~vm
                        self.pos[0] = False  # let BOS/sink develop
                    self._in.clear()
                    acc_list.append(self._score(self._logits(b), gt))
            rec[f"freeze_{mode}"], rec[f"freeze_{mode}_null"] = res, res_null
        self.mode = "none"
        return rec


@torch.no_grad()
def main():
    model_id, kind, out_path = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    n_causal = int(sys.argv[5]) if len(sys.argv) > 5 else 150
    ds = load_vmcbench()
    n = min(n, len(ds))
    stride = max(1, n // max(n_causal, 1))
    causal_set = set(range(0, n, stride))
    done = sum(1 for _ in open(out_path)) if out_path.exists() else 0
    pr = FreezeProbe(model_id, kind)
    # relative freeze depths; includes deep values to chart the full curve
    rels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8]
    ks = sorted({max(1, round(r * pr.N)) for r in rels})
    print(
        f"[freeze] {model_id} kind={kind} N={pr.N} ks={ks} n={n} nc={len(causal_set)} resume={done}",
        flush=True,
    )
    f = open(out_path, "a")
    for i in range(done, n):
        doc, donor = ds[i], ds[(i + 37) % n]
        try:
            rec = pr.run(doc, donor, do_causal=(i in causal_set), ks=ks)
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
            print(f"[freeze] {i + 1}/{n}", flush=True)
    f.close()
    print("[freeze] done", flush=True)


if __name__ == "__main__":
    main()
