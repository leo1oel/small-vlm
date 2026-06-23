"""Per-layer VISUAL-PATHWAY vs TEXT-PATHWAY maturation for VLMs, VMCBench dev.

Core question: do encoder-free (native) VLMs spend early decoder layers
"encoding" the raw image patches (work a vision encoder would otherwise do),
so the image-token stream keeps changing through the early stack, whereas
encoder-based VLMs receive an already-encoded image (ViT->projector) and the
image-token stream is comparatively stable from layer 0?

For every decoder layer l we measure the residual-stream relative update of
each token, then average separately over image tokens and over text tokens:

  u_img(l) = mean_{p in image} ||h_l[p] - h_{l-1}[p]|| / ||h_{l-1}[p]||
  u_txt(l) = mean_{p in text}  ||h_l[p] - h_{l-1}[p]|| / ||h_{l-1}[p]||

plus the answer-position (last token) update u_last(l) and the mean image /
text hidden-state norms. h_0 = token embeddings (input to layer 0). Uses a
single forward with output_hidden_states=True (no hooks). cheap: 1 fwd/sample.

Handles encoder-based (llava, qwen) and encoder-free (gemma) HF models. NEO
and SAIL have their own scripts (custom trunks).

Usage: python devtools/pathway_maturation.py <model_id> <kind> <out.jsonl> [n]
  kind in {llava, qwen, gemma, onevision, llavanext}
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "neo_analysis"))
DEV = "cuda"
POST = "Answer with the option's letter from the given choices directly.\n"


def load_vmcbench():
    import datasets
    return datasets.load_dataset("suyc21/VMCBench", split="dev")


def doc_to_prompt(doc):
    op = "Options:\n" + "".join(f"{k}. {doc[k]}\n" for k in "ABCD")
    return f"Question: {doc['question']}\n{op}{POST}"


class MatProbe:
    def __init__(self, model_id, kind):
        from transformers import AutoProcessor
        self.kind = kind
        self.proc = AutoProcessor.from_pretrained(model_id)
        if kind == "llava":
            from transformers import LlavaForConditionalGeneration as M
            self.model = M.from_pretrained(model_id, dtype=torch.bfloat16).to(DEV).eval()
        elif kind == "qwen":
            from transformers import Qwen2_5_VLForConditionalGeneration as M
            self.model = M.from_pretrained(model_id, dtype=torch.bfloat16).to(DEV).eval()
        elif kind == "gemma":
            from transformers import Gemma4UnifiedForConditionalGeneration as M
            self.model = M.from_pretrained(model_id, dtype=torch.bfloat16).to(DEV).eval()
        elif kind == "onevision":
            from transformers import LlavaOnevisionForConditionalGeneration as M
            self.model = M.from_pretrained(model_id, dtype=torch.bfloat16).to(DEV).eval()
        elif kind == "llavanext":
            from transformers import LlavaNextForConditionalGeneration as M
            self.model = M.from_pretrained(model_id, dtype=torch.bfloat16).to(DEV).eval()
        else:
            raise ValueError(kind)
        self.img_id = (getattr(self.model.config, "image_token_index", None) or
                       getattr(self.model.config, "image_token_id", None))

    def _build(self, image, question):
        if self.kind == "gemma":
            content = [{"type": "image", "image": image},
                       {"type": "text", "text": question}]
            msg = [{"role": "user", "content": content}]
            b = self.proc.apply_chat_template(msg, add_generation_prompt=True,
                                              tokenize=True, return_dict=True,
                                              return_tensors="pt")
            b = {k: (v.to(DEV) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        else:
            msg = [{"role": "user", "content": [{"type": "image"},
                    {"type": "text", "text": question}]}]
            prompt = self.proc.apply_chat_template(msg, add_generation_prompt=True)
            b = self.proc(images=[image], text=[prompt], return_tensors="pt").to(DEV)
        vm = (b["input_ids"][0] == self.img_id)
        return b, vm

    @torch.no_grad()
    def run(self, doc):
        gt = str(doc["answer"]).strip()
        if gt not in "ABCD":
            return dict(skip="bad_gt")
        q = doc_to_prompt(doc)
        b, vm = self._build(doc["image"].convert("RGB"), q)
        if not bool(vm.any()):
            return dict(skip="no_vis")
        out = self.model(**b, output_hidden_states=True)
        hs = out.hidden_states  # tuple (N+1) of (1,S,H)
        N = len(hs) - 1
        txt = ~vm
        u_img, u_txt, u_last, n_img, n_txt = [], [], [], [], []
        for l in range(1, N + 1):
            a = hs[l - 1][0].float()
            c = hs[l][0].float()
            d = (c - a).norm(dim=-1)
            na = a.norm(dim=-1).clamp_min(1e-6)
            rel = d / na
            u_img.append(rel[vm].mean().item())
            u_txt.append(rel[txt].mean().item())
            u_last.append(rel[-1].item())
            n_img.append(c[vm].norm(dim=-1).mean().item())
            n_txt.append(c[txt].norm(dim=-1).mean().item())
        return dict(gt=gt, category=doc.get("category"), N=N,
                    n_vis=int(vm.sum()), seq_len=int(b["input_ids"].shape[1]),
                    u_img=u_img, u_txt=u_txt, u_last=u_last,
                    norm_img=n_img, norm_txt=n_txt)


@torch.no_grad()
def main():
    model_id, kind, out_path = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    ds = load_vmcbench()
    n = min(n, len(ds))
    done = sum(1 for _ in open(out_path)) if out_path.exists() else 0
    print(f"[mat] {model_id} kind={kind} n={n} resume={done}", flush=True)
    pr = MatProbe(model_id, kind)
    f = open(out_path, "a")
    for i in range(done, n):
        try:
            rec = pr.run(ds[i]); rec["i"] = i
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); rec = dict(i=i, skip="oom")
        except Exception as e:  # noqa: BLE001
            rec = dict(i=i, skip=f"{type(e).__name__}: {e}")
        f.write(json.dumps(rec) + "\n")
        if (i + 1) % 25 == 0:
            f.flush(); torch.cuda.empty_cache()
            print(f"[mat] {i + 1}/{n}", flush=True)
    f.close()
    print("[mat] done", flush=True)


if __name__ == "__main__":
    main()
