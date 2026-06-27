"""Does native VLM training compress the pretrained LLM's text processing?

Per-layer text update norm + text-only NLL, on a fixed text corpus, comparing
a finetuned VLM against its own base LLM. If native training pushed the early
LLM layers toward identity-for-text (the user's capacity-compression
hypothesis), the VLM's early-layer update norms drop below the base LLM's and
its text NLL rises.

  update_norm[l] = mean_t ||h_l[t]-h_{l-1}[t]|| / ||h_{l-1}[t]||   (low = identity-like)
  cos[l]         = mean_t cos(h_l[t], h_{l-1}[t])                  (high = identity-like)
  nll            = mean next-token CE on the same text

Runs in the MAIN .venv (transformers 5.10). Handles two model kinds:
  qwen   : a plain AutoModelForCausalLM path (the base LLM)
  vlm    : a small-vlm checkpoint via inference.load_model, text-only forward

Usage: python devtools/text_capacity_probe.py <out.json> <tag> <kind> <path> [n]
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neo_analysis"))
DEV = "cuda"


def corpus(n=200):
    # Neutral prose (out of every checkpoint's VQA training distribution) so
    # NLL reflects general language modeling, not task-format familiarity.
    from neutral_corpus import TEXTS

    return TEXTS[:n] if n < len(TEXTS) else TEXTS


@torch.no_grad()
def metrics(hidden_states, logits, ids):
    H = [h[0].float() for h in hidden_states]  # (T,D) each, len N+1 (emb + N layers)
    norms, coss = [], []
    for l in range(1, len(H)):
        d = H[l] - H[l - 1]
        rel = d.norm(dim=-1) / (H[l - 1].norm(dim=-1) + 1e-6)
        cs = torch.nn.functional.cosine_similarity(H[l], H[l - 1], dim=-1)
        norms.append(rel[1:].mean().item())  # skip BOS-ish first position
        coss.append(cs[1:].mean().item())
    lg = logits[0].float()
    lp = lg[:-1].log_softmax(-1)
    tgt = ids[0][1:]
    nll = -lp[torch.arange(tgt.shape[0]), tgt].mean().item()
    return norms, coss, nll


@torch.no_grad()
def run_qwen(path, texts):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(DEV).eval()
    nl = model.config.num_hidden_layers
    agg = {"norm": [[] for _ in range(nl)], "cos": [[] for _ in range(nl)], "nll": []}
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to(DEV)
        if ids.shape[1] < 3:
            continue
        out = model(ids, output_hidden_states=True)
        norms, coss, nll = metrics(out.hidden_states, out.logits, ids)
        for l in range(nl):
            agg["norm"][l].append(norms[l])
            agg["cos"][l].append(coss[l])
        agg["nll"].append(nll)
    return agg, nl


@torch.no_grad()
def run_vlm(path, texts):
    from vlm.inference.eval import load_model

    model, processor, _ = load_model(path, bf16=True, attn_implementation="eager")
    tok = processor.tokenizer
    nl = model.config.num_hidden_layers
    agg = {"norm": [[] for _ in range(nl)], "cos": [[] for _ in range(nl)], "nll": []}
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to(DEV)
        if ids.shape[1] < 3:
            continue
        out = model(input_ids=ids, output_hidden_states=True)
        norms, coss, nll = metrics(out.hidden_states, out.logits, ids)
        for l in range(nl):
            agg["norm"][l].append(norms[l])
            agg["cos"][l].append(coss[l])
        agg["nll"].append(nll)
    return agg, nl


def main():
    out_path, tag, kind, path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    texts = corpus(n)
    print(f"[textcap] tag={tag} kind={kind} n_text={len(texts)}", flush=True)
    agg, nl = run_vlm(path, texts) if kind == "vlm" else run_qwen(path, texts)
    from statistics import mean

    rec = dict(
        tag=tag,
        kind=kind,
        path=path,
        n_layers=nl,
        norm=[mean(x) for x in agg["norm"]],
        cos=[mean(x) for x in agg["cos"]],
        nll=mean(agg["nll"]),
        nll_n=len(agg["nll"]),
    )
    out = json.loads(Path(out_path).read_text()) if Path(out_path).exists() else {}
    out[tag] = rec
    Path(out_path).write_text(json.dumps(out))
    print(
        f"[textcap] {tag}: nll={rec['nll']:.4f} nl={nl} "
        f"norm[:6]={[round(x, 3) for x in rec['norm'][:6]]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
