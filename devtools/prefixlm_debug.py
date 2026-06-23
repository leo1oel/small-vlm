"""Diagnose prefix_lm degenerate (prompt-echo / empty) generation.

Confirmed so far: prefix_lm bidirectional prefill -> top1 = EOS (151645) -> empty;
pure causal -> top1 = correct letter. No NaN. So the bidirectional INFERENCE path
emits EOS. This script localizes it: for the SAME inputs_embeds + bidirectional 4D
mask, compare the last-position logits under use_cache=False vs True (training uses
no cache; generation prefill uses cache). If they differ, the cached-prefill path
mishandles the 4D mask.

Usage: python devtools/prefixlm_debug.py <prefixlm_ckpt_dir>
"""

import sys

import torch

from vlm.inference.eval import generate_response, load_model

CKPT = sys.argv[1] if len(sys.argv) > 1 else (
    "/gscratch/scrubbed/leoym/small-vlm-outputs/sft-unified-bee-mix-prefixlm/checkpoint-3000"
)
POST = "Answer with the option's letter from the given choices directly.\n"


def get_sample(i=0):
    import datasets
    ds = datasets.load_dataset("suyc21/VMCBench", split="dev")
    d = ds[i]
    op = "Options:\n" + "".join(f"{k}. {d[k]}\n" for k in "ABCD")
    return d["image"].convert("RGB"), f"<image>\nQuestion: {d['question']}\n{op}{POST}", str(d["answer"]).strip()


def instrument(model):
    import functools
    cls = type(model)
    orig = cls.forward
    state = {"calls": 0, "done_probe": False}

    @functools.wraps(orig)
    def logged(self, *a, **kw):
        ie = kw.get("inputs_embeds", a[1] if len(a) > 1 else None)
        gm = getattr(self, "_xmodal_gen_mask", None)
        is_prefill = ie is not None and ie.shape[1] > 1
        # One-shot use_cache probe at the bidirectional prefill, BEFORE consuming the stash.
        if is_prefill and gm is not None and not state["done_probe"]:
            state["done_probe"] = True
            with torch.no_grad():
                pid = kw.get("position_ids")
                for uc in (False, True):
                    o = self.model(inputs_embeds=ie, attention_mask=gm,
                                   position_ids=pid, use_cache=uc)
                    h = o[0] if isinstance(o, tuple) else o.last_hidden_state
                    lg = self.lm_head(h[0, -1]).float()
                    t = torch.topk(lg, 3)
                    print(f"   [PROBE inner-model bidir mask use_cache={uc}] "
                          f"nan={bool(torch.isnan(lg).any())} top3_ids={t.indices.tolist()} "
                          f"vals={[round(x,2) for x in t.values.tolist()]}", flush=True)
        out = orig(self, *a, **kw)
        if is_prefill and hasattr(out, "logits") and out.logits is not None and state["calls"] < 4:
            lg = out.logits[0, -1].float()
            t = torch.topk(lg, 5)
            print(f"[fwd #{state['calls']}] ie={tuple(ie.shape)} gen_mask={None if gm is None else tuple(gm.shape)} "
                  f"top5_ids={t.indices.tolist()}", flush=True)
        state["calls"] += 1
        return out

    cls.forward = logged


def main():
    print(f"=== loading {CKPT} ===", flush=True)
    model, processor, _ = load_model(CKPT, bf16=True)
    print("attn_impl =", model.config._attn_implementation,
          "| mode =", getattr(model.config, "cross_modal_mask_mode", None), flush=True)
    image, query, gold = get_sample(0)
    print(f"GOLD = {gold}", flush=True)
    instrument(model)

    print("=== A) prefix_lm (bidirectional) ===", flush=True)
    a = generate_response(model, processor, query=query, images=image, temperature=0.0, max_new_tokens=32)
    print(f"OUTPUT_A = {a!r}", flush=True)

    print("=== B) force causal (mode=none) ===", flush=True)
    model.config.cross_modal_mask_mode = "none"
    model._xmodal_gen_mask = None
    b = generate_response(model, processor, query=query, images=image, temperature=0.0, max_new_tokens=32)
    print(f"OUTPUT_B = {b!r}", flush=True)
    print(f"=== SUMMARY: A={a!r} B={b!r} ===", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
