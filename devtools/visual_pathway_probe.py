"""Visual-pathway causal probe: is vision actually USED, or is the pathway broken?

Background: visualffn/baseline score ~0.40 on VMCBench and POPE collapses to a
96.7% "yes" constant classifier. Question: is that a BUG (image never reaches /
never influences the LLM) or a CAPACITY/DATA ceiling (pathway works, visual
features are just weak)? This probe answers it by ablation at two layers:

  Behaviour layer — for the SAME question, greedy-decode the answer under three
    image conditions: REAL image / SWAPPED image (a different sample's image) /
    BLANK image (all-zero pixels). If the answer almost never changes, the LLM
    is ignoring vision.
  Representation layer — hook the connector (RawPatchConnector) and compare the
    image embedding for REAL vs BLANK (cosine) and across DIFFERENT real images
    (cosine). If REAL≈BLANK, the projection isn't encoding pixels at all.

Decision table (printed at the end):
  repr REAL≈BLANK (cos>0.98)                  -> projection dead         (DEEP BUG)
  repr differs but behaviour REAL≈BLANK≈SWAP  -> LLM ignores image       (FUSION BUG)
  both differ, accuracy still low             -> pathway OK, vision weak (ARCH/DATA ceiling)

Usage: python devtools/visual_pathway_probe.py <ckpt_dir> [N]
"""

import sys

import torch
from PIL import Image

from vlm.inference.eval import generate_response, load_model

CKPT = sys.argv[1] if len(sys.argv) > 1 else (
    "/gscratch/scrubbed/leoym/small-vlm-outputs/sft-unified-bee-mix-visualffn/checkpoint-5000"
)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 50
POST = "Answer with the option's letter from the given choices directly.\n"


def get_samples(n):
    import datasets
    ds = datasets.load_dataset("suyc21/VMCBench", split="dev")
    out = []
    for i in range(min(n, len(ds))):
        d = ds[i]
        op = "Options:\n" + "".join(f"{k}. {d[k]}\n" for k in "ABCD")
        q = f"<image>\nQuestion: {d['question']}\n{op}{POST}"
        out.append((d["image"].convert("RGB"), q, str(d["answer"]).strip().upper()))
    return out


def first_letter(s):
    s = (s or "").strip().upper()
    return s[0] if s and s[0] in "ABCD" else "?"


def main():
    print(f"=== loading {CKPT} ===", flush=True)
    model, processor, _ = load_model(CKPT, bf16=True)
    model.eval()

    captured = {}
    conn = model.model.connector

    def hook(_m, _i, out):
        t = out[0] if isinstance(out, tuple) else out
        if torch.is_tensor(t):
            captured["emb"] = t.detach().float().mean(dim=0)  # mean over patches -> (D,)
    h = conn.register_forward_hook(hook)
    print(f"hooked connector: {type(conn).__name__}", flush=True)

    samples = get_samples(N)
    print(f"N={len(samples)} VMCBench dev samples\n", flush=True)

    n_correct = 0
    chg_blank = 0   # answer(real) != answer(blank)
    chg_swap = 0    # answer(real) != answer(swap)
    cos_real_blank = []   # repr: same question, real vs blank image
    real_embs = []        # repr: collect real-image embeddings for cross-image cos

    cos = torch.nn.functional.cosine_similarity
    for i, (img, q, gold) in enumerate(samples):
        blank = Image.new("RGB", img.size, 0)
        swap = samples[(i + 1) % len(samples)][0]

        a_real = first_letter(generate_response(model, processor, query=q, images=img,
                                                temperature=0.0, max_new_tokens=8))
        emb_real = captured.get("emb")
        if emb_real is not None:
            real_embs.append(emb_real)

        a_blank = first_letter(generate_response(model, processor, query=q, images=blank,
                                                 temperature=0.0, max_new_tokens=8))
        emb_blank = captured.get("emb")
        if emb_real is not None and emb_blank is not None and emb_real.shape == emb_blank.shape:
            cos_real_blank.append(cos(emb_real[None], emb_blank[None]).item())

        a_swap = first_letter(generate_response(model, processor, query=q, images=swap,
                                                temperature=0.0, max_new_tokens=8))

        n_correct += (a_real == gold)
        chg_blank += (a_real != a_blank)
        chg_swap += (a_real != a_swap)
        if i < 12:
            print(f"[{i:02d}] gold={gold} real={a_real} blank={a_blank} swap={a_swap} "
                  f"cos(real,blank)={cos_real_blank[-1]:.3f}" if cos_real_blank else
                  f"[{i:02d}] gold={gold} real={a_real} blank={a_blank} swap={a_swap}", flush=True)
    h.remove()

    # cross-image repr similarity: how distinct are different real images?
    cross = []
    for i in range(min(len(real_embs), 20)):
        for j in range(i + 1, min(len(real_embs), 20)):
            cross.append(cos(real_embs[i][None], real_embs[j][None]).item())

    n = len(samples)
    acc = n_correct / n
    cb = sum(cos_real_blank) / len(cos_real_blank) if cos_real_blank else float("nan")
    cx = sum(cross) / len(cross) if cross else float("nan")
    rb_rate = chg_blank / n
    rs_rate = chg_swap / n

    print("\n========== SUMMARY ==========", flush=True)
    print(f"accuracy (real image)            : {acc:.3f}  (random=0.25)", flush=True)
    print(f"answer change REAL->BLANK        : {rb_rate:.3f}  ({chg_blank}/{n})", flush=True)
    print(f"answer change REAL->SWAP         : {rs_rate:.3f}  ({chg_swap}/{n})", flush=True)
    print(f"repr cosine REAL vs BLANK        : {cb:.3f}  (low=projection encodes pixels)", flush=True)
    print(f"repr cosine across DIFFERENT imgs: {cx:.3f}  (low=embeddings vary by image)", flush=True)

    print("\n========== DIAGNOSIS ==========", flush=True)
    if cb > 0.98:
        print("-> PROJECTION DEAD: real vs blank image embeddings near-identical.", flush=True)
        print("   The connector is NOT encoding pixel content. DEEP BUG.", flush=True)
    elif rb_rate < 0.10 and rs_rate < 0.10:
        print("-> LLM IGNORES IMAGE: connector encodes pixels (repr differs) but the", flush=True)
        print("   answer barely moves when the image is blanked/swapped. FUSION BUG", flush=True)
        print("   (image tokens not attended / spliced wrong / masked out).", flush=True)
    else:
        print("-> PATHWAY OK, VISION WEAK: image measurably changes both the", flush=True)
        print(f"   representation (cos real/blank={cb:.2f}) and the answer", flush=True)
        print(f"   (change rate blank={rb_rate:.2f}, swap={rs_rate:.2f}), yet accuracy", flush=True)
        print(f"   is only {acc:.2f}. Not a bug -> capacity/data ceiling.", flush=True)
    print("=================================", flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
