"""Fair-eval recipe sweep for the BREEN caption test (captain ask 2026-06-27).

Greedy decode on a mid-training native checkpoint loops on one token
("Image Image Image …"), which would read as a FALSE 'blind' verdict at
ckpt-1000. This loads ONE checkpoint and runs several decode recipes
(repetition_penalty x no_repeat_ngram_size x max_new_tokens) on the same diverse
images, flags which produce degenerate (looping) vs real text, and prints the
recipe to standardize across all 5 arms. Prompt is left at the trained format
(plain template drops the instruction; generate_response injects <query> after
the image — both already match training).

  CUDA_VISIBLE_DEVICES=5 python devtools/breen_caption_recipe_sweep.py --ckpt .../checkpoint-500
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vlm.inference.eval import generate_response  # noqa: E402
from vlm.inference.eval import load_model as eval_load_model  # noqa: E402

# (repetition_penalty, no_repeat_ngram_size, max_new_tokens)
RECIPES = [
    (None, None, 48),  # baseline greedy — should reproduce the loop
    (1.3, 3, 48),
    (1.3, 3, 64),
    (1.5, 3, 64),
    (1.2, 4, 96),
]


def _degenerate(text: str) -> bool:
    """Token-loop heuristic: one word dominates, or very few distinct words."""
    words = text.split()
    if len(words) < 4:
        return False
    c = Counter(w.lower() for w in words)
    top_frac = c.most_common(1)[0][1] / len(words)
    distinct_frac = len(c) / len(words)
    return top_frac > 0.5 or distinct_frac < 0.3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument(
        "--qual-dir",
        default="/mmfs1/gscratch/krishna/leoym/nemo/data/breen-s2val-j8/qual_images",
    )
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, processor, _ = eval_load_model(
        args.ckpt, bf16=True, attn_implementation="sdpa", device=device
    )
    model.eval()
    gc = model.generation_config
    if getattr(gc, "bos_token_id", None) is None:
        gc.bos_token_id = (
            processor.tokenizer.pad_token_id
            if processor.tokenizer.pad_token_id is not None
            else processor.tokenizer.eos_token_id
        )

    imgs = sorted(Path(args.qual_dir).glob("*.png"))[: args.n]
    pil = [Image.open(f).convert("RGB") for f in imgs]
    names = [f.name for f in imgs]
    print(f"=== recipe sweep: {args.ckpt} | {len(imgs)} images ===\n")

    for rp, ng, mx in RECIPES:
        caps = []
        for im in pil:
            try:
                cap = generate_response(
                    model,
                    processor,
                    query="<image>\nDescribe this image.",
                    images=im,
                    max_new_tokens=mx,
                    repetition_penalty=rp,
                    no_repeat_ngram_size=ng,
                )
            except Exception as e:  # noqa: BLE001
                cap = f"<gen failed: {type(e).__name__}: {e}>"
            caps.append(cap)
        n_degen = sum(_degenerate(c) for c in caps)
        distinct = len(set(caps))
        prefix_distinct = len({c[:40] for c in caps})
        tag = f"rp={rp} ngram={ng} max_new={mx}"
        verdict = (
            "DEGENERATE"
            if n_degen >= len(caps) - 1
            else ("BLIND" if prefix_distinct == 1 else "REAL-TEXT")
        )
        print(
            f"--- {tag} -> {verdict} "
            f"(degenerate {n_degen}/{len(caps)}, distinct {distinct}/{len(caps)}, "
            f"prefix-distinct {prefix_distinct}/{len(caps)}) ---"
        )
        for nm, c in zip(names, caps):
            print(f"  [{nm}] {c[:200]!r}")
        print()


if __name__ == "__main__":
    main()
