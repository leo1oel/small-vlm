"""Round-5A caption read: greedy caption generation on a checkpoint for a handful
of diverse images. Reports the ACTUAL strings + whether they are grounded
(different, describing each image) or blind (byte-identical / image-independent).
No distill teacher needed — pure generation path.

  CKPT=.../checkpoint-1000 GPU=3 bash devtools/native_distill_probe_srun.sh  # (adapt)
or directly:
  python devtools/breen_caption_test.py --ckpt .../checkpoint-1000
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vlm.inference.eval import generate_response  # noqa: E402
from vlm.inference.eval import load_model as eval_load_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument(
        "--qual-dir",
        default="/mmfs1/gscratch/krishna/leoym/nemo/data/breen-s2val-j8/qual_images",
    )
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="HF repetition_penalty (~1.3 breaks the greedy 'Image Image…' loop)",
    )
    ap.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=None,
        help="HF no_repeat_ngram_size (~3 forbids verbatim n-gram loops)",
    )
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
    if args.repetition_penalty is not None:
        gc.repetition_penalty = args.repetition_penalty
    if args.no_repeat_ngram_size is not None:
        gc.no_repeat_ngram_size = args.no_repeat_ngram_size

    imgs = sorted(Path(args.qual_dir).glob("*.png"))[: args.n]
    recipe = (
        f"max_new={args.max_new} repetition_penalty={args.repetition_penalty} "
        f"no_repeat_ngram_size={args.no_repeat_ngram_size}"
    )
    print(f"=== caption read: {args.ckpt} | {len(imgs)} images | {recipe} ===")
    caps: list[str] = []
    for f in imgs:
        try:
            cap = generate_response(
                model,
                processor,
                query="<image>\nDescribe this image.",
                images=Image.open(f).convert("RGB"),
                max_new_tokens=args.max_new,
            )
        except Exception as e:  # noqa: BLE001 — best-effort; note the fallback
            cap = f"<generate failed: {type(e).__name__}: {e}>"
        caps.append(cap)
        print(f"\n[{f.name}]\n  {cap!r}")

    # Grounded vs blind read
    uniq = set(caps)
    n_uniq = len(uniq)
    # also compare just the first 40 chars (prefix-identical = blind even if a
    # late token diverges)
    prefixes = {c[:40] for c in caps}
    print("\n=== cross-image read ===")
    print(f"  distinct full captions: {n_uniq}/{len(caps)}")
    print(f"  distinct 40-char prefixes: {len(prefixes)}/{len(caps)}")
    if n_uniq == 1:
        print("  VERDICT: BLIND — byte-identical across all images (image-independent).")
    elif len(prefixes) == 1:
        print("  VERDICT: BLIND-ish — identical prefix, only late tokens differ.")
    else:
        print(
            f"  VERDICT: captions DIFFER across images ({n_uniq} distinct) — inspect for grounding."
        )


if __name__ == "__main__":
    main()
