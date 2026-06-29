"""Multi-arm caption read: greedy bare-`<image>` caption generation across several
checkpoints at once, for grounded-vs-blind string-distinctness. Caption models
train on the PLAIN template (bare image -> caption), so bare `<image>` is their
natural inference mode; generate splices the image at prefill so captions should
be image-distinct.

For each checkpoint, N diverse images, fair decoder (rep_penalty 1.3,
no_repeat_ngram 3, max_new 64, greedy): generate the bare-`<image>` caption via
generate_response (= model.generate). Reports distinct ratio + all N captions.

Works on ANY native checkpoint (no distill head). The BREEN campaign found these
trivially "blind" under a FROZEN LM — but the 10-experiment arms UNFREEZE the LM
at S1+S2, so this becomes a genuinely informative grounding signal.

  CUDA_VISIBLE_DEVICES=5 python devtools/breen_caption_retest.py --arms A3=/path ...
"""

import argparse
import gc
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vlm.inference.eval import generate_response  # noqa: E402
from vlm.inference.eval import load_model as eval_load_model  # noqa: E402

QUAL = "/mmfs1/gscratch/krishna/leoym/nemo/data/breen-s2val-j8/qual_images"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="LABEL=ckpt ...")
    ap.add_argument("--qual-dir", default=QUAL)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    imgs = sorted(Path(args.qual_dir).glob("*.png"))[: args.n]
    pil = [Image.open(f).convert("RGB") for f in imgs]
    names = [f.name for f in imgs]

    for spec in args.arms:
        label, _, ckpt = spec.partition("=")
        if not Path(ckpt).exists():
            print(f"##### {label}: MISSING {ckpt}\n")
            continue
        m, proc, _ = eval_load_model(ckpt, bf16=True, attn_implementation="sdpa", device=dev)
        m.eval()
        m.config.loss_chunk_size = 0
        gc_ = m.generation_config
        if getattr(gc_, "bos_token_id", None) is None:
            gc_.bos_token_id = (
                proc.tokenizer.pad_token_id
                if proc.tokenizer.pad_token_id is not None
                else proc.tokenizer.eos_token_id
            )
        caps = []
        for im in pil:
            try:
                c = generate_response(
                    m,
                    proc,
                    query="<image>",
                    images=im,
                    max_new_tokens=args.max_new,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                )
            except Exception as e:  # noqa: BLE001
                c = f"<failed: {type(e).__name__}: {e}>"
            caps.append(c.strip())
        d = len(set(caps))
        pre = len({c[:40] for c in caps})
        verdict = (
            "IMAGE-DISTINCT"
            if pre >= max(2, len(caps) - 1)
            else ("partial" if pre > 1 else "BLIND")
        )
        print(f"##### {label}  ({ckpt})")
        print(f"   distinct {d}/{len(caps)} | prefix-distinct {pre}/{len(caps)} -> {verdict}")
        for nm, c in zip(names, caps):
            print(f"   [{nm}] {c[:240]!r}")
        print()
        del m
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
