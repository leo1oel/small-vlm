"""Distill-free cross-image feature probe for the NON-distill native arms
(exps 1, 2, 3, 6, 10 of the 10-experiment ablation; spec data/tenexp-audit/
report.md §7 "Needs building: a distill-free feature probe").

The distill arms get breen_probe_xshape.py, which reads the (student, teacher)
pair off model.visual_distill_head. The non-distill arms have no head, so this
twin reuses the SAME _pool / retrieval / centering block
(devtools/breen_probe_common.py) but reads the RAW LLM hidden state at image
positions — so it runs on ANY native (encoder-free) checkpoint.

Cross-image discrimination WITHOUT a teacher = SPLIT-HALF retrieval: split each
image's per-patch LLM hidden states into two interleaved halves (even / odd
patch index — each half samples the whole image, no spatial bias), pool+L2-norm
each into A_i / B_i, and ask whether A_i retrieves its own B_i among all images.

  LEARNING: A_i·B_i (self) >> A_i·B_j (cross), retrieval >> chance, self_centered
            stays high  => the LLM builds DISTINCT per-image features at the image
            positions.
  COLLAPSED: self ~= cross, retrieval ~= chance, centered ~= 0  => image positions
             carry a near-constant (the language-prior-machine failure mode).

`full_spread_offdiag` (mean off-diagonal cos among the all-patch pooled
descriptors) is the direct collapse detector: ~1.0 => every image looks the same.

Image positions are located robustly by forcing `with_image_block_ids=True` on
the multimodal splice and reading back image_block_ids (>= 0 marks patch rows) —
no assumption about where the image lands in the sequence.

Run via SLURM (heavy model load; the login node OOMs):
  CKPT=.../checkpoint-1000 LABEL=nepa@1000 OUT=/path/out.json PROBE=feat \
    sbatch devtools/s1_eval_probe.slurm
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from breen_probe_common import (  # noqa: E402
    _offdiag_mean,
    _pool,
    apply_query_placement_override,
    discrimination_metrics,
    load_probe_images,
)

from vlm.inference.eval import load_model as eval_load_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-images", type=int, default=30)
    ap.add_argument(
        "--images-dir",
        default=None,
        help="if set, probe the sorted PNGs in this dir instead of VMCBench "
        "(offline-friendly; e.g. .../breen-s2val-j8/qual_images).",
    )
    ap.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="hidden_states index to read (0 = input embeds, -1 = final layer). "
        "Default final layer — the most processed image representation.",
    )
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--query-placement",
        default=None,
        choices=["after_image", "after_text"],
        help="defensive override for a stale query checkpoint (no-op for "
        "main-trained arms; native arms have no query block anyway).",
    )
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    imgs = load_probe_images(args.n_images, images_dir=args.images_dir)

    model, processor, _ = eval_load_model(
        args.ckpt, bf16=True, attn_implementation="sdpa", device=device
    )
    apply_query_placement_override(model, args.query_placement)
    model.eval()
    img_tok = int(model.config.image_token_index)
    mdtype = next(model.parameters()).dtype

    # Locate image positions robustly: force the multimodal splice to emit
    # image_block_ids (>= 0 on the spliced patch rows) and read them back. This
    # is the only path that builds them for a plain arm in eval (visual_aux /
    # distill / expert gates are all off), and it makes no assumption about where
    # the image block lands in the sequence.
    cap: dict[str, torch.Tensor] = {}
    orig_splice = model.prepare_inputs_labels_for_multimodal

    def spy(*a: object, **kw: object):
        kw["with_image_block_ids"] = True
        out = orig_splice(*a, **kw)
        cap["ibid"] = out[6]  # (input_ids, pos, attn, pkv, embeds, labels, IBID, qbid)
        return out

    model.prepare_inputs_labels_for_multimodal = spy

    A_, B_, FULL_ = [], [], []
    skipped = 0
    for im in imgs:
        feat = processor.image_processor.preprocess([im])
        px = feat["pixel_values"][0].to(device=device, dtype=mdtype)
        pos = feat["image_position_ids"][0].to(device)
        seq = torch.tensor([[img_tok, 13]], device=device)  # <image> + 1 dummy tok
        cap.clear()
        with torch.no_grad():
            out = model(
                input_ids=seq,
                attention_mask=torch.ones_like(seq),
                images=[px],
                image_position_ids=[pos],
                output_hidden_states=True,
                use_cache=False,
            )
        ibid = cap.get("ibid")
        if ibid is None:
            skipped += 1
            continue
        mask = ibid[0] >= 0  # (L,) bool — image patch rows
        hidden = out.hidden_states[args.layer][0]  # (L, H)
        img_hidden = hidden[mask.to(hidden.device)]  # (m, H) image positions
        if img_hidden.shape[0] < 2:  # need >= 2 patches to split-half
            skipped += 1
            continue
        A_.append(_pool(img_hidden[0::2]))  # even patches
        B_.append(_pool(img_hidden[1::2]))  # odd patches
        FULL_.append(_pool(img_hidden))  # all patches

    model.prepare_inputs_labels_for_multimodal = orig_splice

    if len(A_) < 2:
        raise SystemExit(
            f"need >= 2 usable images for a cross-image probe, got {len(A_)} "
            f"({skipped} skipped); check --images-dir / VMCBench and --n-images"
        )
    A = torch.stack(A_)  # (N, d) half-1 pooled
    B = torch.stack(B_)  # (N, d) half-2 pooled
    FULL = torch.stack(FULL_)
    m = discrimination_metrics(A, B)  # Q=half-1, K=half-2 (split-half retrieval)
    N = m["n_images"]

    n_layers = len(out.hidden_states)
    res = {
        "label": args.label or Path(args.ckpt).name,
        "ckpt": args.ckpt,
        "probe": "feat-splithalf",
        "layer": args.layer,
        "n_hidden_states": n_layers,
        "n_images": N,
        "n_skipped": skipped,
        "chance_top1": m["chance_top1"],
        "self_pooled": m["self_pooled"],  # within-image half-to-half cos
        "cross_pooled": m["cross_pooled"],
        "self_minus_cross": m["self_minus_cross"],
        "retrieval_top1": m["retrieval_top1"],
        "median_rank": m["median_rank"],
        "full_spread_offdiag": _offdiag_mean(FULL @ FULL.t()),  # collapse detector
        "self_centered": m["self_centered"],
        "cross_centered": m["cross_centered"],
        "retrieval_top1_centered": m["retrieval_top1_centered"],
    }

    top1_threshold = 3.0 / N
    top1_ok = (
        res["retrieval_top1"] >= 1.0
        if top1_threshold >= 1.0
        else res["retrieval_top1"] > top1_threshold
    )
    learning = res["self_minus_cross"] > 0.10 and top1_ok
    res["verdict"] = "LEARNING" if learning else "COLLAPSED/WEAK"

    print(json.dumps(res, indent=2))
    print(
        f"\nVERDICT[feat L{args.layer} @ {res['label']}]: {res['verdict']}  "
        f"self={res['self_pooled']:.3f} cross={res['cross_pooled']:.3f} "
        f"gap={res['self_minus_cross']:.3f} "
        f"top1={res['retrieval_top1']:.1%} (chance {m['chance_top1']:.1%})  "
        f"centered-self={res['self_centered']:.3f}  "
        f"full-spread={res['full_spread_offdiag']:.3f}"
    )
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"[wrote {args.out}]")


if __name__ == "__main__":
    main()
