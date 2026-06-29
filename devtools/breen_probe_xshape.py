"""Cross-image discrimination probe for the DISTILL arms (exps 4,5,7,8,9 of the
10-experiment ablation; spec data/tenexp-audit/report.md §7).

The killer test from data/distillchk-d4/report.md: distill_cos alone is a MIRAGE
(a collapsed constant satisfies a per-row cosine). This probe loads a trained
distill-arm checkpoint, attaches the frozen CLIP teacher, runs the TRAINING-mode
forward on ~30 diverse images, and captures the exact per-image (student
projected pred, frozen teacher target) pairs the distill loss aligns — via a
monkeypatch on VisualDistillHead._align (the common chokepoint for eve/repa/
softdepth). It then asks whether the distilled features carry real PER-IMAGE
structure or collapsed onto a shared constant (see discrimination_metrics in
devtools/breen_probe_common.py):

  self_cos (per-patch) = mean_patch cos(pred_i, target_i)         (== distill_cos)
  self_pooled          = cos(pool(pred_i), pool(target_i))
  cross_pooled         = mean_{i!=j} cos(pool(pred_i), pool(target_j))
  retrieval_top1       = frac_i [ argmax_j cos(pool(pred_i), pool(target_j))==i ]
  *_spread             = mean off-diagonal cos among pooled targets / pooled preds
  *_centered           = self/cross/retrieval after subtracting the across-image
                         MEAN (the constant) — distillchk's clincher.

  LEARNING (good): self_pooled >> cross_pooled, retrieval >> chance, centered-self
                   stays high  => the queries/patches build per-image features.
  COLLAPSED (bad): self ~= cross, retrieval ~= chance, centered-self ~= 0.

REQUIRES a checkpoint trained WITH visual_distill (reads model.visual_distill_head;
attach_distill_teacher raises otherwise). For the non-distill arms (1,2,3,6,10)
use devtools/breen_probe_feat.py (same metric on the raw LLM hidden state).

Plus a caption eyeball (generate_response, plain template) on 3 qual images:
does the caption actually describe THAT image?

Run via SLURM (heavy model + CLIP load; the login node OOMs):
  CKPT=.../checkpoint-1000 LABEL=eve@1000 OUT=/path/out.json PROBE=xshape \
    sbatch devtools/s1_eval_probe.slurm
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from breen_probe_common import (  # noqa: E402
    _pool,
    apply_query_placement_override,
    discrimination_metrics,
    load_probe_images,
    seed_bos,
)

from vlm.inference.eval import generate_response  # noqa: E402
from vlm.inference.eval import load_model as eval_load_model  # noqa: E402
from vlm.vlm import attach_distill_teacher  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-images", type=int, default=30)
    ap.add_argument(
        "--qual-dir",
        default="/mmfs1/gscratch/krishna/leoym/nemo/data/breen-s2val-j8/qual_images",
    )
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--images-dir",
        default=None,
        help="if set, run the cross-image probe on the sorted PNGs in this dir "
        "(e.g. the 6 caption-retest qual images) instead of VMCBench — for "
        "apples-to-apples cross-arm comparison on an identical image set, and "
        "to avoid the VMCBench download on an offline compute node.",
    )
    ap.add_argument(
        "--query-placement",
        default=None,
        choices=["after_image", "after_text"],
        help="defensive override for a stale/external query checkpoint whose "
        "config.json disagrees with how it trained; no-op for main-trained arms "
        "(vlm() already serializes the correct value).",
    )
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    imgs = load_probe_images(args.n_images, images_dir=args.images_dir)

    model, processor, _ = eval_load_model(
        args.ckpt, bf16=True, attn_implementation="sdpa", device=device
    )
    apply_query_placement_override(model, args.query_placement)
    attach_distill_teacher(model)
    model.config.visual_distill_weight = 1.0
    if int(getattr(model.config, "loss_chunk_size", 0) or 0) <= 0:
        model.config.loss_chunk_size = 1024

    head = model.visual_distill_head
    method = head.method
    img_tok = int(model.config.image_token_index)
    ign = int(model.config.ignore_index)
    mdtype = next(model.parameters()).dtype

    # Capture the (pred, target) pair the loss aligns, for every method.
    cap: dict[str, torch.Tensor] = {}
    orig_align = head._align

    def spy(pred: torch.Tensor, target: torch.Tensor):
        cap["pred"] = pred.detach().float().cpu()
        cap["target"] = target.detach().float().cpu()
        return orig_align(pred, target)

    head._align = spy
    model.train()  # distill_on requires self.training (no_grad still runs distill)

    preds, targs, self_cos_pp = [], [], []
    for im in imgs:
        feat = processor.image_processor.preprocess([im])
        px = feat["pixel_values"][0].to(device=device, dtype=mdtype)
        pos = feat["image_position_ids"][0].to(device)
        seq = torch.tensor([[img_tok, 13]], device=device)  # <image> + 1 dummy caption tok
        lab = torch.tensor([[ign, 13]], device=device)  # supervise the dummy so CE runs
        cap.clear()
        with torch.no_grad():
            model(
                input_ids=seq,
                attention_mask=torch.ones_like(seq),
                labels=lab,
                images=[px],
                image_position_ids=[pos],
            )
        if "pred" not in cap:  # degenerate image (<2-cell grid) -> distill anchored, skip
            continue
        p, t = cap["pred"], cap["target"]  # (m, d_teacher)
        preds.append(_pool(p))
        targs.append(_pool(t))
        self_cos_pp.append(float(F.cosine_similarity(p, t, dim=-1).mean()))

    head._align = orig_align

    if len(preds) < 2:
        raise SystemExit(
            f"need >= 2 captured (pred, target) pairs for a cross-image probe, got "
            f"{len(preds)}; the head._align spy captured no/too-few pairs. The "
            f"'relational' method never calls _align so this probe does not apply to "
            f"it; otherwise check --images-dir / VMCBench and --n-images."
        )
    P = torch.stack(preds)  # (N, d) pooled+normalized student
    T = torch.stack(targs)  # (N, d) pooled+normalized teacher
    m = discrimination_metrics(P, T)  # Q=pred, K=target
    N = m["n_images"]

    res = {
        "label": args.label or Path(args.ckpt).name,
        "ckpt": args.ckpt,
        "method": method,
        "n_images": N,
        "chance_top1": m["chance_top1"],
        "self_cos_perpatch": float(np.mean(self_cos_pp)),  # == distill_cos
        "self_pooled": m["self_pooled"],
        "cross_pooled": m["cross_pooled"],
        "self_minus_cross": m["self_minus_cross"],
        "retrieval_top1": m["retrieval_top1"],
        "median_rank": m["median_rank"],
        "target_spread_offdiag": m["k_spread_offdiag"],
        "pred_spread_offdiag": m["q_spread_offdiag"],
        "self_centered": m["self_centered"],
        "cross_centered": m["cross_centered"],
        "retrieval_top1_centered": m["retrieval_top1_centered"],
    }

    # Caption eyeball on 3 qual images. The plain template feeds an image-only
    # prompt; seed_bos anchors generate on pad/eos (Qwen3 bos is None).
    model.eval()
    seed_bos(model, processor)
    caps: dict[str, str] = {}
    for f in sorted(Path(args.qual_dir).glob("*.png"))[:3]:
        try:
            caps[f.name] = generate_response(
                model,
                processor,
                query="<image>\n",
                images=Image.open(f).convert("RGB"),
                max_new_tokens=64,
            )
        except Exception as e:  # noqa: BLE001 — eyeball is best-effort
            caps[f.name] = f"<generate failed: {type(e).__name__}: {e}>"
    res["captions"] = caps

    learning = res["self_minus_cross"] > 0.10 and res["retrieval_top1"] > 3.0 / N
    res["verdict"] = "LEARNING" if learning else "COLLAPSED/WEAK"

    print(json.dumps(res, indent=2))
    print(
        f"\nVERDICT[{method} @ {res['label']}]: {res['verdict']}  "
        f"self={res['self_pooled']:.3f} cross={res['cross_pooled']:.3f} "
        f"gap={res['self_minus_cross']:.3f} "
        f"top1={res['retrieval_top1']:.1%} (chance {m['chance_top1']:.1%})  "
        f"centered-self={res['self_centered']:.3f}"
    )
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"[wrote {args.out}]")


if __name__ == "__main__":
    main()
