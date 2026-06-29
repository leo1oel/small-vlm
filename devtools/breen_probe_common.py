"""Shared helpers for the S1 representation-eval probes (10-experiment ablation).

The cross-image discrimination metric (the killer test from
`data/breen-port-h7/3arm-distill-ablation.md` / `distillchk-d4/report.md`):
`distill_cos` alone is a MIRAGE — a collapsed constant satisfies a per-row
cosine. What matters is whether per-image descriptors carry real PER-IMAGE
structure or collapsed onto a shared constant. Both probes pool each image's
per-patch features into a single L2-normalized descriptor and ask the same
question via `discrimination_metrics`, differing only in WHERE the features come
from:

  * `breen_probe_xshape.py` (distill `_align` arms 4,5,7,8 AND the single-pool
    breen query arm 9): the distill head's aligned (student pred, teacher target)
    pair — needs `model.visual_distill_head`. For breen it injects a <query>
    sentinel and captures the (query_hidden, projected debiased CLIP target) pair.
  * `breen_probe_feat.py` (non-distill arms 1,2,3,6,10): the RAW LLM hidden state
    at image positions, split into two interleaved patch halves — runs on ANY
    native (encoder-free) checkpoint.

`discrimination_metrics(Q, K)` takes two (N, d) sets of paired per-image
descriptors (row i = image i's two views) and returns the retrieval / self-vs-
cross / centered metrics both reports use.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _pool(x: torch.Tensor) -> torch.Tensor:
    """(m, d) per-patch -> (d,) L2-normalized image descriptor (mean over patches)."""
    return F.normalize(x.float().mean(0), dim=-1)


def _offdiag_mean(M: torch.Tensor) -> float:
    n = M.shape[0]
    if n < 2:
        return float("nan")
    return float(M[~torch.eye(n, dtype=torch.bool, device=M.device)].mean())


def load_probe_images(
    n_images: int, images_dir: str | None = None, split: str = "dev"
) -> list[Image.Image]:
    """Diverse image set for the cross-image probe.

    `images_dir` (sorted *.png) is the offline-friendly default — pass the
    caption-retest qual set (`.../breen-s2val-j8/qual_images`) for an
    apples-to-apples cross-arm comparison on an identical image set. Otherwise
    pull `n_images` evenly-spaced VMCBench-dev images, which needs the dataset
    cached (a compute node is usually offline — see the error below).
    """
    if images_dir:
        paths = sorted(Path(images_dir).glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"--images-dir {images_dir} contains no *.png files")
        if len(paths) > n_images:
            idx = np.linspace(0, len(paths) - 1, n_images).round().astype(int).tolist()
            paths = [paths[int(i)] for i in idx]
        return [Image.open(p).convert("RGB") for p in paths]
    try:
        from datasets import load_dataset

        ds = load_dataset("suyc21/VMCBench", split=split)
    except Exception as e:  # noqa: BLE001 — turn an opaque offline failure into guidance
        raise RuntimeError(
            "could not load VMCBench (suyc21/VMCBench); a compute node is usually "
            "offline. Either pre-cache it once on a login node "
            '(`uv run python -c "from datasets import load_dataset; '
            "load_dataset('suyc21/VMCBench', split='dev')\"`) or pass --images-dir "
            "with local PNGs (e.g. .../breen-s2val-j8/qual_images)."
        ) from e
    col = "image" if "image" in ds.column_names else ds.column_names[0]
    idx = np.linspace(0, len(ds) - 1, n_images).round().astype(int).tolist()
    return [ds[int(i)][col].convert("RGB") for i in idx]


def seed_bos(model: object, processor: object) -> None:
    """The plain template feeds an image-only prompt (no text), so input_ids is
    empty after the image splice; Qwen3's bos_token_id is None -> generate can't
    seed. Anchor on pad/eos (the image embeds dominate the prefix; the anchor is
    inert). No-op when a bos is already set."""
    gc = model.generation_config  # type: ignore[attr-defined]
    if getattr(gc, "bos_token_id", None) is None:
        tok = processor.tokenizer  # type: ignore[attr-defined]
        gc.bos_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def apply_query_placement_override(model: object, placement: str | None) -> None:
    """Defensive eval-side override for the learnable-query placement.

    Main-trained checkpoints ALREADY serialize the correct value — `vlm()`
    refreshes `learnable_query_placement` from the composed config at train time,
    and (ST-3 Part D, eval-config-fix) `load_model` now ALSO re-applies it from
    the run config on the `from_pretrained` reload path that eval uses, so eval
    self-describes correctly even if the checkpoint config.json is stale. So this
    is a no-op for any arm-9 checkpoint evaluated via the composed run config. Use
    it ONLY to force a placement for a stale/external query checkpoint loaded
    WITHOUT its run config (the symptom the validation report flagged: a checkpoint
    that trained `after_text` serializing the base's `after_image`).
    """
    if placement is not None:
        model.config.learnable_query_placement = str(placement)  # type: ignore[attr-defined]


def discrimination_metrics(Q: torch.Tensor, K: torch.Tensor) -> dict[str, float]:
    """Cross-image retrieval / centering metrics for paired per-image descriptors.

    Q, K: (N, d) L2-normalized pooled descriptors; row i = image i's two views
    (xshape: student pred vs teacher target; feat: patch split-half A vs B).

      self_pooled   = mean_i cos(Q_i, K_i)            (within-image agreement)
      cross_pooled  = mean_{i!=j} cos(Q_i, K_j)       (across-image leak)
      retrieval_top1= frac_i [ argmax_j cos(Q_i, K_j) == i ]
      *_centered    = same after subtracting the across-image MEAN (the constant)
                      — distillchk's clincher: a collapsed constant scores high
                      self_pooled but ~0 self_centered.
      q/k_spread    = mean off-diagonal cos among the Q / K descriptors (the
                      collapse detector: ~1.0 => all images look identical).

    LEARNING: self_pooled >> cross_pooled, retrieval >> chance, self_centered
    stays high.  COLLAPSED: self ~= cross, retrieval ~= chance, centered ~= 0.
    """
    N = Q.shape[0]
    S = Q @ K.t()  # cos(Q_i, K_j)
    self_pooled = float(S.diag().mean())
    cross_pooled = _offdiag_mean(S)
    top1 = float((S.argmax(1) == torch.arange(N, device=S.device)).float().mean())
    ranks = (S >= S.diag().unsqueeze(1)).sum(1).float()  # 1 = own match ranked best
    med_rank = float(ranks.median())

    # Centering clincher: remove the across-image constant, re-test.
    Qc = F.normalize(Q - Q.mean(0, keepdim=True), dim=-1)
    Kc = F.normalize(K - K.mean(0, keepdim=True), dim=-1)
    Sc = Qc @ Kc.t()

    return {
        "n_images": N,
        "chance_top1": 1.0 / N,
        "self_pooled": self_pooled,
        "cross_pooled": cross_pooled,
        "self_minus_cross": self_pooled - cross_pooled,
        "retrieval_top1": top1,
        "median_rank": med_rank,
        "q_spread_offdiag": _offdiag_mean(Q @ Q.t()),
        "k_spread_offdiag": _offdiag_mean(K @ K.t()),
        "self_centered": float(Sc.diag().mean()),
        "cross_centered": _offdiag_mean(Sc),
        "retrieval_top1_centered": float(
            (Sc.argmax(1) == torch.arange(N, device=Sc.device)).float().mean()
        ),
    }
