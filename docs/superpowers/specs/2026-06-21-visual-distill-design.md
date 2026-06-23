# Visual-encoder distillation for the native VLM (2026-06-21)

## Problem

The native (encoder-free) VLM — raw patches → linear connector → Qwen3 decoder,
no vision tower — left to plain SFT rides the language prior and never learns to
condition on pixels (FDI probe R0 ≈ 0; POPE degenerate "always yes"). The
linear-predictivity probe (`neo_analysis/predictivity_n1000.json`) shows the
model DOES build encoder-grade features internally (peak ImageNet linear-probe
top-1 reaches CLIP-level), but **gradually and late**, discovered only through
next-token loss. Hypothesis: supervise the visual pathway directly by aligning
the LLM's hidden states at image positions to a frozen vision encoder.

This is the distillation pivot the user chose after the analysis phase. Priority
order (user, verbatim): **(1) generalization, (2) simple/reasonable/practical —
practicality FIRST, (3) novelty is a bonus.**

## Mechanism (one config, six methods)

`model.visual_distill.method`:

| method | what it aligns | head | loss | lineage |
|---|---|---|---|---|
| `repa` | one ~0.3-depth layer → CLIP final patch features | MLP | neg-cosine | REPA 2410.06940 (gen→understanding) |
| `eve` | final post-norm hidden → CLIP final | MLP | neg-cosine | EVE 2406.11832 PAL, last-layer |
| `vora` | first-N blocks, each 1:1 to a depth-matched CLIP block | RMSNorm+Linear/block | neg-cosine | VoRA 2503.20680 (no LoRA, CLIP not AIMv2) |
| `softdepth` | **learned softmax over a layer pool selects the depth** | MLP on mixed hidden | neg-cosine | OURS |
| `relational` | token-token Gram matrix → CLIP Gram | none | MSE | OURS |
| `vae` | one layer → frozen VAE latent grid | MLP | smooth-L1 | low-level control |

`repa`/`eve` are published baselines; `vora` is the published block-wise method
(our scheme 3 single-layer is a simplification of it); `softdepth`/`relational`
are the novel contributions; `vae` is the semantic-vs-reconstructive control.

**Why softdepth is the headline.** VoRA hard-codes "the first N blocks ARE the
ViT, in lockstep." Softdepth instead lets the model *self-select* which depth
hosts its internal encoder via a learned softmax (`distill_sel_depth` is logged)
— directly testing the user's thesis that the internal-encoder placement should
be model-decided, not fixed like a ViT.

## Key engineering decisions

- **Teacher sees the same pixels, zero dataloader changes.** The raw patches in
  the batch are a lossless [0,1] RGB image (RawImageProcessor is rescale-only,
  known 48×48×3 row-major layout). `reconstruct_image_from_patches` inverts it
  exactly (verified 0.0 round-trip error), resizes to 224, runs CLIP. No
  `teacher_images` field, no collator change. Same-view distillation.
- **Per-patch spatial correspondence.** CLIP's 14×14 patch grid is bilinearly
  resized to the image's native patch grid and gathered at each patch's (x, y).
  Resize uses the FULL native grid (from all of the image's positions) so a
  tail-truncated block keeps correct correspondence.
- **Teacher is off the module tree.** Stored `model._distill_teacher = [teacher]`
  (list-wrapped) so it is invisible to `.parameters()`/optimizer/`state_dict`.
  This is load-bearing: set_trainable sweeps any unprefixed param into the
  `language_model` group and trains it — a registered teacher would be TRAINED
  and bloat every checkpoint by ~170MB. The small projection head IS a real
  submodule (`visual_distill_head`), trained with the LM (it falls through to
  the language_model optimizer group — every native-SFT distill run trains LM).
- **Loss lives in `chunked_ce_forward`** (training path only, `loss_chunk_size>0`),
  alongside grounding/visual-aux/aux-exit, reusing the same forward-hook layer
  capture and the same rank-symmetric component-stash + anchor pattern (so the
  trainer's cross-rank `sorted(components)` all-reduce never deadlocks).

## Files

- `src/vlm/models/visual_distill.py` — teacher, reconstruction, head + loss math.
- `src/vlm/models/modeling_vlm.py` — `_build_visual_distill_head`, `compute_distill_loss`, forward/chunked wiring.
- `src/vlm/vlm.py` — `attach_distill_teacher`, config flattening, teacher-dim resolution.
- `src/vlm/config/config_schema.py` — `VisualDistillConfig`, `visual_distill_weight`.
- `devtools/distill_smoke.py` (+ `.slurm`) — GPU overfit proof.
- `src/vlm/config/sft-unified-bee-mix-distill-{repa,eve,vora,softdepth,relational,vae}.yaml`.

## Experiment matrix

All six arms share the baseline-mix blend/steps(5000)/seed/token-budget →
identical energon stream, directly comparable to `sft-unified-bee-mix`
(the native no-distill baseline). save_steps 500 → 10 checkpoints for
checkpoint-wise curves and early stopping (eval at 500/1000, stop if flat).

**Final comparison MUST be benchmark scores** (lmms-eval vmcbench/pope/mmvp),
per the user — not just R0. Watch in wandb: `distill` (should fall), `distill_cos`
(should rise), and crucially `ce_final` (must NOT degrade vs baseline; lower
`visual_distill_weight` if it does). `softdepth`: watch `distill_sel_depth`.

Open: a true CLIP-encoder upper bound (`sft-clip`) baseline is still pending and
should be trained alongside for the headline comparison.
