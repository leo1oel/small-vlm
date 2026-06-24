# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.

## BREEN port (encoder-free CLIP-distilled learnable queries; arXiv:2503.12446)

Spec: the `breen-plan-q9` report (code-grounded port plan). BREEN = learnable
queries distilled to a frozen CLIP teacher + per-layer image-expert FFN +
frozen-LLM warmup. The port reuses the existing `VisualDistillTeacher`,
`visual_expert`/`mlp_visual`, splice/label-mask machinery, and staging; it adds
the BREEN *interface*. All BREEN behavior is behind new flags — disabled by
default, so encoder-based and native paths are bit-identical.

### What is wired, and where
- **Learnable queries** — `nn.Parameter(num_fine+num_coarse, hidden)` on the
  ForCausalLM (`model.learnable_query.{enabled,num_fine,num_coarse,placement}`).
  Built in `_build_learnable_query` (`models/modeling_vlm.py`); randn init
  survives `post_init` (bare Parameter is not an `nn.Module` `_init_weights`
  touches). It IS in the checkpoint state_dict (trained → serialized).
- **Query placeholder** — `<query>` / `query_token_index=-202`
  (`config/config_schema.py:LanguageModelConfig`). The data path injects one
  `<query>` per image (`inject_query_placeholders`, `data/dataset.py`): placement
  `after_image` (pretrain: `<image><query>`) or `after_text` (SFT: query after
  the question). `tokenizer_multimodal_token` / `_media_pattern` /
  `preprocess_qwen` / `preprocess_plain` all recognize `<query>` → -202 when
  `learnable_query_enabled`. Inference injects it too (`inference/eval.py`).
- **Splice + tagging** — `prepare_inputs_labels_for_multimodal` registers the
  query Parameter as a modality (one block per `<query>`), so the existing splice
  inserts + **label-masks** (excluded from CE) the queries for free, and emits a
  new parallel `query_block_ids` (8th return value; image_block_ids stays the
  7th). A query block's id equals its image's index into `distill_images` because
  the query cursor walks in lockstep with the image cursor (one `<query>` per
  `<image>`; query-free rows consume a dummy of each).
- **Routing to the image expert** — the visual-expert mask is
  `(image_block_ids>=0) | (query_block_ids>=0)`, so image AND query tokens go
  through `mlp.mlp_visual` (forward + generate prefill `_ve_gen_mask`).
- **Per-expert sigmoid gate** — `model.visual_expert.gate=true` adds
  `expert_gate_text`/`expert_gate_visual` Linears (`F.sigmoid(gate(x))*expert(x)`);
  `init_visual_expert_gates` inits them near-identity (zero weight, bias 4 →
  sigmoid≈1) fresh-build only. NOTE: `init_visual_experts_from_text` must
  exclude `expert_gate*` keys when copying the text FFN into `mlp_visual`.
- **breen distill method** (`models/visual_distill.py`, `method="breen"`) —
  the head carries a `norm_layer = LayerNorm(1024)+Linear(1024→hidden,bias=False)`
  that projects the **CLIP target UP to LLM-hidden** (faithful direction; cosine
  in LLM space). `compute_breen_distill_loss` (`models/modeling_vlm.py`) gathers
  the LLM final post-norm hidden at query positions, splits the first `num_fine`
  / last `num_coarse` rows, and aligns them to `avg_pool2d(grid,3,3)`=8×8 and
  `avg_pool2d(grid,4,4)`=6×6 of the CLIP grid. Requires `teacher_out_size=336`
  with `clip-vit-large-patch14-336` (24×24 grid → 64+36); a mismatch raises.
  Loss = `CE + visual_distill_weight*(L_fine+L_coarse)`. Runs only on the
  chunked-CE path → **`loss_chunk_size>0` is required**.
- **Teacher** — `teacher_out_size` is threaded through `attach_distill_teacher`
  (`vlm.py`) → `VisualDistillTeacher`. The teacher is training-only and off the
  module tree (not in state_dict / not needed at inference).

### Param-grouping trap (the most likely silent failure — DO NOT regress)
The queries, `visual_distill_head` (incl. `norm_layer`), `mlp_visual`, and expert
gates default into the **`language_model`** optimizer group. A frozen-LLM S0
(`train_language_model:false`) would therefore freeze the very modules S0 must
train. Two coupled fixes (`train/set_trainable.py` + `train/optimizer.py`):
1. `set_trainable_params` force-enables `learnable_query` / `visual_distill_head`
   (dedicated prefix groups) and any `.mlp_visual.` / `expert_gate` param, always,
   regardless of the LM freeze flag (the visual-aux/generation pattern).
   `apply_delta_tuning`'s keep-list includes them too.
2. **`configure_optimizers` SILENTLY DROPS any param group not in its
   `component_to_config` map** — so `learnable_query` and `visual_distill_head`
   are mapped there (→ `connector` lr/wd: BREEN's higher "proj" LR in S0;
   identical to the LM lr in the single-LR SFT configs, so no eve/repa
   regression). `mlp_visual`/gates stay in the `language_model→model` bucket
   (LM lr). Forgetting either fix = the module is trainable but never stepped.

### Configs (S0 → S1 → S2 chain)
Shared architecture: `config/model/qwen3-1.7b-unified-breen.yaml` (visual_distill
breen + CLIP-L-336, visual_expert+gate, learnable_query 64/36). Stages flip only
placement / unfreeze / LR / dataset:
- `pretrain-unified-breen.yaml` — S0 frozen-LLM caption alignment, after_image,
  LR 4e-4(connector)/4e-5(LM), `loss_chunk_size:1024`.
- `pretrain-unified-breen-s1.yaml` — S1 full unfreeze, from `VLM_BREEN_S0_CKPT`, LR 4e-5.
- `sft-unified-breen.yaml` — S2 Honey SFT, after_text, from `VLM_BREEN_S1_CKPT`, LR 2e-5.

Caption data: `config/dataset/energon-breen-caption.yaml` — the format-generic
energon understanding loader; the captain's ~14M caption set drops in by changing
the single `folders` key (validated default = existing `Bee-Training-Data-Stage1`).
SFT data: `config/dataset/energon-honey-sft.yaml` → `yiming/Honey-Data-1M`.

### Phase-0 smoke (port correctness)
`devtools/breen_smoke.py` (+ `breen_smoke.slurm`, run from the worktree so it
exercises worktree code — the `.venv` is symlinked to the main checkout, so set
`PYTHONPATH=<worktree>/src`). Heavy model loads MUST run via SLURM (the login
node kills them / OOMs). It asserts: (a) `distill_cos` rises from ~0; (b) query
positions are label-masked and routed to `mlp_visual`; (c) teacher absent from
named_parameters/state_dict and a checkpoint loads + forwards with no CLIP
teacher (inference path); (d) frozen-LLM config trains query/norm_layer/
mlp_visual/gate while the LM trunk stays frozen.

**Smoke result (Qwen3-0.6B + CLIP-L/14-336, L40):** ALL PASS — (a) distill_cos
−0.011 → 1.000, distill loss 2.02 → 0.0008; (b) 200 query rows (100×2)
label-masked + routed; (c) teacher absent, reload forwards with no CLIP; (d)
frozen-LM grads flow to query/norm_layer/mlp_visual while LM trunk + embeddings
stay frozen.

### Numerical traps (found + fixed in Phase 0 — keep these)
1. **Uninitialized bare Parameter → NaN forward (the big one).** A
   `nn.Parameter(...)` created in `__init__` is a MISSING KEY in the base-LM
   `from_pretrained` checkpoint, and the backbone `_init_weights` only inits
   recognized module types (Linear/Embedding/RMSNorm) — it never touches a
   top-level Parameter. Left as `to_empty` garbage it produced NaN logits from
   step 0 (even at plain inference). Fix: `init_learnable_query` materializes +
   randn-inits it (at `initializer_range`) AFTER from_pretrained, fresh-build
   only (reloads carry the trained value). Any future bare Parameter on the
   ForCausalLM needs the same post-load init.
2. **CLIP-L/14-336 teacher is bf16-unstable.** Its ViT attention overflows to
   NaN in bf16, poisoning the distill target. `compute_breen_distill_loss` runs
   the (frozen, detached) teacher in fp32. (The smaller CLIP-B used by the eve/
   repa arms is bf16-stable, so the shared `compute_distill_loss` path is left
   bf16.)
3. **`init_visual_experts_from_text` must exclude `expert_gate*` keys** when
   copying the text FFN into `mlp_visual` (the gateless expert has no such keys
   → load_state_dict raises on unexpected keys).
