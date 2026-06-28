# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.

## Hub export template is GENERATED, not hand-edited

`templates/modeling_vlm.py.j2` is the static, `trust_remote_code` mirror of the dynamically-built classes in `src/vlm/models/modeling_vlm.py`, shipped by `utils/push_to_hub.py`.
It is no longer hand-maintained: it drifted (missing the BREEN learnable-query, visual-FFN-expert, visual-prefix and text→image generation arms, plus `torch.arrange` typos) and exported broken models.
It is now produced by a mechanical AST transform in `src/vlm/utils/export_template.py`, which emits the two `create_dynamic_*_class` factories' inner methods as real class bodies (the `type(name, bases, {...})` assembly dict gives the exact method set + order), copies every other module-level helper verbatim, rewrites `super(self.__class__, self)` → `super()` and the `pretrain_class` closure → the static `VLM` name, drops the `@override` hint, and leaves `{{ parent_class }}` / `{{ causal_parent_class }}` placeholders for push_to_hub to fill.
The one hand-maintained piece is the import block in `export_template._HEADER`; `build_modeling_template` statically asserts every global name the emitted bodies reference is bound (imported by `_HEADER`, an emitted helper/class, or a builtin), so a new top-level import in `modeling_vlm.py` that `_HEADER` lacks fails generation (and the in-sync test) instead of shipping a `NameError` — add the matching import to `_HEADER` when this fires.
After ANY change to `modeling_vlm.py`, regenerate with `uv run python -m vlm.utils.export_template` and commit the updated template.
`tests/test_inference.py::test_export_template_in_sync_with_live_model` pins the committed template to a fresh generation, so a missed regenerate fails CI instead of silently shipping a broken hub artifact; companion tests assert state_dict + numerical parity per arm.
`push_to_hub._copy_from_models` must ship every sibling module the rendered file imports (`xmodal_mask`, `gen_diffusion`, `gen_image`, `gen_rope`, `visual_distill`, `gen_perceptual`); add to that list whenever the model grows a new import.

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
    (`config/config_schema.py:LanguageModelConfig`). The data path injects
    `<query>` placeholders (`inject_query_placeholders`, `data/dataset.py`):
    placement `after_image` (pretrain: `<image><query>`, one query block PER image)
    or `after_text` (SFT: ONE query block per sample, appended to the first
    image-bearing human turn after the question). `preprocess_qwen` additionally
    dedups multiple query tokens to the first (BREEN's `eve/train.py:524-531`,
    gated on `learnable_query_enabled`) with a `log.warning`, so a multi-image SFT
    sample keeps a single query block and distills only the first image's queries.
    `tokenizer_multimodal_token` / `_media_pattern` /
    `preprocess_qwen` / `preprocess_plain` all recognize `<query>` → -202 when
    `learnable_query_enabled`. Inference injects it too (`inference/eval.py`).
- **Splice + tagging** — `prepare_inputs_labels_for_multimodal` registers the
    query Parameter as a modality (one block per `<query>`), so the existing splice
    inserts + **label-masks** (excluded from CE) the queries for free, and emits a
    new parallel `query_block_ids` (8th return value; image_block_ids stays the
    7th). A query block's id equals its image's index into `distill_images` because
    the query cursor walks in lockstep with the image cursor; query-free rows
    consume a dummy of each. Lockstep holds when each sample carries one query per
    image: pretrain (`after_image`) injects one `<query>` per `<image>`, and SFT
    data is single-image-per-sample with one query block (the dedup guarantees at
    most one) — so multi-image SFT is out of scope (its extra images stay
    undistilled and would desync the cursor; keep SFT single-image).
- **Routing to the image expert** — the visual-expert mask is
    `(image_block_ids>=0) | (query_block_ids>=0)`, so image AND query tokens go
    through `mlp.mlp_visual` (forward + generate prefill `_ve_gen_mask`).
- **Per-expert sigmoid gate** — `model.visual_expert.gate=true` adds
    `expert_gate_text`/`expert_gate_visual` Linears (`F.sigmoid(gate(x))*expert(x)`);
    `init_visual_expert_gates` inits them near-identity (zero weight, bias 4 →
    sigmoid(4)≈0.982) fresh-build only — near-identity, NOT a literal t=0 no-op: an
    enabled gate attenuates the text FFN by ~1.8%/layer from step 0 (raise the bias
    to ~6-8 for a closer-to-identity start). NOTE: `init_visual_experts_from_text` must
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
1. **`configure_optimizers` SILENTLY DROPS any param group not in its
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
1. **CLIP-L/14-336 teacher is bf16-unstable.** Its ViT attention overflows to
    NaN in bf16, poisoning the distill target. `compute_breen_distill_loss` runs
    the (frozen, detached) teacher in fp32. (The smaller CLIP-B used by the eve/
    repa arms is bf16-stable, so the shared `compute_distill_loss` path is left
    bf16.)
1. **`init_visual_experts_from_text` must exclude `expert_gate*` keys** when
    copying the text FFN into `mlp_visual` (the gateless expert has no such keys
    → load_state_dict raises on unexpected keys).

## Energon train-loader layouts (`build_energon_train_loader`)

Two mutually-exclusive layouts, selected by `DatasetConfig` (set exactly one of
`dataset.wds_path` / `dataset.folders`):

- **jsonl-loose** (`dataset.folders`): one `train.jsonl` per blob folder + loose
    media files; `cook_mm_chat` fetches each image with one Azure GET via
    `media_root.get(path)`. First use auto-downloads each jsonl + builds its index
    locally; media always streams lazily.
- **prepared CrudeWebdataset** (`dataset.wds_path`, `vlm/data/energon_wds.py`):
    the output of `energon prepare` — `{00000..NNNNN}.tar` shards with image bytes
    bundled IN the tar + a `.nv-meta/` dir. `cook_mm_chat_wds` reads image bytes
    from the in-tar sample fields (match by json-path basename, else positional
    two-pass fallback over sorted image-ext fields; it fails loud on an in-tar
    audio item — prepared-WDS audio is unsupported), and `get_train_dataset` reads
    the `.nv-meta` dir directly — no jsonl download / index / metadataset. One
    sequential GET streams ~10k samples, so no per-image round-trip and far fewer
    fat-tail stragglers (the cold shuffle-buffer fill itself remains — see sharp
    edge 1). Example config:
    `config/dataset/energon-bee-stage2-wds.yaml` →
    `msc://azure/data/yiming/bee_stage2/train-wds`. The WDS task encoders subclass
    the jsonl ones and override ONLY `cookers`, so
    encode/collate/bucketing/BREEN-query/savable-resume are shared verbatim.

### Sharp edges (keep these)

1. **The "PicklingError" is a PHANTOM — never re-add a `num_workers=0` guard.** It
    is an energon watchdog all-thread stack dump fired when the cold Azure
    shuffle-buffer fill exceeds the 60 s watchdog default; fork never pickles, so
    there is no real pickle error (`data/datapipe-rootcause-m6`). The loader raises
    `watchdog_initial_timeout_seconds=600` to cover the cold fill (measured ~111 s
    for bee_stage2/train-wds) while keeping the 60 s steady-state watchdog.
    `num_workers=0` ~halves throughput (~44 %/step data-wait); use `num_workers`
    8–12.
1. **`dataset.num_workers` (energon's) owns loading + rank sharding;**
    `trainer.dataloader_num_workers` MUST stay 0 for `type='energon'`.
1. **`conversation_kind` gates template/data validation
    (`validate_dataset_config`).** Mark multi-turn datasets `instruct` so the
    2-turn `plain` caption template is rejected (it drops all human text except
    media placeholders); 2-turn caption data is `caption`; `auto` (default) skips
    the check.
1. **The collator truncation guard is crash-loop-safe.** When right-truncation at
    `model_max_length` would drop a media sentinel, the collator drops that
    instance's orphaned trailing image/audio feature(s) and warns (keeping
    sentinel↔feature counts aligned) rather than raising inside energon's savable
    pipeline — a raise there can deterministically crash-loop on resume.
1. **Local GPU smoke needs `module load cuda/<ver>`** even with
    `trainer.deepspeed=null`: accelerate's `extract_model_from_parallel`
    unconditionally `from deepspeed import DeepSpeedEngine`, whose op-builder
    probes `nvcc` at import → `FileNotFoundError: .../bin/nvcc` at trainer
    construction without it.

## Cross-modal 4D mask correctness (xmodal_mask.py / install_xmodal_masks)

The `prefix_lm` / `img2q_window` arms derive a per-row prefix from the labels (`_prefix`): non-pad positions before the first supervised label.
Two sharp edges make that boundary wrong unless explicitly corrected, and both degrade silently to "looks like plain causal" rather than crashing.

- **Chat-delimiter unmasking collapses the prefix (THE cross-cutting trap).**
  Some conversation preprocessors globally unmask their structural delimiters into the labels, *including* the leading delimiter at position 0: `preprocess_qwen` unmasks the ChatML delimiters (`<|im_start|>` / `<|im_end|>` / newline), and the llama3 template unmasks `<|begin_of_text|>` / `<|start_header_id|>` / `<|end_header_id|>` / `<|eot_id|>` / `\n\n`.
  Naive "first supervised label" then lands at position 0, the prefix is empty, and the cross-modal edges vanish — the arm becomes a no-op you won't notice without inspecting the mask.
  Fix: `_prefix`/`build_cross_modal_mask` take `prefix_skip_ids` (the delimiter token ids) and exclude them from the boundary search so it falls on the first real answer token.
  `vlm.py` (`_cross_modal_prefix_skip_ids`) computes that skip set at load time for the `qwen` and `llama_v3` conversation templates — keyed off the ACTIVE template (resolved the same way the preprocess dispatch is, not the raw `version` string), since other templates (e.g. gemma) supervise answer tokens only — and stows it on `config.cross_modal_prefix_skip_ids`; `install_xmodal_masks` reads it.
  Any new conversation template that unmasks structural tokens into the labels must extend this skip set, or its prefix will be wrong.

- **BREEN learnable-query rows are not question text.**
  `img2q_window`'s question-text key set is `prefix & ~is_img`, which would wrongly include the BREEN `<query>` rows (they sit in the prefix and are not image tokens).
  `build_cross_modal_mask` takes `query_block_ids` (the 8th splice return, `>=0` at query rows) and excludes those columns; it is threaded through `forward` and `generate`.

- **Generation must not leak mask state, and must not skip the install.**
  `install_xmodal_masks` writes a per-layer `_xmodal_mask` plus one-shot `_xmodal_gen_mask` / `_ve_gen_mask`; `generate()` clears all three in a `finally` so nothing carries into the next call.
  A direct `model.generate(images=...)` caller may omit `attention_mask`, and the install is gated on it being non-None — `generate()` materializes a default all-ones mask before the splice so the install still runs.

## Self-describing config fields must be refreshed on `from_pretrained`

Inference rebuilds prompts and masks from the saved `config.json`, so any field that records *how the model was trained* must agree with the actual training run on BOTH the fresh-build and reload paths.
`vlm()` refreshes these from the composed config after load (beside `image_position`): `learnable_query_placement` (an S2 SFT can flip `after_image`→`after_text` while loading an S1 checkpoint whose config still says `after_image`), the `cross_modal_mask_*` dials, and `cross_modal_prefix_skip_ids`.
A reload that keeps the stale checkpoint value makes inference silently disagree with training; add new branch-agnostic self-describing fields to this refresh block.

## Load-time guards (fail loud, don't crash deep)

- **Generation modules:** enabling generation training on a reloaded understanding-only checkpoint routes any `target_patches` batch into `forward_generation` against `None` gen modules; `require_generation_modules` (`vlm.py`) fails at load.
- **Visual distillation is native/raw-patch only:** an encoder-backed model carries no per-patch `image_position_ids`, so `compute_distill_loss` indexes `None`; `load_model` rejects `visual_distill` on a model with a `vision_model`.

## Misc correctness invariants

- **`ignore_index` must be honored by the loss, not hard-coded.** The splice fills ignored/media labels with `self.config.ignore_index`, so BOTH loss paths (chunked CE and the HF `ForCausalLMLoss`) must drop that same value — a non-default `ignore_index` otherwise trains on padded/media positions.
- **`floating_point_ops` sees two shapes of `image_position_ids`.** The understanding splice passes a list of `(N_i, 2)` tensors; generation batching stacks a single `(B, N, 2)` tensor — test `is not None` / non-empty, never truth-value. BREEN `<query>` sentinels each expand to `num_fine + num_coarse` rows in the count.
- **Auto-resume prefers a checkpoint with optimizer state.** `save_only_model` (legacy pretrain) checkpoints carry no optimizer/scheduler state, so `_checkpoint_is_resumable` (`train.py`) treats them as non-resumable and `_resolve_auto_resume_checkpoint` falls back to the newest OLDER full checkpoint rather than silently restarting the optimizer while advancing the step counter. If no full checkpoint exists the action is backend-aware: plain DDP/single-process resumes WEIGHTS ONLY from the snapshot (optimizer/scheduler/RNG reset, loud warning) to preserve weight progress, while DeepSpeed/FSDP safe-skip auto-resume and train from the loaded base (their resume path needs a `global_step*/`/`optimizer_*/` engine-state dir the snapshot lacks and would otherwise crash with "Can't find a valid checkpoint"). Explicit `resume_from_checkpoint` still opts in.
- **`use_start_end_tokens` unfreezes the embedding modules directly.** `group_params_by_prefix` never creates an `"embeddings"` group, so the old `grouped_params.get("embeddings", [])` was a silent no-op; the new image start/end rows must train even with a frozen LM trunk.
