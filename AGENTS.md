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

## Local-JSON data path: media-sentinel ↔ feature invariant (DO NOT regress)

The splice (`prepare_inputs_labels_for_multimodal`, `models/modeling_vlm.py`)
consumes media features from a **global, per-modality cursor** that advances one
step per sentinel across the whole batch (a row with zero sentinels of a modality
consumes one zero-width dummy). So the hard invariant is: **per sample, the number
of `<image>`/`<audio>`/`<query>` sentinels in the tokenized text must equal the
number of features that sample queued for that modality.** Any mismatch doesn't
just drop the local image — it desyncs the cursor and leaks features into the
*next* sample (silent cross-sample corruption). The local-JSON and energon paths
now feed the collator the same media contract:

- `preprocess_multimodal` hoists a lone `<image>` to the front of its turn but
    must **preserve every placeholder for interleaved multi-image turns** (never
    collapse N→1). Local-JSON path only — the energon path normalizes separately.
- `apply_image_position` ("sandwich"/"random" repeat the question text) takes
    `protected_tokens` and **protects non-image media placeholders in place** —
    it pulls them out before building the question and reinserts each exactly once,
    so it can never duplicate `<audio>`/`<query>`. Passed at all three call
    sites — the local and energon data paths and inference
    (`eval.py::generate_response`, #8) — so the trained layout reproduces at
    inference too.
- `DataCollatorForSupervisedDataset` takes `media_token_ids` +
    `media_feature_token_ids` and, when `model_max_length` truncation would drop a
    sentinel whose feature is still queued, **neutralizes** — it realigns that
    sample's per-modality feature lists to the surviving sentinels (with a `WARN`)
    rather than raising, because a raise crash-loops energon's buffer-restore on
    resume. Wired at both `make_supervised_data_module` (local) and the energon
    collator.
- Direct local dataset files load via `_load_local_records` (`.json` array or
    `.jsonl` one-object-per-line; other extensions rejected). JSONL is not just a
    YAML-mixture feature — a bare `dataset.path=foo.jsonl` is supported.

## BREEN port (encoder-free CLIP-distilled learnable queries; arXiv:2503.12446)

Spec: the `breen-plan-q9` report (code-grounded port plan). BREEN = learnable
queries distilled to a frozen CLIP teacher + per-layer image-expert FFN +
frozen-LLM warmup. The port reuses the existing `VisualDistillTeacher`,
`visual_expert`/`mlp_visual`, splice/label-mask machinery, and staging; it adds
the BREEN *interface*. All BREEN behavior is behind new flags — disabled by
default, so encoder-based and native paths are bit-identical.

### What is wired, and where

- **Learnable queries** — `nn.Parameter(num_query, hidden)` on the
    ForCausalLM (`model.learnable_query.{enabled,num_query,placement}`). NOTE:
    `num_query` REPLACED the old two-pool `num_fine`/`num_coarse` pair in ST-3
    (single-pool simplification — see "Single-pool query distill (ST-3)" below);
    `num_query` must be a perfect square. Built in `_build_learnable_query`
    (`models/modeling_vlm.py`); randn init
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
- **Routing to the visual experts** — the visual-expert mask is
    `(image_block_ids>=0) | (query_block_ids>=0)`, so image AND query tokens go
    through EVERY enabled visual sibling (forward + generate prefill `_ve_gen_mask`).
    See "Visual experts" below — `mlp_visual` is one of three (FFN / norm / attention).
- **Per-expert sigmoid gate** — `model.visual_expert.gate=true` adds
    `expert_gate_text`/`expert_gate_visual` Linears (`F.sigmoid(gate(x))*expert(x)`) to
    EVERY enabled expert sublayer (FFN/norm/attention), sized to the sublayer's input
    (`in_features` for a projection — o_proj differs from hidden — else hidden);
    `init_visual_expert_gates` inits them near-identity (zero weight, bias 4 →
    sigmoid(4)≈0.982) fresh-build only — near-identity, NOT a literal t=0 no-op: an
    enabled gate attenuates each sublayer by ~1.8% from step 0 (raise the bias to
    ~6-8 for a closer-to-identity start). NOTE: `init_visual_experts_from_text` must
    exclude `expert_gate*` keys when copying the text weights into the sibling.
- **breen distill method** (`models/visual_distill.py`, `method="breen"`) —
    SINGLE-POOL since ST-3 (was a fine/coarse two-pool; see "Single-pool query
    distill (ST-3)" below). The head carries a `norm_layer = LayerNorm(teacher_dim)+Linear(teacher_dim→hidden,bias=False)` that projects the
    **CLIP target UP to LLM-hidden** (faithful direction; cosine in LLM space).
    `compute_breen_distill_loss` (`models/modeling_vlm.py`) gathers the LLM final
    post-norm hidden at query positions and aligns ALL `num_query` rows to ONE
    `adaptive_avg_pool2d(grid, (√num_query,√num_query))` of the CLIP grid (e.g.
    14×14 CLIP-B/16-224 → 8×8=64). The adaptive pool handles ANY teacher grid side,
    so the old `teacher_out_size=336 / clip-vit-large-patch14-336` 24×24 requirement
    is GONE — CLIP-base teachers now work; only `num_query` must be a perfect square
    (else it raises). Loss = `CE + visual_distill_weight*L_query`. Runs only on the
    chunked-CE path → **`loss_chunk_size>0` is required**.
- **Teacher** — `teacher_out_size` is threaded through `attach_distill_teacher`
    (`vlm.py`) → `VisualDistillTeacher`. The teacher is training-only and off the
    module tree (not in state_dict / not needed at inference).

## Visual experts (FFN / norm / attention; EVEv2 divide-and-conquer)

`model.visual_expert` carries THREE independently-toggleable per-decoder-layer
modality-routed experts under one master `enabled` switch + shared
`layers`/`init_from_text`/`gate` (`config/config_schema.py:VisualExpertConfig`,
flattened in `vlm.py`, installed in `install_visual_experts`,
`models/modeling_vlm.py`):

- **`ffn`** (default True for back-compat: an `enabled:true` config with no
    ffn/norm/attention keys = the historical FFN expert alone) → sibling
    `mlp.mlp_visual` (Mono-InternVL).
- **`norm`** → sibling RMSNorms `input_layernorm.norm_visual` /
    `post_attention_layernorm.norm_visual` (EVEv2 modality norms).
- **`attention`** → sibling projections `self_attn.{q,k,v,o}_proj.proj_visual`
    (EVEv2 modality attention). Only the projection WEIGHTS split per token; RoPE,
    `q_norm`/`k_norm`, the causal/cross-modal mask and the KV-cache are untouched
    (decode tokens take the text path; cached image k/v keep their visual
    projection), so the attention PATTERN is unchanged.

All three share one mechanism. `_install_routed_sibling` attaches the sibling
(`type(mlp)(config)` for the FFN, meta-safe `copy.deepcopy` for norm/projections),
then overrides the sublayer's `forward` with `_routed_expert_forward`
(`text*(1-mask)+visual*mask`, with the optional gate). `_set_visual_mask` stashes
the (B,N,1) image+query mask on every routed sublayer (`_visual_routed_modules`).
`init_visual_experts_from_text` copies the text weights into each sibling
(fresh-build only — step-0 no-op modulo the gate). `_visual_expert_mlps` is kept
as the FFN-only list (devtools/breen_smoke read it). The structural fields
serialize into `config.json`, so reload + inference (incl. the hub-export
`templates/modeling_vlm.py.j2`, regenerated via `python -m vlm.utils.export_template`)
rebuild the same experts (train/infer parity). FLOPs ride `num_parameters` in
`floating_point_ops` automatically. `is_visual_expert_param(name)` (modeling_vlm)
is the single source of truth for "this param is a visual-expert sibling/gate",
shared by `set_trainable.py` + `grad_probe.py`. Smoke config:
`config/model/qwen3-0.6b-unified-experts.yaml` (all three on).

### Param-grouping trap (the most likely silent failure — DO NOT regress)

The queries, `visual_distill_head` (incl. `norm_layer`), the visual experts
(`mlp_visual`/`norm_visual`/`proj_visual`) and their gates default into the
**`language_model`** optimizer group. A frozen-LLM S0 (`train_language_model:false`)
would therefore freeze the very modules S0 must train. Two coupled fixes
(`train/set_trainable.py` + `train/optimizer.py`):

1. `set_trainable_params` force-enables `learnable_query` / `visual_distill_head`
    (dedicated prefix groups) and any `is_visual_expert_param` (`.mlp_visual.` /
    `.norm_visual.` / `.proj_visual.` / `expert_gate`) param, always, regardless of
    the LM freeze flag (the visual-aux/generation pattern). `apply_delta_tuning`'s
    keep-list includes them too (the visual attention expert `proj_visual` stays
    trainable even under `DELTA_TUNING=2`, which freezes the SHARED attention).
1. **`configure_optimizers` SILENTLY DROPS any param group not in its
    `component_to_config` map** — so `learnable_query` and `visual_distill_head`
    are mapped there (→ `connector` lr/wd: BREEN's higher "proj" LR in S0;
    identical to the LM lr in the single-LR SFT configs, so no eve/repa
    regression). The visual experts (`mlp_visual`/`norm_visual`/`proj_visual`)
    and gates stay in the `language_model→model` bucket (LM lr). Forgetting either
    fix = the module is trainable but never stepped.

### Configs (S0 → S1 → S2 chain)

Shared architecture: `config/model/qwen3-1.7b-unified-breen.yaml` (visual_distill
breen + CLIP-L-336, visual_expert+gate, `learnable_query.num_query: 100` —
single-pool since ST-3, was the 64+36 two-pool). Stages flip only
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
    copying the text weights into the sibling (the gateless sibling has no such
    keys → load_state_dict raises on unexpected keys).

## Encoder-ablation 2-stage native recipe + 10-experiment ablation arms

The shared base for the native (encoder-free, raw-patch) ablation family is a
2-stage chain on `model/qwen3-1.7b-unified` — **not** the BREEN model above:

- **S1 `exp-encabl-native-s1.yaml`** — caption pretrain on
    `dataset/energon-bee-stage2-wds`, `trainer: pretrain` (connector-only
    unfreeze), `batch_token_budget: 24000`, `max_steps: 34722`, LR 8e-4. Experts
    / aux / distill all OFF → this IS the no-method control.
- **S2 `exp-encabl-native-s2.yaml`** — instruction SFT on
    `dataset/energon-honey-sft-wds` (new WDS twin of bee-stage2; `wds_path`,
    `conversation_kind: instruct`, `strip_empty_think`), `trainer: finetune`
    (LM+connector unfreeze), `version: qwen_2_5` ChatML, `batch_token_budget: 14000`, `max_steps: 11905`, LR 6e-5, `from_pretrained: ${oc.env:VLM_S1_CKPT}`.

The **G1 encoder twin** is `exp-encabl-clip-s{1,2}.yaml` on `model/qwen3-1.7b-clip`
(frozen CLIP-ViT-B/16 tower, 196 img tokens) — the canonical-encoder comparison
arm for the encoder-free native (G2) family above. It is byte-for-byte symmetric
with native-s1/s2 (SAME dataset stream, `batch_token_budget`, `max_steps`, LR,
template) for **equal-token fairness**; the ONLY intended difference is the visual
pathway (CLIP tower vs raw-patch connector), which is exactly the variable under
test. CLIP stays frozen throughout BOTH stages (pretrain = connector-only;
finetune = LM+connector, tower still frozen). The captain chose the 1.7B backbone,
so the `qwen3-0.6b-clip` model config is dead — do not use it.

**Effective-token budget convention (hold this constant across every arm):**
`max_steps = effective_tokens / (batch_token_budget × world_size)`, world_size=6.
S1 targets **5e9** effective tokens → 5e9/(24000·6) = 34722. S2 targets **1e9** →
1e9/(14000·6) = 11905. Budgets (24000 S1 / 14000 S2) are OOM-verified on A100-80GB
(32000/20000 OOM); a 48G launch re-verifies per-config (smaller budget +
grad-accum to preserve effective tokens). `energon-bee-stage2-wds.yaml`'s
`batch_token_budget` is the S1 24000 default (no run config but this recipe uses
it; it was 32000 pre-recipe).

**Arm configs inherit the base via Hydra `defaults` + add only toggle keys**
(`defaults: [exp-encabl-native-s{1,2}, _self_]`, then `model:`/`trainer:`
overrides + a distinctive `trainer.run_name`) — so budgets/steps/seed/data stream
are identical and arms stay directly comparable. Toggle → flattened `VLMConfig`
(set in `vlm.py` load_model, read by `getattr` in `modeling_vlm.py`):

| Arm                                                    | `model.*` toggles                                                                                                                                                                                                                                                                                                                                                 | flattened keys                         |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| Exp 1 `exp-encabl-e1-nepa`                             | `visual_aux.objective: nepa` (+ `trainer.visual_aux_weight: 0.5`)                                                                                                                                                                                                                                                                                                 | `visual_aux_objective`                 |
| Exp 2 `exp-encabl-e2-vexpert`                          | `visual_expert: {enabled, ffn}`                                                                                                                                                                                                                                                                                                                                   | `visual_expert`, `visual_expert_ffn`   |
| Exp 3 `exp-encabl-e3-vexpert-norm`                     | `visual_expert: {enabled, ffn, norm}`                                                                                                                                                                                                                                                                                                                             | `+ visual_expert_norm`                 |
| Exp 6 `exp-encabl-e6-vexpert-norm-nepa`                | Exp 3 toggles + `visual_aux.objective: nepa` (+ weight 0.5)                                                                                                                                                                                                                                                                                                       | union of above                         |
| Exp 4 `exp-encabl-e4-vexpert-distill-eve`              | `visual_expert: {enabled, ffn}` + `visual_distill: {enabled, method: eve, teacher_kind: clip, teacher_name: openai/clip-vit-base-patch16, debias_target: true, debias_momentum: 0.9, rkd_dist_weight: 1.0}` (+ `trainer.visual_distill_weight: 1.0`)                                                                                                              | `visual_distill*` + `visual_expert*`   |
| Exp 5 `exp-encabl-e5-vexpert-norm-distill-eve`         | Exp 4 + `visual_expert.norm: true`                                                                                                                                                                                                                                                                                                                                | `+ visual_expert_norm`                 |
| Exp 7 `exp-encabl-e7-vexpert-norm-distill-repa`        | Exp 5 but `visual_distill.method: repa`, `layers: [8]` (mid)                                                                                                                                                                                                                                                                                                      | `visual_distill_method/layers`         |
| Exp 8 `exp-encabl-e8-vexpert-norm-distill-softdepth`   | Exp 5 but `method: softdepth`, `layers: [4,8,12,16,20,24]`                                                                                                                                                                                                                                                                                                        | `visual_distill_method/layers`         |
| Exp 9 `exp-encabl-e9-vexpert-norm-query-distill-breen` | `visual_expert: {enabled, ffn, norm}` + `learnable_query: {enabled, num_query: 64}` + `visual_distill: {enabled, method: breen, teacher_kind: clip, teacher_name: openai/clip-vit-base-patch16, teacher_out_size: 224, debias_target: true, debias_momentum: 0.9}` (+ `trainer.visual_distill_weight: 1.0`). NO `rkd_dist_weight`/B — breen routes only A debias. | `learnable_query*` + `visual_distill*` |
| Exp 10 `exp-encabl-e10-captiondrop`                    | `caption_token_dropout: {enabled, p_start: 0.10, p_end: 0.30}` (**S1 only**; the `-s2` pair inherits native-s2 unchanged — dropout default OFF)                                                                                                                                                                                                                   | `caption_token_dropout_*`              |

Each arm is a `-s1`/`-s2` pair. `visual_expert.ffn` defaults True, `norm`/
`attention` default False, `visual_aux.objective` defaults `none`, so the listed
keys are the full minimal delta. Verify any new arm with
`uv run python -m vlm -cn <config> --cfg job` (a misspelled toggle fails against
the structured schema at compose, not at train launch). Exps 4/5/7/8 (distill)
add the anti-collapse port (ST-2, now landed — see below); exp 9 (query distill)
landed in ST-3 (single-pool breen + debias-on-query — see "Single-pool query
distill (ST-3)"); exp 10 (caption-token dropout) now landed — see "S1 caption-token
(input word) dropout" below. The `data/tenexp-audit/report.md` audit mis-scoped exp
10 as `attention_dropout` (a 1-line decoder dial); the captain re-specced it as
input-side caption-token dropout (the mechanism actually documented below).

### Anti-collapse distill port (ST-2) — `visual_distill` trick A + B

Distillation alone makes every image's per-patch CLIP target collapse onto a
shared "mean-image" constant (cross-image cosine ~0.98, retrieval at chance), so
the LM ignores the visual pathway. The anti-collapse recipe (ported from
`fm/breen-exp` commit `633c89c` as a **clean port, NOT a branch merge** — that
branch forked pre-#20 and a merge regresses the norm/attention experts) breaks
the collapse. It lives entirely behind `VisualDistillConfig`'s 16 new dials, all
**default OFF**, so `eve`/`repa`/`softdepth`/`vae` stay byte-identical to the
clean core until a dial is set:

- **Trick A — `debias_target`** (`debias_momentum`, `debias_std`): subtract a
    running **EMA per-channel mean** of the teacher target before the cosine —
    removes the shared constant. This is the lever that breaks collapse.
- **Trick B — `rkd_dist_weight`**: a **bounded Gram relational** term over
    per-image pooled features (cosine-matrix MSE, in [-1,1] → finite gradients,
    unlike raw RKD distance/angle whose `1/‖edge‖` blows up the untrained
    connector). `rkd_angle_weight`/`_rkd_distance` are dead reference code.
- **Trick C — VICReg** (`vicreg_var_weight`/`vicreg_cov_weight`) **HURTS**
    (over-regularizes, retrieval back to ~chance) → **shipped but default 0; the
    distill arms keep it 0.** SIGReg/PHI-S/MGD are optional extras, also default 0.
- The recipe routes through `_compute_anticollapse` via a dispatch at the top of
    `compute()` that fires **only** when `_anticollapse_on()` (any dial on) AND
    `method ∈ {eve, repa, softdepth, vae}` (`breen`/`vora` bypass that dispatch).
    NOTE (ST-3): `breen` still bypasses `_compute_anticollapse`, but the **A debias
    is now wired into the breen query path separately** — `compute_breen` calls the
    shared `_apply_debias` helper (extracted from `_compute_anticollapse`) on the
    CLIP target before `norm_layer`. B/RKD and C/VICReg are NOT routed for breen.
- **Dial recipe the distill arms use = A+B:** `debias_target: true`,
    `debias_momentum: 0.9`, `rkd_dist_weight: 1.0`, everything else default.

**The `to_empty` buffer trap (why "identical loss code died at random").** The
debias EMA buffers (`debias_mean`/`debias_var`/`debias_inited`, registered only
when `debias_target`) are MISSING KEYS in the base-LM checkpoint; a fresh
`from_pretrained` build leaves them as `to_empty` garbage (`_init_weights` never
touches registered buffers). A garbage-truthy `debias_inited` makes the EMA
subtract an uninitialized (often Inf) mean → NaN cosine → **every microbatch
skipped, non-deterministically per run**. Fix = `init_visual_distill_buffers()`
(mirrors `init_learnable_query`), called from `vlm.py`'s **fresh-build branch
only** (the `else:` that builds from the base LM) — so **S2 `from_pretrained: ${VLM_S1_CKPT}` (the reload branch at the top of `load_model`) does NOT reset the
trained EMA**; it carries it (buffers are `persistent=True`). Defense-in-depth: an
in-loss `isfinite` re-init guard re-warms the EMA if a non-finite mean ever slips
through, plus eps-inside-sqrt RMSNorm in `_align`, `nan_to_num`, and Gram
sanitize/clamp (port these NaN guards together — they are load-bearing).

**Six touch-points (all four files).** ① `visual_distill.py` `_compute_anticollapse`

- helpers; ② `config_schema.py` 16 `VisualDistillConfig` fields; ③
    `modeling_vlm.py` `_build_visual_distill_head` reads them via `getattr` +
    `init_visual_distill_buffers`; ④ `vlm.py` fresh-build init call; ⑤ `vlm.py`
    **flattens the 16 fields onto `VLMConfig`** — miss ⑤ and the `getattr` reads
    silently default OFF (the dials no-op); ⑥ `modeling_vlm.py` `compute_distill_loss`
    **no-image anchor** must emit the SAME weight-gated component-key set as
    `_compute_anticollapse` (single source of truth: `head.anticollapse_keys()`) — miss
    ⑥ and an image-free microbatch yields fewer keys than an image-bearing one, so the
    trainer's cross-rank `all_reduce(stack(sorted(comps)))` size-mismatches and
    NCCL-hangs every multi-GPU run. The warmup counter `_ac_step` is likewise the
    trainer's `global_step` (mirrored once per optimizer step via
    `VLMTrainer._sync_distill_warmup_step`), never a per-forward count, so the ramp is
    rank-identical under gradient accumulation. Regression test:
    `tests/test_visual_distill_anticollapse.py` (gated-off == plain cosine; buffer
    trap + step-0 finite). Teacher for these arms = **CLIP-base**
    (`openai/clip-vit-base-patch16`, default `teacher_out_size: 224`) for a fair
    comparison vs the CLIP-encoder run — captain's call, NOT CLIP-L/14-336.

### Single-pool query distill (ST-3) — exp 9 (`breen` simplified + debias-on-query)

The captain's exp-9 = keep the learnable queries but distill-align CLIP **directly
on the queries with NO two-pooling**. ST-3 is a *removal*, not a port, across four
files:

- **`num_query` REPLACED `num_fine`+`num_coarse`** everywhere (`LearnableQueryConfig`,
    the flattened `learnable_query_num_query` key in `vlm.py`, `DataArguments` +
    `effective_sample_length` bucketing, `VisualDistillHead`, `_build_learnable_query`,
    the FLOPs splice count, `breen_smoke.py`, `qwen3-1.7b-unified-breen.yaml`). It is
    the single query-row count; **must be a perfect square** (8×8=64 for exp 9).
- **Single pool**: `compute_breen_distill_loss` does ONE
    `adaptive_avg_pool2d(grid,(√nq,√nq))` (emits a single `"target"` key), and
    `compute_breen` does ONE `_align(query_hidden, norm_layer(target))` per image
    (was fine_pred/coarse_pred slices + two `_align`s). The old `(side//3)/(side//4)`
    grid assertion (which forced CLIP-L/14-336's 24×24 grid) is GONE → **CLIP-base
    teachers work** (exp 9 uses `clip-vit-base-patch16` @ 224 → 14×14 → 8×8).
- **Debias A on the query path** (captain's call): aligning every image's queries to
    a single near-constant CLIP pool with plain cosine can re-collapse, so
    `compute_breen` runs the same EMA per-channel `_apply_debias` the per-patch path
    uses (on the CLIP target, teacher space, before `norm_layer`). `debias_target: true, debias_momentum: 0.9`; B/C stay off. The `debias_mean/var/inited` buffers +
    `init_visual_distill_buffers` reset apply unchanged (the breen head registers them
    because `debias_target`).
- **Eval-config-fix (Part D)**: `load_model`'s `from_pretrained` **reload** branch now
    re-applies `learnable_query_placement` from the run config (it is non-structural —
    does not change the query Parameter shape). Before, a reloaded query checkpoint
    kept the stale serialized placement and `eval.py:205` (reads it straight off the
    checkpoint config) would inject the query block at the wrong position. Only bites
    the query arm.
- **xshape probe covers arm 9**: `breen_probe_xshape.py` injects a `<query>` sentinel
    (else `compute_breen_distill_loss` anchors out and `_align` is never called), and
    **freezes the debias EMA (`head.eval()`) during capture** so the single-image probe
    forwards don't drift the trained debias mean. The single-pool breen calls `_align`
    once per image, so the existing spy captures it cleanly; the captured pair is in
    LLM-hidden space (query_hidden vs projected debiased target) — `discrimination_metrics`
    is dimension-agnostic.

### S1 caption-token (input word) dropout — `caption_token_dropout` (exp 10)

The language-prior trap: in S1 caption pretrain the model can lower the loss by
predicting the next caption word from the *previous words* and ignoring the
image (blandly "This is an image of a dog"). Exp 10 forces grounding by applying
**dropout to the model's OWN teacher-forcing caption INPUT tokens** — it blanks a
random fraction of the supervised caption-content input embeddings, so the model
cannot lean on its preceding words and must read the image. Lives entirely behind
`CaptionTokenDropoutConfig` (`enabled`/`p_start`/`p_end`), **default OFF →
bit-identical baseline** (the forward never touches `inputs_embeds`). Mechanics:

- **Where:** a single chokepoint in the ForCausalLM `forward`, AFTER the
    multimodal splice produces final `inputs_embeds`+`labels` and BEFORE both
    loss dispatches (chunked-CE and the parent forward) — so it sees the real
    teacher-forcing inputs and covers both paths. It only fires once the splice
    *materializes* `inputs_embeds`, i.e. when **media is present** (text-only
    batches return `inputs_embeds=None` and no-op) — which is exactly the S1
    caption regime (every sample is `<image> → caption`).
- **What is dropped:** positions where `labels != ignore_index` (the supervised
    caption span; under `plain` the whole caption is supervised). Image / audio /
    query / BOS / prompt / padding tokens all carry `ignore_index`, so they are
    **never** eligible. The selected input embeddings are **zeroed** (the repo's
    "blank" idiom — identical to grounding `corruption="blank"`; no new params, no
    tokenizer dependency, trivially bit-identical when off). `masked_fill` is
    out-of-place: kept positions keep their autograd graph, dropped ones become
    zero constants (zero gradient).
- **Labels are NEVER touched** — every position stays supervised; the model is
    only blinded to a fraction of its own preceding caption *inputs*.
- **Rate ramp:** linear `p(step) = p_start + (p_end - p_start)·min(1, step/max_steps)`,
    `p_start=0.10 → p_end=0.30` over the S1 step budget. `step` is the
    rank-identical HF Trainer `global_step` mirrored into a non-persistent
    `_caption_dropout_step` buffer once per optimizer step
    (`VLMTrainer._sync_caption_dropout_step`, mirroring the `_ac_step` pattern) —
    **NOT a per-microbatch counter** (that advances at grad_accum× the rate and
    diverges across DDP ranks; this is the exact bug class that bit ST-2's warmup).
    `max_steps` is `trainer.max_steps`; if it is `<= 0` (epoch-based training) the
    ramp denominator is unresolved, so the rate is **held constant at `p_end`**
    from step 0 and `vlm.py` emits a non-fatal load-time `log.warning` — the
    shipped configs set `max_steps=34722` explicitly to get the intended ramp.
- **Touch-points:** ① `config_schema.py` `CaptionTokenDropoutConfig` on
    `ModelConfig`; ② `vlm.py` **flattens**
    `caption_token_dropout_{enabled,p_start, p_end}` +
    `caption_token_dropout_max_steps` (= `trainer.max_steps`) onto
    `VLMConfig` (grounding/cross_modal pattern; miss this and `getattr` defaults
    OFF); ③ `modeling_vlm.py` module-level `caption_token_dropout_rate` /
    `caption_token_dropout_prob` / `apply_caption_token_dropout` helpers + the
    `_apply_caption_token_dropout` method + the `_caption_dropout_step` buffer
    (registered **unconditionally** but **non-persistent**, so the baseline
    state_dict is unchanged regardless — gating registration on the enabled flag
    is unsound because `vlm.py` flattens that flag onto `model.config` *after*
    `__init__` has already run, so a fresh build would never register it and the
    ramp would be pinned at `p_start`); ④
    `vlm_trainer.py` `_sync_caption_dropout_step`. Regenerate the hub export
    template after the `modeling_vlm.py` edit. **S1 only** — the `-s2` config is
    the native S2 unchanged (the regularizer targets caption pretraining). Test:
    `tests/test_caption_token_dropout.py` (ramp formula; input-only/labels-intact;
    image/structural never dropped; bit-identical when off).

### S1 representation-eval harness (`devtools/` probes)

The S1 stage is read with two complementary signals.
**Cross-image discrimination** (the killer test: `distill_cos` alone is a MIRAGE — a collapsed constant satisfies a per-row cosine, so what matters is whether per-image descriptors carry PER-IMAGE structure): pool each image's per-patch features into one L2-normalized descriptor and check `self_pooled >> cross_pooled`, `retrieval_top1 >> chance`, and `self_centered` staying high after removing the across-image mean (the centering clincher).
**Caption-tracking** (grounded-vs-blind string distinctness): the campaign found these "blind" under a FROZEN LM, but these 10 arms UNFREEZE the LM at S1+S2, so distinct captions become a genuine grounding signal.
All probes are heavy (model + CLIP load) — run on a GPU node, never the login node.
Shared retrieval/centering/pooling lives in `devtools/breen_probe_common.py` (`discrimination_metrics`).

| Probe                      | Script                          | Arms                                                                                                      | What it reads                                                                |
| -------------------------- | ------------------------------- | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| cross-image (distill)      | `breen_probe_xshape.py`         | **4,5,7,8** (distill `_align` methods) **+ 9** (single-pool breen; injects `<query>`, freezes debias EMA) | `visual_distill_head` aligned (student pred, teacher target); needs the head |
| cross-image (distill-free) | `breen_probe_feat.py`           | **1,2,3,6,10** (any native ckpt)                                                                          | raw LLM hidden at image positions, split-half (even/odd patches) retrieval   |
| caption read               | `breen_caption_test.py`         | any                                                                                                       | greedy caption strings + blind/distinct verdict                              |
| caption multi-arm          | `breen_caption_retest.py`       | any                                                                                                       | bare-`<image>` captions across several `LABEL=ckpt` at once                  |
| decode-recipe sweep        | `breen_caption_recipe_sweep.py` | any                                                                                                       | which `rep_penalty`×`no_repeat_ngram` avoids the greedy token-loop           |

Launch each as a fresh self-contained GPU job (no held alloc needed):

```bash
# distill arms (4,5,7,8): cross-image discrimination on the distill head
CKPT=/path/checkpoint-1000 PROBE=xshape LABEL=eve@1000 OUT=/path/out.json sbatch devtools/s1_eval_probe.slurm
# non-distill arms (1,2,3,6,10): distill-free split-half on the raw LLM hidden
CKPT=/path/checkpoint-1000 PROBE=feat  LABEL=nepa@1000 OUT=/path/out.json sbatch devtools/s1_eval_probe.slurm
# caption grounding (any arm): PROBE=caption (single ckpt) or PROBE=recipe (decode sweep)
CKPT=/path/checkpoint-1000 PROBE=caption sbatch devtools/s1_eval_probe.slurm
# multi-arm caption read (any arms)
ARMS="e1=/p/ck e2=/p/ck" PROBE=retest sbatch devtools/s1_eval_probe.slurm
```

`devtools/native_distill_probe_srun.sh` is the interactive twin for `srun`-ing `xshape`/`feat` into a live alloc you already hold (`JOBID=<squeue --me> CKPT=... PROBE=xshape bash devtools/native_distill_probe_srun.sh`); its old hard-coded dead `--jobid`/worktree path are gone (REPO is derived from the script location, JOBID is required).
**Images:** both cross-image probes default to VMCBench-dev (`NIMG`, default 30), which must be HF-cached when offline — pre-cache once on a login node (`uv run python -c "from datasets import load_dataset; load_dataset('suyc21/VMCBench', split='dev')"`) or pass `IMAGES_DIR=.../breen-s2val-j8/qual_images` to run on local PNGs (also the way to get an identical cross-arm image set); the probe errors with this exact guidance if neither is available.
**Query-placement (exp 9 only):** arm-9's representation eval is itself deferred to ST-3 — `breen_probe_xshape.py` does NOT cover it (its image-only sequence carries no `<query>`, so `compute_breen_distill_loss` anchors out without aligning anything). The `QUERY_PLACEMENT=after_text` override below is retained as a forward-looking defensive hook for when ST-3's query-aware probe lands. Context: the eval reads `learnable_query_placement` from the checkpoint `config.json`, and main-trained checkpoints already serialize the correct value (`vlm()` refreshes it on both build paths — see "Self-describing config fields" below), so no load-path fix is needed for arm-9 checkpoints trained on main; the override exists only for a stale/external query checkpoint whose config disagrees with how it trained.

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
1. **The MSC cold-cache lock lands at a root path → PermissionError on the
    FIRST read of a cold `wds_path` stream (`_patch_msc_cold_lock_path`).**
    `multistorageclient` (≤0.49.0) `CacheManager.acquire_lock` builds the download
    lock as `os.path.dirname(os.path.join(cache_dir, key))`, but `key` is the
    ABSOLUTE remote object path (e.g. `/data/yiming/.../.nv-meta/.info.json`) and
    `os.path.join` DROPS the `cache_dir` prefix when its second arg is absolute —
    so the lock lands at `/<container>/...` (a root-owned mount) and a COLD
    cache-miss download dies with `PermissionError`. Cache HITS use a
    slash-stripped key, so only the cold path breaks — it bites every FRESH run
    that streams a prepared CrudeWebdataset (`.nv-meta` read in text mode by
    energon's `get_dataset_info`). `_bootstrap_env` monkeypatches `acquire_lock`
    to `key.lstrip("/")` so the lock sits under the cache dir (it no-ops if the
    streaming extras aren't installed). Do NOT remove this patch until the upstream
    fix ships — without it cold Azure streaming never starts.
1. **`resolve_wds_path` passes through an ABSOLUTE local path verbatim, not just a
    `://` URL.** A pre-staged shard dir like `/gscratch/scrubbed/leoym/encabl-data`
    must be used as-is (`p.startswith("/")`); without that clause it falls through
    to `remote_url(p)` and is mis-resolved into a container-relative `msc://` URL,
    silently pointing the loader at the wrong (remote) location. energon's EPath
    wants the bare absolute path, NOT a `file://` URL. Covered by
    `test_resolve_wds_path`.
1. **Corrupt images are SKIPPED, not fatal — `encode_sample` is wrapped with
    `@skip_corrupt_samples` (`energon_dataset.py`).** Honey-Data-1M (and likely
    bee_stage2) carry occasional corrupt images: a truncated/broken PNG makes PIL
    raise `SyntaxError`, which energon classes as a `SYSTEM_EXCEPTION` and
    re-raises as `FatalSampleError` — killing the DataLoader worker → the rank →
    torch-elastic SIGTERMs the whole job. ONE bad image at shard `00059.tar`
    sample `0000599478` killed `encabl-native-s2` at step ~5000/11905. The wrapper
    converts ANY per-sample encode exception into energon's `SkipSample` (log ONE
    warning with key+shard + a running per-worker drop count, then continue), the
    only signal energon treats as non-fatal — a configured error *handler* can't
    rescue a `SyntaxError` because `SYSTEM_EXCEPTIONS` bypasses it. The WDS
    encoders (`energon_wds.py`) inherit the wrapped `encode_sample` unchanged, so
    the prepared-shard path (where the crash hit) is covered too; if you ever
    override `encode_sample` in a subclass, re-apply `@skip_corrupt_samples`.
    Because `SkipSample` bypasses energon's consecutive-failure tolerance, the
    wrapper keeps its OWN per-worker CONSECUTIVE-skip counter (resets to 0 on any
    successful encode): once it hits `VLM_MAX_CONSECUTIVE_SKIPS` (default 100,
    `0` = disabled) it raises `FatalSampleError` instead of skipping. Because the
    count is per-worker and resets on ANY success, this targets only a SYSTEMATIC
    ~TOTAL failure — essentially every sample failing so there is no successful
    encode to reset the counter (a code bug like `NameError`/`ImportError`, a
    total misconfig such as the audio-off `ValueError`, or data so corrupt that
    nothing decodes) — which would otherwise silently drop every sample and hang
    the run at zero throughput; that silent hang is the danger the backstop
    exists to prevent. It does NOT detect partial corruption: an isolated bad
    sample, or even a single fully-corrupt shard whose failures are INTERLEAVED
    with good samples from other shards/datasets (the shuffle buffer and
    blended-dataset mixing share one task encoder), rarely reaches N consecutive,
    so it is skipped+logged+continued — we do not want an unattended multi-day
    run dying over one bad shard. Covered by `test_energon_skip_corrupt.py`.

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
- **`floating_point_ops` sees two shapes of `image_position_ids`.** The understanding splice passes a list of `(N_i, 2)` tensors; generation batching stacks a single `(B, N, 2)` tensor — test `is not None` / non-empty, never truth-value. BREEN `<query>` sentinels each expand to `num_query` rows in the count (single-pool, ST-3).
- **Auto-resume prefers a checkpoint with optimizer state.** `save_only_model` (legacy pretrain) checkpoints carry no optimizer/scheduler state, so `_checkpoint_is_resumable` (`train.py`) treats them as non-resumable and `_resolve_auto_resume_checkpoint` falls back to the newest OLDER full checkpoint rather than silently restarting the optimizer while advancing the step counter. If no full checkpoint exists the action is backend-aware: plain DDP/single-process resumes WEIGHTS ONLY from the snapshot (optimizer/scheduler/RNG reset, loud warning) to preserve weight progress, while DeepSpeed/FSDP safe-skip auto-resume and train from the loaded base (their resume path needs a `global_step*/`/`optimizer_*/` engine-state dir the snapshot lacks and would otherwise crash with "Can't find a valid checkpoint"). Explicit `resume_from_checkpoint` still opts in.
- **`use_start_end_tokens` unfreezes the embedding modules directly.** `group_params_by_prefix` never creates an `"embeddings"` group, so the old `grouped_params.get("embeddings", [])` was a silent no-op; the new image start/end rows must train even with a frozen LM trunk.
- **Inference must reproduce the training media layout.** `eval.py::generate_response` mirrors the training data path: inject missing placeholders (`ensure_placeholders`), THEN `apply_image_position(..., protected_tokens=(data_args.audio_token,))`, THEN inject `<query>`. The `protected_tokens` argument is not optional when audio can co-occur with an image: `sandwich`/`random` repositioning repeats the question text, so an unprotected `<audio>` would be duplicated and desync the splice (it must stay 1:1 with audio features). All three call sites (local `dataset.py`, energon `energon_dataset.py`, inference `eval.py`) pass it. `build_prompt`'s legacy conv-template branch likewise mirrors `preprocess_multimodal`'s `n_image == 1` guard (`dataset.py`): it hoists a lone `<image>` to the front but leaves every placeholder in place for an interleaved multi-image turn (never collapse N→1 — else the splice consumes image 1 and silently drops images 2..N while the encoder still passes all N). The mmtag `<Image>…</Image>` wrap is de-nested to apply for `n_image >= 1`, matching the training side. As defense-in-depth, `generate_response` then asserts the surviving `<image>` sentinel count equals the number of images passed, raising `ValueError` rather than answering on a subset; `plain`/`qwen`/BREEN paths preserve all placeholders and never trip it.
- **`preprocess_qwen` first-turn role probe is schema-agnostic.** Read the leading turn's role as `source[0].get("from") or source[0].get("role")` — never an unconditional `source[0]["from"]`, which `KeyError`s on the OpenAI `role`/`content` schema. Re-read `source[0]` after stripping a leading system turn (the earlier `first_role` is stale).

## CI lint gate (`devtools/lint.py` → basedpyright)

- **Observed CI failure: surviving warnings turned the `build` job red.** GitHub Actions run 28318894868 (main HEAD `0e02bdb`) ran the gate command `devtools/lint.py` issues — `basedpyright --level error --stats src tests devtools` — which reported `0 errors, 6 warnings` and then `returned non-zero exit status 1` (`devtools/lint.py`: `Lint failed`), while `ruff` and `codespell` passed in the same run. The build genuinely failed on the 6 `reportPrivateLocalImportUsage` warnings in `src/vlm/data/energon_wds.py`.
- **Runner-vs-local divergence is real; treat the cited CI run as authoritative.** Local reproductions with the same locked basedpyright 1.31.4 may instead *filter* those warnings under `--level error` and exit 0 (matching `devtools/lint.py`'s own "warnings stay informational" comment). The CI run shows the opposite, and the `[tool.basedpyright]` baseline comment plus PR #16 disabling a batch of warning-level rules specifically to green CI only make sense if surviving warnings fail the gate on the runner.
- **Fix at the source by extending `[tool.basedpyright]` in `pyproject.toml`** (the PR #16 baseline pattern), not by relaxing `--level error`. Disabling the offending rule next to its siblings drops the diagnostic to `0 errors, 0 warnings, exit 0` in *every* mode; keep the error-level checks intact.
- **`reportPrivateLocalImportUsage` ≠ `reportPrivateImportUsage`.** The plain rule covers private imports from third-party packages; the `Local` variant is basedpyright-only and fires when a *first-party* module re-imports a symbol the source module never lists in `__all__`. `vlm.data.energon_wds` deliberately re-imports megatron-energon helpers (`Cooker`, `WorkerConfig`, `get_train_dataset`, …) through `vlm.data.energon_dataset`, so the `Local` variant is disabled too.

## Running tests in a worktree

The repo's tests import `torch`, which on the CUDA build eagerly imports `triton`; pytest's default assertion-rewriting then breaks triton's `@jit` source inspection (`ValueError: @jit functions should be defined in a Python file` at collection).
Run pytest with `--assert=plain` to skip the rewrite hook, e.g. `PYTHONPATH=<worktree>/src uv run --no-sync python -m pytest --assert=plain -p no:cacheprovider tests/...`.
The `.venv` is symlinked to the main checkout and its editable `vlm` install can point at a DIFFERENT worktree's `src` (whichever last ran `uv sync`), so `PYTHONPATH=<worktree>/src` is REQUIRED to exercise your own code, and you must NOT `uv sync` here (it re-points the shared venv and steals it from the other worktree).
Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` when the Qwen3 tokenizer/config are already cached, so collection skips network and degrades cleanly on a login node.
