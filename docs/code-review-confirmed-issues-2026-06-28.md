# Code Review: Confirmed Training, Inference, Data, and Export Issues

Date: 2026-06-28

This document records confirmed issues found by static code review of the
repository. "Confirmed" here means the issue has direct code evidence and a
plausible execution path; it does not mean every issue was reproduced with a
runtime test in this environment.

The review focused on:

- training correctness and checkpoint/resume behavior;
- inference correctness and train/inference alignment;
- multimodal data loading and batching;
- BREEN learnable-query behavior;
- text-to-image generation paths;
- lmms-eval and devtool evaluation paths;
- Hub export templates.

Runtime note: targeted pytest execution was attempted, but this container did
not have `pytest` installed. Findings below are therefore based on static code
evidence.

## High severity

### 1. BREEN S2 query placement is not persisted on `from_pretrained`

**Status:** confirmed by code inspection.

`sft-unified-breen.yaml` trains S2 with `learnable_query.placement: after_text`
while loading from an S1 checkpoint. The `from_pretrained` branch loads the
checkpoint config but does not overwrite `model.config.learnable_query_placement`
from the composed S2 config. Inference then reads placement from the saved model
config, so an S2 checkpoint can be trained with `after_text` but infer with the
stale S1 `after_image` placement.

**Evidence:**

- `src/vlm/config/sft-unified-breen.yaml:17-23`
- `src/vlm/vlm.py:227-263`
- `src/vlm/vlm.py:416-419`
- `src/vlm/inference/eval.py:194-200`

**Impact:** BREEN S2 train/inference mismatch; query blocks are inserted at the
wrong prompt position during inference.

**Suggested fix:** when loading from a checkpoint, refresh branch-agnostic
self-describing config fields such as `learnable_query_placement` from the
current composed config before training/saving.

### 2. BREEN plain inference drops `<query>` placeholders

**Status:** confirmed by code inspection.

`generate_response()` injects BREEN `<query>` placeholders, but `build_prompt()`
for the `plain` template keeps only placeholders matched by `_media_regex()`.
That regex includes image/audio tokens only. Training-side `preprocess_plain()`
uses `_media_pattern()`, which includes `<query>` when BREEN is enabled.

**Evidence:**

- `src/vlm/inference/eval.py:68-73`
- `src/vlm/inference/eval.py:312-326`
- `src/vlm/inference/eval.py:509-515`
- `src/vlm/data/dataset.py:131-138`
- `src/vlm/data/dataset.py:1078-1108`

**Impact:** BREEN S0/S1 plain caption-style inference does not use the trained
learnable-query pathway.

**Suggested fix:** include `query_token` in inference `_media_regex()` when
`learnable_query_enabled` is true, and add prompt-parity tests for plain+BREEN.

### 3. Local JSON multi-image samples collapse all `<image>` tokens to one

**Status:** confirmed by code inspection.

The local JSON path loads all images from `sample["image"]` when it is a list,
but `preprocess_multimodal()` removes every `<image>` occurrence and prepends
exactly one image token. The collator still flattens all loaded images. The
model splice consumes one image feature per sentinel, leaving extra features
unused or available to be consumed by later rows.

**Evidence:**

- `src/vlm/data/dataset.py:337-342`
- `src/vlm/data/dataset.py:1363-1371`
- `src/vlm/data/dataset.py:1399-1413`
- `src/vlm/data/dataset.py:1554-1561`
- `src/vlm/models/modeling_vlm.py:1830-1835`

**Impact:** multi-image local JSON training/inference data can silently lose
images or misalign image features with later samples.

**Suggested fix:** preserve the exact number/order of image placeholders for
multi-image samples, or fail loudly when the local JSON path receives
unsupported multi-image input.

### 4. BREEN token-budget bucketing undercounts query expansion

**Status:** confirmed by code inspection.

`effective_sample_length()` accounts for image/audio sentinel expansion only.
BREEN uses one `<query>` sentinel per image, and each sentinel expands into
`num_fine + num_coarse` learnable-query rows. Current BREEN configs use
length-bucketed token-budget batching.

**Evidence:**

- `src/vlm/data/energon_dataset.py:768-790`
- `src/vlm/config/model/qwen3-1.7b-unified-breen.yaml:49-53`
- `src/vlm/config/dataset/energon-breen-caption.yaml:34-36`
- `src/vlm/config/dataset/energon-honey-sft.yaml:16-21`
- `src/vlm/models/modeling_vlm.py:1758-1763`
- `src/vlm/models/modeling_vlm.py:1835-1842`

**Impact:** BREEN microbatches can be larger than intended, causing OOMs and
wrong effective batch/gradient semantics.

**Suggested fix:** subtract query sentinels and add
`learnable_query_num_fine + learnable_query_num_coarse` rows per query
placeholder in `effective_sample_length()`.

### 5. `pretrain-unified` composes plain preprocessing with instruct data

**Status:** confirmed by config/code inspection.

`pretrain-unified` selects `trainer: pretrain`, whose conversation version is
`plain`, while the default dataset `energon-mix` includes
`LLaVA-OneVision-1.5-Instruct`. The plain preprocessor expects exactly two
turns and drops all human text except media placeholders.

**Evidence:**

- `src/vlm/config/pretrain-unified.yaml:3-6`
- `src/vlm/config/trainer/pretrain.yaml:3`
- `src/vlm/config/dataset/energon-mix.yaml:30-32`
- `src/vlm/config/dataset/energon-vision.yaml:1-4`
- `src/vlm/data/dataset.py:1083-1094`

**Impact:** multi-turn instruct samples can crash, and two-turn VQA/instruct
samples train as media-only captioning instead of question-conditioned
instruction following.

**Suggested fix:** either use a caption-only dataset for `pretrain-unified` or
select a Qwen chat template for instruct data. Add a config validation check.

### 6. Qwen ChatML delimiter unmasking breaks cross-modal prefix detection

**Status:** confirmed by code inspection.

Cross-modal masks define prefix positions as tokens before the first supervised
label. Qwen preprocessing first masks system/user tokens, then globally unmasks
ChatML delimiters/newline tokens even inside system/user turns. This can make
the first supervised label occur at the system/user delimiter, producing an
empty or tiny prefix for `prefix_lm` and `img2q_window` training masks. During
generation, labels are absent, so the whole prompt becomes prefix.

**Evidence:**

- `src/vlm/models/xmodal_mask.py:19-28`
- `src/vlm/data/dataset.py:650-677`
- `src/vlm/models/modeling_vlm.py:667-673`
- affected configs include `sft-unified-bee-mix-prefixlm.yaml` and
  `sft-unified-bee-mix-windowearly.yaml`

**Impact:** cross-modal mask arms can train with a different attention pattern
from inference and may not provide the intended image/question access.

**Suggested fix:** compute prefix boundaries from role/turn metadata or a
separate prompt mask, not from labels after delimiter unmasking.

### 7. Hub export template is stale and can export broken models

**Status:** confirmed by code inspection.

The export template still contains `torch.arrange` typos and lacks several live
model paths: BREEN learnable queries, visual-expert routing, visual-prefix
support, and text-to-image generation modules/API. The push-to-hub path renders
this template directly.

**Evidence:**

- `templates/modeling_vlm.py.j2:506`
- `templates/modeling_vlm.py.j2:651`
- `templates/modeling_vlm.py.j2:674`
- `templates/modeling_vlm.py.j2:109-120`
- `templates/modeling_vlm.py.j2:372-403`
- `src/vlm/utils/push_to_hub.py:246-253`
- `tests/test_inference.py:359-365`

**Impact:** exported checkpoints can fail at runtime or run with behavior that
does not match local inference.

**Suggested fix:** regenerate/synchronize the template from the live dynamic
model implementation and extend export tests to BREEN, visual-prefix,
visual-expert, and generation checkpoints.

### 8. Enabling generation on an understanding checkpoint is not guarded

**Status:** confirmed by code inspection.

When `trainer.from_pretrained` loads an understanding checkpoint, generation
modules are rebuilt only if the checkpoint config already has generation
enabled. A composed config can still enable generation training, and batches
with `target_patches` will call `forward_generation()` even though
`gen_x_head`/`gen_t_embed`/`gen_patch_embed` are absent or `None`.

**Evidence:**

- `src/vlm/vlm.py:227-263`
- `src/vlm/models/modeling_vlm.py:590-592`
- `src/vlm/models/modeling_vlm.py:2022-2033`

**Impact:** runtime crash instead of a clear load-time configuration error.

**Suggested fix:** add a load-time guard like the existing visual-aux retrofit
guard: fail if generation is requested but the checkpoint lacks generation
modules.

## Medium severity

### 9. Inference applies `image_position` before auto-inserting `<image>`

**Status:** confirmed by code inspection.

Training inserts missing media placeholders first, then applies
`apply_image_position()`. Inference does the reverse and only repositions if the
query already contains exactly one image token.

**Evidence:**

- `src/vlm/inference/eval.py:492-508`
- `src/vlm/data/energon_dataset.py:723-736`

**Impact:** common calls such as `query="What is this?", images=...` ignore the
trained image-layout policy.

**Suggested fix:** call `ensure_placeholders()` before applying
`apply_image_position()` in inference.

### 10. Qwen local JSON path crashes on first-turn `role`/`content`

**Status:** confirmed by code inspection.

`preprocess_qwen()` partially supports `role`/`content`, but after reading
`first_role`, it unconditionally accesses `source[0]["from"]`.

**Evidence:**

- `src/vlm/data/dataset.py:634-640`
- later per-turn handling supports both schemas at `src/vlm/data/dataset.py:655-662`

**Impact:** OpenAI/HF-style local JSON conversations fail in the Qwen path.

**Suggested fix:** use the already-computed first role or a shared helper that
supports both schemas.

### 11. Literal media-token text can steal real typed media placement

**Status:** confirmed by code inspection.

Energon typed content converts image/audio items into literal placeholder
strings. `inject_missing_media_tokens()` then counts raw placeholder substrings
in all text and neutralizes surplus occurrences from the end. If user text
quotes `<image>` before a real image item, the real item can be neutralized and
the quoted literal can remain as the model sentinel.

**Evidence:**

- `src/vlm/data/energon_dataset.py:643-667`
- `src/vlm/data/dataset.py:196-221`

**Impact:** image/audio features can be trained at the wrong textual position.

**Suggested fix:** preserve typed media positions structurally until after
literal text has been escaped, or mark generated placeholders so they cannot be
confused with quoted text.

### 12. Collator truncation can silently drop media sentinels

**Status:** confirmed by code inspection.

The collator truncates `input_ids` and `labels` before model splice. If a
placeholder appears after a long text prefix, it can be truncated while the
media feature remains in the batch. The model then treats the row as missing
the modality and consumes the feature as a zero-width dummy.

**Evidence:**

- `src/vlm/data/dataset.py:1540-1561`
- `src/vlm/models/modeling_vlm.py:1782-1787`

**Impact:** samples can silently become text-only while still paying media
loading/encoding cost; training signal is wrong.

**Suggested fix:** validate media placeholder counts after truncation, or
truncate in a modality-aware way that keeps required sentinels.

### 13. `sandwich` image-position mode duplicates audio placeholders

**Status:** confirmed by code inspection.

`apply_image_position()` removes only the image token. In `sandwich` mode it
duplicates the remaining question text around the image token. A mixed turn
such as `<image>\n<audio>\nQ` therefore becomes a prompt with two `<audio>`
tokens, but only one audio feature exists. Placeholder-count validation has
already run before this rewrite.

**Evidence:**

- `src/vlm/data/dataset.py:391-407`
- `src/vlm/data/dataset.py:1393-1413`
- `src/vlm/data/energon_dataset.py:724-736`

**Impact:** mixed image+audio samples can hit audio sentinel/feature mismatch.

**Suggested fix:** make `apply_image_position()` operate only on text with all
non-image placeholders temporarily protected, or re-run validation after
position rewriting.

### 14. Generation data/model patch size mismatch is not validated

**Status:** confirmed by code/config inspection.

The generation dataset uses `dataset.gen_patch_size`, while the model uses
`generation.independent_embed` and `generation.embed_patch_size` to size the
embedder/head. The schema documents the required match, but no validation
enforces it.

**Evidence:**

- `src/vlm/data/energon_dataset.py:944-950`
- `src/vlm/data/energon_dataset.py:1096-1104`
- `src/vlm/models/modeling_vlm.py:2037-2056`
- `src/vlm/config/config_schema.py:410-415`

**Impact:** misconfiguration can produce target patch dimensions that do not
match the model head/embedder, for example 768 vs 6912.

**Suggested fix:** validate `dataset.gen_patch_size` against
`model.generation.embed_patch_size` when independent embedding is enabled.

### 15. Generation FLOPs accounting can crash on tensor `image_position_ids`

**Status:** confirmed by code inspection.

Generation batching stacks `image_position_ids` into a tensor, but
`floating_point_ops()` tests it in boolean context.

**Evidence:**

- `src/vlm/data/energon_dataset.py:1014-1018`
- `src/vlm/models/modeling_vlm.py:2303-2305`

**Impact:** HF Trainer FLOPs accounting can raise "Boolean value of Tensor with
more than one value is ambiguous" for generation batches.

**Suggested fix:** test `image_position_ids is not None` and handle both list
and tensor shapes explicitly.

### 16. BREEN FLOPs accounting undercounts query rows

**Status:** confirmed by code inspection.

`floating_point_ops()` expands image/audio tokens but not BREEN query tokens.

**Evidence:**

- `src/vlm/models/modeling_vlm.py:2289-2339`

**Impact:** BREEN throughput/FLOPs metrics are underreported.

**Suggested fix:** add query-sentinel replacement using the serialized
learnable-query row count.

### 17. Auto-resume can select model-only pretrain checkpoints

**Status:** confirmed by code/config inspection.

Legacy pretrain configs save model-only checkpoints, but `train()` automatically
resumes from the latest checkpoint in `output_dir` if present. Model-only
checkpoints do not carry optimizer/scheduler/RNG state.

**Evidence:**

- `src/vlm/config/trainer/pretrain.yaml:11-12`
- `src/vlm/config/pretrain-llava.yaml:1-6`
- `src/vlm/train/train.py:357-395`

**Impact:** reruns/requeues can fail to resume or resume with incorrect
optimizer/scheduler continuity.

**Suggested fix:** skip auto-resume for `save_only_model` checkpoints or require
explicit resume mode.

### 18. `img2q_window` generation mask state can leak into later generations

**Status:** confirmed by code inspection.

`install_xmodal_masks()` writes `_xmodal_mask` onto layer attention modules.
The generation path installs these masks but does not reliably clear the
per-layer state before returning.

**Evidence:**

- `src/vlm/models/modeling_vlm.py:280-306`
- `src/vlm/models/modeling_vlm.py:1456-1479`
- `src/vlm/models/xmodal_mask.py:102-115`

**Impact:** a later generation call with matching shapes can reuse stale masks.

**Suggested fix:** clear per-layer `_xmodal_mask` in a `finally` block around
generation prefill/use.

### 19. Direct model API can skip xmodal generation masks without attention mask

**Status:** confirmed by code inspection.

The project inference wrapper passes `attention_mask`, but a direct
`model.generate(..., images=...)` caller can omit it. Internal splice creates a
mask and later restores `None`; generation xmodal mask installation is gated on
`attention_mask is not None`.

**Evidence:**

- `src/vlm/models/modeling_vlm.py:1708-1711`
- `src/vlm/models/modeling_vlm.py:1994-1997`
- `src/vlm/models/modeling_vlm.py:1449-1458`

**Impact:** direct users can evaluate an xmodal checkpoint with plain causal
prefill.

**Suggested fix:** make `generate()` create and keep a default all-ones
attention mask before splice.

### 20. Visual distillation is not rejected on classic encoder-backed models

**Status:** confirmed by code inspection.

The non-BREEN distill path assumes raw-patch `image_position_ids`. Classic
encoder batches provide images and image sizes, but no `image_position_ids`.
`compute_distill_loss()` indexes `distill_positions[k]` without a `None` guard
once image blocks exist.

**Evidence:**

- `src/vlm/data/dataset.py:1562-1567`
- `src/vlm/models/modeling_vlm.py:744-746`
- `src/vlm/models/modeling_vlm.py:1130-1196`

**Impact:** enabling visual distillation on a classic encoder-backed model can
crash at training time.

**Suggested fix:** add config validation that visual distillation is native/raw
patch only, or implement classic-path positions.

### 21. lmms-eval hardcodes media placeholder strings

**Status:** confirmed by code inspection.

The adapter inserts literal `<image>` and `<audio>` instead of reading the
checkpoint-configured tokens.

**Evidence:**

- `src/vlm/inference/lmms_eval.py:121-124`
- `src/vlm/inference/eval.py:175-200`

**Impact:** checkpoints trained with custom media tokens are evaluated with
wrong prompts.

**Suggested fix:** build placeholders from `_data_args_from_config()` or the
loaded model config.

### 22. lmms-eval cache key omits doc/media identity

**Status:** confirmed by code inspection.

The cache key uses only `(ctx, gen_kwargs)`, while the request includes task,
split, doc id, and extracted media.

**Evidence:**

- `src/vlm/inference/lmms_eval.py:130-158`

**Impact:** repeated text prompts across different images can reuse the wrong
cached answer.

**Suggested fix:** include `(task, split, doc_id)` or a media identity hash in
the cache key.

### 23. lmms-eval drops or changes generation kwargs

**Status:** confirmed by code inspection.

The adapter only forwards a small subset of generation kwargs. It also treats
`do_sample=True` as ineffective unless temperature is positive, because
`generate_response()` derives `do_sample` from `temperature > 0`.

**Evidence:**

- `src/vlm/inference/lmms_eval.py:139-155`
- `src/vlm/inference/eval.py:546-557`

**Impact:** benchmark generation settings can differ from the lmms-eval task
request.

**Suggested fix:** thread through supported generation kwargs explicitly and
allow `do_sample` to be passed independently.

### 24. BREEN fusion/devtool probes bypass query injection

**Status:** confirmed by code inspection.

Some analysis scripts hand-build prompts with only `<image>` and never call the
BREEN query injection path.

**Evidence:**

- `devtools/aux_fusion_probe.py:128-147`
- `devtools/aux_fusion_full.py:139-153`

**Impact:** BREEN checkpoints can be probed/evaluated on a different input
contract from training and live inference.

**Suggested fix:** reuse `generate_response()` prompt construction helpers or
explicitly call `inject_query_placeholders()`.

## Low severity and configuration footguns

### 25. Chunked CE hard-codes `-100`

**Status:** confirmed by code inspection.

The schema exposes `ignore_index`, and splice uses `self.config.ignore_index`,
but chunked CE pads and filters with literal `-100`.

**Evidence:**

- `src/vlm/config/config_schema.py:44`
- `src/vlm/models/modeling_vlm.py:872-879`
- `src/vlm/models/modeling_vlm.py:1837-1840`

**Impact:** non-default `ignore_index` configurations train incorrectly or
error.

**Suggested fix:** use `self.config.ignore_index` consistently.

### 26. Direct local `.jsonl` paths are documented but not supported

**Status:** confirmed by code inspection.

The schema comment says local json/jsonl/yaml-mixture is supported. Direct
`dataset.path=foo.jsonl` falls through to `json.load(file)`, while JSONL support
exists only inside YAML mixtures.

**Evidence:**

- `src/vlm/config/config_schema.py:351`
- `src/vlm/data/dataset.py:1215-1219`
- `src/vlm/data/dataset.py:1246-1251`

**Impact:** direct JSONL dataset paths fail unexpectedly.

**Suggested fix:** either implement direct JSONL parsing or update the schema
comment and validation.

### 27. `batch_token_budget` is silently ignored without `length_buckets`

**Status:** confirmed by code inspection.

The schema says token-budget batching needs length buckets. The loader only
constructs the bucketed encoder when `length_buckets` is set; otherwise
`batch_token_budget` has no effect.

**Evidence:**

- `src/vlm/config/config_schema.py:381-387`
- `src/vlm/data/energon_dataset.py:1106-1119`

**Impact:** a config typo can silently switch to fixed batch size and alter
memory/samples-per-step behavior.

**Suggested fix:** fail validation if `batch_token_budget` is set while
`length_buckets` is empty.

### 28. `use_start_end_tokens` tries to unfreeze a missing parameter group

**Status:** confirmed by code inspection.

`set_trainable_params()` attempts to unfreeze grouped params named
`"embeddings"`, but `group_params_by_prefix()` never creates that group.

**Evidence:**

- `src/vlm/train/set_trainable.py:37-68`
- `src/vlm/train/set_trainable.py:175-177`

**Impact:** frozen-LM runs with newly added start/end tokens can leave those
new embeddings frozen.

**Suggested fix:** add an embeddings group or directly unfreeze token embedding
rows/modules when start/end tokens are enabled.

### 29. `img2q_window` treats BREEN query tokens as text keys

**Status:** confirmed by code inspection.

`img2q_window` defines question text as `prefix & ~is_img`. Query block IDs are
not passed into the mask builder, so BREEN query rows are not excluded from
question-text keys.

**Evidence:**

- `src/vlm/models/xmodal_mask.py:63-65`
- `src/vlm/models/modeling_vlm.py:280-300`
- `src/vlm/models/modeling_vlm.py:667-673`

**Impact:** with BREEN `after_image`, image rows can attend future query blocks
inside the image-to-question window.

**Suggested fix:** pass query block IDs to mask construction and exclude query
positions from question-text keys.

### 30. `eval_model()` cannot pass `image_aspect_ratio`

**Status:** confirmed by code inspection.

`generate_response()` supports an `image_aspect_ratio` override, but the
one-shot `eval_model()` wrapper does not expose or forward it.

**Evidence:**

- `src/vlm/inference/eval.py:378-381`
- `src/vlm/inference/eval.py:442-455`
- `src/vlm/inference/eval.py:575-613`

**Impact:** older classic encoder checkpoints without saved aspect-ratio config
can be evaluated with the wrong preprocessing.

**Suggested fix:** add an `image_aspect_ratio` parameter to `eval_model()` and
forward it.

### 31. `goal_dashboard` selects nearest checkpoint, not nearest previous checkpoint

**Status:** confirmed by code inspection.

The comment says to choose the nearest bar checkpoint `<= step`, but the code
uses absolute nearest and can select a future checkpoint.

**Evidence:**

- `devtools/goal_dashboard.py:69-75`

**Impact:** dashboard comparisons can use a later training budget.

**Suggested fix:** filter candidates to `s <= step` before choosing the nearest.

## Recommended test additions

Add focused tests for:

1. BREEN S2 `after_text` placement persistence after `from_pretrained`.
2. Plain+BREEN prompt parity: `<image><query>` must survive inference prompt
   construction.
3. Local JSON multi-image samples preserving placeholder count/order.
4. BREEN `effective_sample_length()` query expansion.
5. Qwen first-turn `role`/`content` local JSON compatibility.
6. Mixed image+audio `image_position="sandwich"` placeholder count validation.
7. Media-placeholder count validation after collator truncation.
8. Direct local `.jsonl` dataset loading or explicit rejection.
9. Generation data/model patch-size validation.
10. Hub export smoke tests for BREEN, visual-prefix, visual-expert, and
    generation checkpoints.
11. lmms-eval cache keys with repeated text but different images.
12. Cross-modal mask prefix construction under Qwen ChatML with delimiter
    unmasking.
