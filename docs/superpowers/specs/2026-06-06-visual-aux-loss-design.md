# Pluggable Visual Auxiliary Losses at Image Positions (visual-aux)

**Date:** 2026-06-06
**Status:** implemented 2026-06-06 (branch worktree-visual-aux; GPU gate devtools/test_visual_aux.py passed)
**Owner experiment:** visual-target deep supervision ablation on the encoder-free unified VLM
**Sibling spec:** `2026-06-05-aux-exit-loss-design.md` (same mount point, orthogonal axis:
aux-exit varies *where* CE supervision attaches; this spec varies *what* image positions predict)

## 1. Goal & hypothesis

In the encoder-free unified model (`sft-unified.yaml`, Qwen3-1.7B, raw 48px
patches → `RawPatchConnector` → trunk, fully causal attention incl. within
image blocks), image-token representations receive only *indirect* gradient
from text CE. **Intervention:** add an auxiliary next-patch prediction loss at
image positions — teacher-forced, causal, no attention-mask changes, no
sampling/generation — with the prediction target pluggable per config:

- `aim_pixel` — predict the **next patch's real pixels** (AIM/AIMv2 lineage,
  arXiv:2401.08541 / 2411.14402; AIMv2's decoder-side objective is natively
  causal-AR, exactly our regime).
- `nepa` — predict the **next patch's connector embedding** (NEPA,
  arXiv:2512.16922: causal + shift-by-one + stop-grad target + bidirectional
  L2-norm cosine).

**Novelty (verified 2026-06-06, 2-agent lit sweep):** every direct
encoder-free comparable — NEO (2510.14979, same Qwen3-1.7B trunk), SAIL
(2504.10462), EVEv2 (2502.06788), Mono-InternVL (2410.08202) — uses **no**
visual aux loss, but all of them rely on bidirectional intra-image attention
and/or large data. No published result isolates a visual aux loss under
*fully causal* intra-image attention at ~2B scale; NEPA itself has no VLM
result and no latent-vs-pixel control. Arms 1–2 are novel measurements.

**Collapse analysis (the design's central correctness argument):**

| | target source | network can manipulate target? | degenerate zero-loss solution |
|---|---|---|---|
| `aim_pixel` | real pixels (external ground truth) | no | **none exists** — identity/pass-through yields `‖patch_i − patch_{i+1}‖² ≫ 0`; loss floor = true conditional uncertainty of the next patch |
| `nepa` | own trainable connector output | yes (map all patches to a constant → cosine ≡ 1) | exists; **prevented by stop-grad on the target** (SimSiam-style; NEPA ablation: no stop-grad → loss → −1 collapse, no shift → diverges) |

Both objectives predict position *j+1* from position *j* (**shift-by-one,
strictly within each image block**). Same-position reconstruction is
forbidden — it admits the identity shortcut.

Known weak shortcut for `aim_pixel` (not collapse): raster-adjacent patches
share an edge, so local texture extrapolation reduces loss without semantics.
Mitigants: 48px patches carry more content than the 14–16px in the
literature; AIMv2 still yields SOTA-semantic features with pixel targets +
text CE. The pixel-vs-latent delta is itself a primary readout of this study.

## 2. Loss specification

Notation: image k (flat batch-order index, same order as the `images` /
`image_features` lists) occupies a contiguous spliced span; `j` is the
intra-block patch index; `h` is the trunk hidden state (post final RMSNorm
for the default last-layer attachment); `P_k[j] ∈ R^6912` the raw patch row;
`E_k[j] ∈ R^2048` the connector output row.

Prediction pairs: for every position carrying patch `j` of image k such that
`j+1 ≤ N_k − 1` targets exist (truncation may remove trailing *positions*;
targets come from the tensors, so surviving positions keep valid pairs):

```
aim_pixel:  pred = head(h_j)                       ∈ R^6912
            tgt  = zscore(P_k[j+1])                 # per-patch: (p − mean(p)) / sqrt(var(p) + 1e-6), MAE formula
            L    = mean over pairs of mean over 6912 dims of (pred − tgt)²    # fp32

nepa:       pred = F.normalize(head(h_j), dim=-1)  ∈ R^2048
            tgt  = F.normalize(E_k[j+1].detach(), dim=-1)    # stop-grad: MANDATORY
            L    = −mean over pairs of ⟨pred, tgt⟩
```

`L_total = L_CE + λ · L_visual`, with `λ = trainer.visual_aux_weight`
(default 0.5 for both arms — user decision 2026-06-06; AIMv2's α=0.4 noted as the literature prior for pixel targets).

- **Head (both objectives):** fresh MLP on `h`, depth
  `visual_aux_head_depth` (default 2: `Linear(2048→hidden) → GELU →
  Linear(hidden→out)`), `hidden = visual_aux_head_hidden` (default 2048),
  `out = 6912 (patch_dim, derived from connector config) | 2048
  (hidden_size)`. Deviation from NEPA (which has no head) is required: `h`
  simultaneously serves CE and cannot be pinned to `E[j+1]` directly.
  AIMv2 precedent: single linear head on a causal decoder; depth stays a dial.
- **Reductions:** pixel MSE is **mean over patch dims** (sum-reduction would
  re-weight CE:FM silently whenever patch geometry changes); both losses
  mean over all pairs in the microbatch, fp32.
- **First patch of each block is target-only** (nothing predicts `j=0`;
  NEPA's CLS→patch-1 pair has no clean analogue here — the preceding
  position is a text token already supervised by CE). Blocks with `N_k < 2`
  (incl. zero-width dummy images) produce no pairs.
- **Optional depth axis:** `visual_aux_layer: null | k` — `null` attaches to
  the post-final-norm last hidden state; `k` reuses the aux-exit scoped-hook
  capture (`chunked_ce_forward` lines ~318–342) and applies the shared final
  RMSNorm functionally (identical recipe to aux-exit) before the head.

## 3. Implementation

### 3.1 Where it lives

`src/vlm/models/modeling_vlm.py :: chunked_ce_forward` only (training-only
path; same gates as aux-exit). `train.py` raises at startup if
`visual_aux_objective != "none"` while `loss_chunk_size == 0`. Eval and
generation paths untouched; the head is dead weight at inference.

### 3.2 Image-block span tracking (the one splice change)

`prepare_inputs_labels_for_multimodal` (modeling_vlm.py:652–876) gains a
parallel `image_block_ids` row built in the same assembly loop as
`cur_new_labels`:

- text/audio segments → `−1`; the image-feature splice at lines 760–773 →
  the **cursor value** consumed for that placeholder (which indexes both the
  flat `images` and `image_features` lists, batch-cursor-ordered).
- Audio features (token index −201) get `−1` — only
  `config.image_token_index` splices receive block ids.
- Zero-width dummy consumption (lines 727–732, text-only rows) never
  receives a block id; dummy images therefore contribute no pairs, and the
  head's anchor term (§3.4) keeps its params in the graph.
- The ids tensor follows the **same truncation
  (`[:tokenizer_model_max_length]`) and right/left padding (pad −1)** as
  `new_labels`.
- Construction is **gated**: ids are built and returned only when the caller
  asks (forward passes `need_block_ids = self.training and objective !=
  "none"`); otherwise the value is `None` and the assembly loop is
  byte-identical to today. Signature grows by one return element; both call
  sites (forward L222–233, generate L457–467) are updated, generate discards
  it.

`chunked_ce_forward` gains pass-through params: `images` (raw patch list),
`image_features` (connector outputs, kept from the encode phase when the
objective is `nepa`), `image_block_ids`.

### 3.3 Targets

- `aim_pixel`: z-score each row of `images[k]` in fp32 at loss time (MAE
  formula, `var` unbiased, eps 1e-6). No stop-grad needed (constants).
- `nepa`: `image_features[k].detach()` — the connector output *as spliced*
  (same tensors the trunk consumed). Holding the extra reference costs no
  extra memory (autograd already retains them as inputs_embeds parts).
- Pair assembly per block: positions of block k in row r are contiguous;
  intra-block index `j = position − block_start`. Predictions = head of `h`
  at positions with `j ≤ N_k − 2` present; targets = rows `j+1`.

### 3.4 Loss computation & logging

1. After the existing CE (and aux-exit) computation, gather block spans from
   `image_block_ids`, build (pred-position, target-row) index lists, run the
   head in fp32-upcast chunks (reuse the `loss_chunk_size` chunking pattern;
   head matmuls are small — 6912 ≪ 152k vocab — so this is cheap).
2. `loss = loss + visual_aux_weight * L_visual`.
3. **Degenerate microbatch (no pairs):** `loss += head(flat_hidden[:1]).float().sum() * 0.0`
   — the established repo pattern (`n_valid == 0` CE branch) that keeps
   DeepSpeed ZeRO-2 gradient reduction shape-stable when a microbatch has
   no image pairs.
4. Stash detached scalars into the existing `_last_ce_components` dict
   (extend, do not replace): `visual_aux` (unweighted `L_visual`), and for
   `nepa` two collapse alarms: `visual_aux_cos` (mean cosine; sustained
   drift toward 1.0 = inspect) and `visual_aux_tgt_std` (per-dim std of the
   normalized targets, mean over dims; → 0 is the unambiguous collapse
   signature). Zeros on degenerate batches so the log-time all-reduce in
   `vlm_trainer.py:114–128` never desyncs ranks. `VLMTrainer.log` emits the
   new keys with the identical all-reduce-mean pattern.

### 3.5 Config plumbing (mirrors aux-exit end-to-end)

| File | Change |
|---|---|
| `src/vlm/config/config_schema.py` (`TrainerConfig`) | structural dials `model.visual_aux.{objective, head_depth, head_hidden}` (ModelConfig section) + trainer dials `visual_aux_weight: float = 0.5`, `visual_aux_layer: int | None`, `visual_aux_head_lr/wd: float | None` on TrainerConfig |
| `src/vlm/train/training_arguments.py` | same fields; pass through in `get_training_args` |
| `src/vlm/train/train.py` | validate: objective in the registry; `loss_chunk_size > 0` when active; `visual_aux_layer` in `[1, num_hidden_layers − 1]` when set; weight > 0 warning-if-zero (mirror aux-exit's loud-warning commit 63bd8c9). Copy plain Python scalars to `model.config` (no OmegaConf leakage into `config.json`). The model **builds the head in `__init__`** from `model.config` fields (so `from_pretrained` re-creates it); train.py only validates + copies trainer-side knobs (λ, lr). |
| `src/vlm/models/modeling_vlm.py` + `templates/modeling_vlm.py.j2` | head construction gated on `config.visual_aux_objective != "none"` (audio-connector pattern: absent → attribute is `None`); loss per §3.4; template mirrored so exported checkpoints carry the head code |
| `src/vlm/models/configuration_vlm.py` + `.j2` | no change needed — `visual_aux_*` keys ride the config kwargs passthrough (`conversation_version` precedent) |
| `src/vlm/train/set_trainable.py` | `component_prefixes["visual_aux_head"] = ["visual_aux_head"]` — **without this the head's params silently fall through to the `language_model` group** (set_trainable.py:61–66) and take the LM lr; also verify the freeze flags never touch it (it must stay trainable in every recipe) |
| `src/vlm/train/optimizer.py` | `component_to_config["visual_aux_head"] = "visual_aux_head"`; `component_configs["visual_aux_head"] = {lr: visual_aux_head_lr or default, wd: visual_aux_head_wd or language_model_wd}` |
| `src/vlm/config/sft-unified-aimpixel.yaml`, `sft-unified-nepa.yaml` (new) | `defaults: [sft-unified, _self_]` + the objective/λ; seeds untouched → identical energon data order to the baseline |

Refinement during implementation (2026-06-06): structural dials moved from
TrainerConfig to `model.visual_aux` so the causal `__init__` can build the
head from the config it is constructed with.

### 3.6 What must NOT change (live-baseline safety)

Running jobs relaunch from this working tree on requeue. With
`visual_aux_objective: "none"` (all existing configs):

- no head module built, no block-id construction, no extra returns consumed,
  loss path bit-exact; splice assembly loop unchanged when ids not requested.
- `model.config` gains new keys; older checkpoints load via `getattr`
  defaults (the `loss_chunk_size` pattern).
- No edits to dataset, collator, attention (FA2 stays), tokenizer,
  checkpoint logic, eval/generate behavior.
- Retrofit note: loading an old `sft-unified` checkpoint into a
  visual-aux-enabled config leaves the head randomly initialized
  (transformers warns on missing keys, does not crash) — fine, the head is
  always fresh.

## 4. Failure modes considered

| Risk | Mitigation |
|---|---|
| NEPA representational collapse | stop-grad target (NEPA-ablation-validated as sufficient in pure SSL; we additionally hold the CE anchor); `visual_aux_tgt_std` / `visual_aux_cos` alarms logged every step |
| aim_pixel low-level shortcut (edge extrapolation) | not collapse (loss floor > 0); 48px patches; pixel-vs-latent arms are the controlled comparison |
| Same-position reconstruction degeneracy | forbidden by construction: shift-by-one within blocks only |
| Head params land in the wrong optimizer group | explicit `visual_aux_head` prefix registration (set_trainable.py falls through to language_model otherwise — verified) |
| DeepSpeed unused-param desync on image-free microbatches | anchor term `head(h[:1]).sum() * 0.0` (existing repo pattern) |
| Shift crossing block/text boundaries | pairs built per contiguous block span from `image_block_ids`; property-tested (§5) |
| Truncation cuts a block mid-way | ids truncated identically to labels; surviving positions keep valid pairs (targets indexed from tensors, not positions) |
| Dummy images create spurious pairs | zero-width splice ⇒ no block id ⇒ no pairs; `N_k < 2` guard |
| Hook re-fire under gradient checkpointing (mid-layer arm) | scoped register/remove + `requires_grad` guard, identical to aux-exit §3.2 |
| OmegaConf types leaking into `config.json` | cast to plain int/float/str in train.py (aux-exit pattern) |
| Cross-rank log desync | components stashed every step incl. zeros on degenerate batches |
| Rank-asymmetric stash on a fully image-free microbatch (`images is None` → splice early-return → no va keys while other ranks stash them → all-reduce hang) | dormant in the shipped vision-only arms: the collator always injects a dummy image entry, so `images` is never None; track before reusing on text-containing blends — safe fix is stashing va zeros whenever the head is active, independent of ids presence |
| bf16 loss noise | head output and targets upcast to fp32 before the loss, matching chunked-CE convention |

## 5. Testing (before any cluster launch)

`devtools/test_visual_aux.py`, mirroring `devtools/test_chunked_ce.py`
(tiny random Qwen3 config, CPU, sdpa):

1. **Baseline parity:** `objective="none"` → loss bit-identical to current
   implementation; splice outputs identical; no head attribute.
2. **Pair-construction property test:** hand-built batch (2 rows: one with
   two images of N=4 and N=3 patches + text, one text-only with dummy) →
   exactly `(4−1)+(3−1)=5` pairs; no pair crosses a boundary; dummy
   contributes none; left- and right-padding both correct; truncation that
   cuts the second image to 2 surviving positions yields the expected pairs.
3. **aim_pixel numerical correctness:** equals a naive reference (z-score →
   MSE mean) computed with full-precision tensors; z-score matches the MAE
   formula on random patches.
4. **nepa correctness + stop-grad routing:** loss equals naive reference;
   `target.requires_grad is False`; connector `.grad` from the *target* path
   is zero (compare: detach removed → grads differ); head grads nonzero.
5. **No-collapse-shortcut sanity (shift guard):** the copy baseline — using
   z-scored patch `j` itself as the "prediction" for target `j+1` — yields
   loss ≫ 0 on natural-image patches, while the same copy against target `j`
   is ≈ 0. A same-position-target bug would collapse the two cases together.
6. **Degenerate batch:** all-text microbatch → finite loss, head grads exist
   (zeros), components stashed as zeros.
7. **Mid-layer arm:** `visual_aux_layer=k` under gradient checkpointing →
   same loss as without checkpointing; captured tensor requires grad.
8. **Optimizer grouping:** head params appear in the `visual_aux_head` group
   with the configured lr, absent from `language_model`.
9. **Local smoke** (per objective): 0.6B, 3 steps,
   `python -m vlm -cn sft-unified-aimpixel model=qwen3-0.6b-unified
   dataset.jsonl_name=train_mini.jsonl dataset.num_workers=1
   trainer.max_steps=3 trainer.per_device_train_batch_size=2
   trainer.gradient_accumulation_steps=1 trainer.deepspeed=null
   trainer.attn_implementation=sdpa trainer.report_to=none` — finite loss,
   `visual_aux` logged; same for `-cn sft-unified-nepa` (+ alarms logged).

## 6. Run & decision plan

| Arm | Config | λ | Question answered |
|---|---|---|---|
| 0 | `sft-unified` (already planned/running) | — | causal small-scale floor |
| 1 | `sft-unified-aimpixel` | 0.5 | does next-patch *pixel* supervision help understanding? |
| 2 | `sft-unified-nepa` | 0.5 | does *latent* beat *pixel* (or no-aux)? collapse behavior in a co-trained VLM |
| 3 (cheap, optional) | arm-1 yaml + `visual_aux_layer: 6` | 0.5 | depth axis; crosses the aux-exit research line |

- Launch: `CONFIG_NAME=sft-unified-aimpixel sbatch train.slurm` (separate
  RUN_DIRs per arm).
- Monitor: text CE must track the baseline within ~1% (aux must not starve
  CE); `visual_aux` decay curve is itself data; nepa alarms (§3.4).
- **Decision point at ckpt-5000** (steps matched to baseline): lmms-eval
  curves (MME/POPE per the existing harness) vs arm 0; kill or continue to
  20k. Pixel-vs-latent delta and either-vs-baseline delta are the paper
  readouts regardless of sign.

## 7. Deferred / out of scope (explicit)

- **Generation/sampling** (flow-matching head, CFG, denoising loop, gen/edit
  data): deferred; the `visual_aux_objective` registry and the per-objective
  target-prep boundary are the re-entry interface. Full design research
  archived in conversation + memory (`visual-aux-loss-pivot-2026-06`).
- Bidirectional intra-image attention arm (the confound all null-result
  comparables share; would reuse the sdpa+4D-mask design from the generation
  research).
- ROSS-style denoise / frozen-encoder distillation arms (break the
  "no pretrained vision components" narrative; oracle-only).
- Multi-step prediction (j+k), text-position→patch-0 pairs, EMA teacher and
  variance regularizers (fallbacks if stop-grad alone proves insufficient).

## 8. References

AIM arXiv:2401.08541 · AIMv2 arXiv:2411.14402 · NEPA arXiv:2512.16922 ·
MAE arXiv:2111.06377 (per-patch norm) · SimSiam arXiv:2011.10566 (stop-grad) ·
NEO arXiv:2510.14979 · SAIL arXiv:2504.10462 · EVE arXiv:2406.11832 ·
EVEv2 arXiv:2502.06788 · Mono-InternVL arXiv:2410.08202 ·
ROSS arXiv:2410.09575 · OLA-VLM arXiv:2412.09585 ·
sibling spec `2026-06-05-aux-exit-loss-design.md`
