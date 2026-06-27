# Auxiliary Intermediate-Layer CE Loss for Early Fusion (aux-exit)

**Date:** 2026-06-05
**Status:** approved (design); implementation pending
**Owner experiment:** early-fusion ablation on the encoder-free unified VLM

## 1. Goal & hypothesis

The encoder-free unified model (`sft-unified.yaml`, Qwen3-1.7B backbone, raw
patches spliced into the embedding sequence) is suspected to process vision
and text largely separately in its early layers — consistent with our NEO
probing (`neo_report/REPORT.md`: decision-critical fusion at relative depth
~0.35–0.65 across architectures) and the literature (Mono-InternVL
arXiv:2410.08202, Cross-modal Information Flow arXiv:2411.18620).

**Intervention:** add an auxiliary CE loss at an intermediate decoder layer,
decoded through the model's own final RMSNorm + lm_head (logit-lens style,
LayerSkip mechanism). To predict image-dependent text tokens correctly at
layer k, visual information must reach text positions by layer k — a direct
gradient pressure toward earlier fusion.

**Scientific framing:** this is an ablation to test whether *induced* early
fusion helps, not just an engineering trick. Conclusions require (a) the
final-CE not degrading, (b) a manipulation check that fusion actually moved
earlier (isolation-depth sweep, separate effort), (c) benchmark deltas.

**Novelty (verified 2026-06-05 via 4-agent lit sweep):** no published work
applies LM-head CE deep supervision in a VLM to encourage modality fusion.
Closest: LayerSkip (arXiv:2404.16710 — same mechanism, text-only, speedup
purpose); OLA-VLM (arXiv:2412.09585 — same purpose-family, embedding
distillation mechanism).

## 2. Loss specification

Let `h_k` be the raw output of decoder layer k (1-indexed, k ∈ [1, 27] for
the 28-layer Qwen3-1.7B; k = 28 is rejected — it duplicates the main loss),
`V` the set of valid target positions (shifted labels ≠ -100; image splice
and user turns are already -100), shared across both losses:

```
L_final = (1/|V|) Σ_V CE( lm_head( h_final_normed ),     y_shift )   # unchanged
L_aux_k = (1/|V|) Σ_V CE( lm_head( RMSNorm(h_k) ),       y_shift )
L_total = L_final + λ · Σ_{k ∈ K} L_aux_k
```

- `RMSNorm` and `lm_head` are the **shared existing modules** (LayerSkip's
  `forward_early` recipe: `model.norm` then `lm_head`); no new parameters.
- `K` = `trainer.aux_exit_layers` (list[int]); this run: `[6]`.
- `λ` = `trainer.aux_exit_weight`; this run: `0.25` (EE-LLM's validated
  single-exit range 0.1–0.5, arXiv:2312.04916).
- Multiple layers contribute **additively** (each weighted λ). Footgun note:
  listing many layers raises total aux mass; that is the user's choice.
- `trainer.aux_exit_detach` (bool, default false): when true, the aux branch
  uses `norm.weight.detach()` / `lm_head.weight.detach()` — gradients flow
  only into layers ≤ k, not into the (tied, cf. arXiv:2603.26663) unembedding.
  Fuse for the tied-embedding gradient-coupling risk; off by default to match
  the literature-validated recipe.

Both losses divide by the same `|V|` and use fp32 CE in `loss_chunk_size`
chunks — numerically the same construction as the existing chunked CE
(`devtools/test_chunked_ce.py` pins it to transformers' ForCausalLMLoss).

## 3. Implementation

### 3.1 Where the aux loss lives

`src/vlm/models/modeling_vlm.py :: chunked_ce_forward` (the training-only
loss path; gated on `self.training`, `labels is not None`,
`loss_chunk_size > 0` — see `forward`, lines ~221–238). The aux loss is
**only** implemented here. If `aux_exit_layers` is set while
`loss_chunk_size == 0`, `train.py` raises at startup (do not silently train
without the aux loss).

### 3.2 Capturing h_k — scoped manual hook, NOT `output_hidden_states`

Verified against the installed transformers **5.10.2**:

- 5.x collects hidden states via `@capture_outputs` + a
  `_CAN_RECORD_REGISTRY` keyed by `str(self.__class__)`
  (`transformers/utils/output_capturing.py`). Our backbone is a **dynamically
  generated subclass** (`create_dynamic_vlm_class`), so registry lookup
  behavior is not guaranteed → `output_hidden_states=True` risks silently
  returning nothing. **Do not use it.**
- Instead: register a forward hook on `self.model.layers[k-1]` immediately
  before the backbone call in `chunked_ce_forward`, capture the layer output
  (`out[0] if isinstance(out, tuple) else out`), and remove the handle in a
  `finally`. Scoped registration means backward-time gradient-checkpointing
  recomputation **cannot** re-fire the hook (handle already removed).
- Gradient flow is sound because transformers 5.10.2 hard-defaults
  gradient checkpointing to `use_reentrant=False`
  (`modeling_utils.py:3232`); non-reentrant checkpointing runs the real
  forward with grad enabled, so the captured tensor is graph-connected.
  Guard anyway: if `self.training` and a captured tensor has
  `requires_grad == False`, raise RuntimeError (catches any future
  reentrant/no-grad regression instead of silently training without aux
  gradients).
- Memory: the captured tensor is the checkpoint-boundary activation that
  autograd stores anyway — one extra reference, ~zero extra VRAM.

### 3.3 Loss computation (restructured `chunked_ce_forward`)

1. Compute shift targets, `valid` mask, `n_valid` **once** (existing code).
2. Factor the existing per-chunk loop into a helper
   `_ce_sum(hidden_valid, targets_valid, head_weight)` returning the fp32 CE
   **sum**; main loss calls it with `self.lm_head.weight`.
3. For each captured `h_k`: flatten → `[valid]` → functional RMSNorm
   replicating `Qwen3RMSNorm.forward` exactly (fp32 upcast, `pow(2).mean`,
   `rsqrt(var + eps)`, downcast to input dtype, **then** multiply by weight —
   order pinned by `modeling_qwen3.py:59-64`), with
   `weight = norm.weight.detach() if aux_exit_detach else norm.weight`,
   `eps = self.model.norm.variance_epsilon` → `_ce_sum` with
   (optionally detached) `lm_head.weight` → accumulate
   `loss += aux_exit_weight * (aux_sum / n_valid)`.
4. Degenerate batch (`n_valid == 0`): existing zero-loss branch unchanged;
   aux contributes nothing (all touched params already get zero grads via
   the main degenerate path).
5. Stash `self._last_ce_components = {"ce_final": t, "ce_aux": t}` as
   detached scalar tensors **every step** (`ce_aux` = unweighted
   `Σ_k L_aux_k`, so the logged value is λ-independent; zeros on the
   degenerate branch,
   so the log-time collective below never desyncs across ranks). Only
   stashed when aux is enabled — baseline behavior byte-identical.

### 3.4 Config plumbing (mirrors `loss_chunk_size` end-to-end)

| File | Change |
|---|---|
| `src/vlm/config/config_schema.py` (`TrainerConfig`) | `aux_exit_layers: list[int] = field(default_factory=list)`, `aux_exit_weight: float = 0.25`, `aux_exit_detach: bool = False` + doc comments |
| `src/vlm/train/training_arguments.py` | same three fields on the custom `TrainingArguments`; pass through in `get_training_args` |
| `src/vlm/train/train.py` | next to `model.config.loss_chunk_size = ...` (line ~84): validate `1 <= k < model.config.num_hidden_layers` (clear error message), require `loss_chunk_size > 0` when layers non-empty, then `model.config.aux_exit_layers = [int(k) for k in ...]` (plain ints — OmegaConf `ListConfig` must not leak into `config.json` serialization), plus weight/detach. |
| `src/vlm/train/vlm_trainer.py` (`VLMTrainer.log`) | when `"loss" in logs` and the model has `_last_ce_components`: all-reduce mean (same deterministic-log-step pattern as `samples_seen_session`) and emit `ce_final` / `ce_aux`. |
| `src/vlm/config/sft-unified-earlyfusion.yaml` (new) | `defaults: [sft-unified, _self_]`; `trainer: {aux_exit_layers: [6], aux_exit_weight: 0.25, aux_exit_detach: false}`. Seed untouched → identical energon data order to the baseline. |

### 3.5 What must NOT change (live-baseline safety)

The baseline `sft-unified` job (35916542 + pending requeue 35921758)
**re-launches from this working tree on every requeue.** Therefore:

- With `aux_exit_layers` empty (all existing configs), every touched code
  path must be behaviorally identical: no hooks registered, no extra
  tensors, loss bit-exact. The refactor of the chunk loop into `_ce_sum`
  must preserve the exact op order (same split, same fp32 cast point, same
  reduction).
- `model.config` gains three new keys in future checkpoints; loading older
  checkpoints uses `getattr` defaults (same pattern as `loss_chunk_size`).
- Eval/generation paths (`super().forward`, `generate`) are untouched: aux
  loss exists only inside `chunked_ce_forward`.
- No edits to dataset, splice (`prepare_inputs_labels_for_multimodal`),
  optimizer, or checkpoint logic.

## 4. Failure modes considered

| Risk | Mitigation |
|---|---|
| Hook re-fires during checkpoint recompute → stale/duplicate capture | scoped register/remove around the backbone call; handle gone before backward |
| Reentrant checkpointing (no-grad forward) → aux gradient silently dead | 5.10.2 defaults `use_reentrant=False` (verified); runtime `requires_grad` guard |
| 5.x `output_hidden_states` + dynamic class registry mismatch | not used at all |
| OmegaConf `ListConfig` breaks `config.json` save | cast to `list[int]` in train.py |
| Cross-rank collective desync on component logging | stash always written (zeros on degenerate batches); log() runs on all ranks at deterministic steps |
| Baseline requeue picks up broken code | parity tests + local smoke pass **before** the next requeue window; aux off ⇒ identical path |
| DeepSpeed unused-parameter issues | aux uses only parameters the main loss already touches |
| FLOPs metric undercounts the aux head pass | accepted, documented (tflops_per_gpu is a session-average diagnostic) |
| torch.compile | chunked CE already graph-breaks (see sft-unified.yaml notes); aux adds breaks only in the already-broken region; compile is off for this run |

## 5. Testing (before any cluster launch)

`devtools/test_aux_exit.py`, mirroring `devtools/test_chunked_ce.py`
(tiny random Qwen3 config, CPU):

1. **Baseline parity:** `aux_exit_layers=[]` → loss bit-identical to the
   pre-change implementation (and to the full-logits reference, as the
   existing test pins).
2. **Numerical correctness:** aux enabled → equals a naive reference that
   runs the backbone, applies the real `model.norm` module + full-vocab
   `lm_head` at layer k, and computes `L_final + λ·L_aux` (fp32, tight atol).
3. **Functional-RMSNorm parity:** the replicated functional norm equals
   `Qwen3RMSNorm.forward` on random bf16/fp32 tensors.
4. **Gradient routing:** (a) λ > 0 changes `embed_tokens` grads vs λ = 0;
   (b) with `aux_exit_detach=True`, `lm_head.weight.grad` is identical to
   the no-aux run on the same batch (aux contributes nothing to the head)
   while layer-≤k grads differ.
5. **Checkpointing interaction:** `gradient_checkpointing_enable()` (default
   non-reentrant) → same loss as without checkpointing; captured tensor
   `requires_grad`; backward succeeds.
6. **Local smoke:** 0.6B, 3 steps
   (`python -m vlm -cn sft-unified-earlyfusion model=qwen3-0.6b-unified
   dataset.jsonl_name=train_mini.jsonl ... trainer.deepspeed=null
   trainer.attn_implementation=sdpa trainer.report_to=none`) — finite loss,
   `ce_final`/`ce_aux` both logged, ce_aux > ce_final at init.

## 6. Run & decision plan

- Launch: `CONFIG_NAME=sft-unified-earlyfusion sbatch train_h200.slurm`
  (separate RUN_DIR `/gscratch/scrubbed/leoym/small-vlm-outputs/sft-unified-earlyfusion`).
- Monitor: `ce_final` must track the baseline loss curve within ~1%;
  `ce_aux`'s own decay curve is data (layer-6 readability over training).
- **Decision point at ckpt-5000** (steps matched to the baseline): (a)
  final-CE parity, (b) isolation-depth-sweep onset shifts earlier vs
  baseline (manipulation check; harness ported from
  `neo_analysis/gemma4_sweep.py` — separate spec), (c) MME/POPE via
  lmms-eval. Kill or continue to 20k.

## 7. References

LayerSkip arXiv:2404.16710 · EE-LLM arXiv:2312.04916 · CALM
arXiv:2207.07061 · tuned lens arXiv:2303.08112 · weight-tying bias
arXiv:2603.26663 · early-exit diminishing returns arXiv:2603.23701 ·
Mono-InternVL arXiv:2410.08202 · cross-modal flow arXiv:2411.18620 ·
OLA-VLM arXiv:2412.09585 · local probing `neo_report/REPORT.md`
