# Early-Fusion Access Arms — spec note (2026-06-10)

Plan: `docs/superpowers/plans/2026-06-10-early-fusion-access-arms.md`. Five
experiment arms over the bee-mix baseline (`sft-unified-bee-mix`), every
default bit-identical off.

## Knobs

| Knob | Where | Values | Default |
|---|---|---|---|
| `dataset.image_position` | DatasetConfig → DataArguments → both data paths | keep / question_first / sandwich / random | keep |
| `model.cross_modal_mask.mode` | ModelConfig → `model.config.cross_modal_mask_mode` | none / prefix_lm / img2q_window | none |
| `model.cross_modal_mask.window` | → `model.config.cross_modal_mask_window` | [lo, hi] 1-based inclusive decoder layers | [1, 9] |
| `model.cross_modal_mask.bidirectional` | schema only | True is validate-rejected (v1) | False |

## Arms

- `sft-unified-bee-mix-sandwich` — "Q `<image>` Q" ordering (data-only, FA2).
- `sft-unified-bee-mix-randpos` — per-sample random image placement (data-only, FA2).
- `sft-unified-bee-mix-prefixlm` — bidirectional [system+image+question] prefix,
  causal suffix; `attn_implementation: sdpa` (4D bool masks via the
  transformers 5.10 `create_causal_mask` passthrough).
- `sft-unified-bee-mix-windowearly` — img→question attention edges in layers
  1-9 ONLY (text→image untouched); `attn_implementation: sdpa_xmodal`
  (registered attention fn consuming per-layer `_xmodal_mask` overrides).
- `sft-unified-bee-mix-sandwich-auxexit` — sandwich ordering + existing
  aux-exit CE (k=6, λ0.25, detach=true).

## Implementation map

- Ordering transform: `apply_image_position` (`src/vlm/data/dataset.py`),
  called in energon `encode_sample` (crc32-of-key seed) and json `_get_item`
  (index seed); inference applies the same transform in `generate_response`
  reading the checkpoint's persisted `image_position`.
- Masks: `src/vlm/models/xmodal_mask.py` (builders + `sdpa_xmodal` in
  `ALL_ATTENTION_FUNCTIONS` + mask-registry entry so non-4D decode keeps
  key-padding). Wiring: `install_xmodal_masks` + forward/generate hunks in
  `modeling_vlm.py`. Cross-modal edges only ever target prefix columns
  (positions before the first supervised label; whole prompt at inference) —
  answer keys are never exposed non-causally.
- Self-describing checkpoints: `image_position`, `cross_modal_mask_mode`,
  `cross_modal_mask_window` serialize into `config.json`; `load_model`
  auto-upgrades `sdpa → sdpa_xmodal` when the checkpoint says `img2q_window`
  (because `_attn_implementation` itself never serializes).

## Known v1 limits

- Mutual (bidirectional) windowing not implemented (needs per-layer decode
  masking); config field reserved, value True rejected at validation.
- Text-only generate keeps stock causal masks in all modes (vision benchmarks
  always carry an image; documented in `generate`).
- Mask arms assume single-turn prompts at inference (prefix = whole prompt);
  multi-turn chat would treat earlier answers as prefix — out of scope for
  lmms-eval.
- GPU verification: `devtools/xmodal_smoke.py` / `.slurm` (no-op equivalence,
  question-sensitivity theorem check, leakage guard, generate, throughput).
