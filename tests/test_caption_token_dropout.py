"""Tests for the Stage-1 caption-token (input word) dropout regularizer (exp10).

The mechanism blanks a ramped fraction of the model's OWN teacher-forcing caption
INPUT embeddings — the supervised positions (labels != ignore_index; under the
`plain` caption recipe the whole caption is supervised) — so it cannot lower the
loss by leaning on its own previous words and must read the image. Three load
bearing invariants (captain's spec):

  * **Ramp = f(global step).** The rate ramps linearly p_start -> p_end over the
    optimizer-step budget, computed from the rank-identical HF Trainer
    global_step (mirrored into the model buffer), NOT a per-microbatch counter.
  * **Input-only, labels intact.** Only INPUT embeddings at supervised positions
    are zeroed; `labels` are never touched, so every position stays supervised.
  * **Image / structural tokens never dropped.** Image / audio / query / BOS /
    prompt tokens all carry ignore_index labels, so they are never eligible.

Two layers: pure-function unit tests (the rate + masking math, exhaustive and
deterministic) and a tiny CPU encoder-free VLM (the feature integrates into the
real splice + loss forward, fires when on, and is bit-identical when off).
"""

from typing import Any

import pytest
import torch
from transformers import AutoConfig

import vlm.models.modeling_vlm as mvlm
from vlm.models.modeling_vlm import (
    apply_caption_token_dropout,
    caption_token_dropout_prob,
    caption_token_dropout_rate,
)

IGN = -100


# ---------------------------------------------------------------------------
# unit layer: the rate ramp + drop probability resolution
# ---------------------------------------------------------------------------


def test_rate_ramp_at_known_steps() -> None:
    """p(step) = p_start + (p_end - p_start) * min(1, step / max_steps)."""
    assert caption_token_dropout_rate(0, 100, 0.10, 0.30) == pytest.approx(0.10)
    assert caption_token_dropout_rate(25, 100, 0.10, 0.30) == pytest.approx(0.15)
    assert caption_token_dropout_rate(50, 100, 0.10, 0.30) == pytest.approx(0.20)
    assert caption_token_dropout_rate(100, 100, 0.10, 0.30) == pytest.approx(0.30)


def test_rate_ramp_clamped_past_budget() -> None:
    """Past max_steps the fraction saturates at 1 -> stays at p_end (a resume
    that overshoots the budget never exceeds p_end)."""
    assert caption_token_dropout_rate(250, 100, 0.10, 0.30) == pytest.approx(0.30)
    # max_steps <= 0 (unset schedule) -> full ramp, never a divide-by-zero.
    assert caption_token_dropout_rate(7, 0, 0.10, 0.30) == pytest.approx(0.30)
    assert caption_token_dropout_rate(7, -1, 0.10, 0.30) == pytest.approx(0.30)


def test_prob_disabled_is_zero() -> None:
    """Disabled config -> 0.0 so apply_caption_token_dropout no-ops (the forward
    is bit-identical to baseline). Resolved from the flat config + step."""
    from types import SimpleNamespace

    off = SimpleNamespace(caption_token_dropout_enabled=False)
    assert caption_token_dropout_prob(off, 50) == 0.0
    # A config with NO caption_token_dropout_* keys (old checkpoint) -> disabled.
    assert caption_token_dropout_prob(SimpleNamespace(), 50) == 0.0
    on = SimpleNamespace(
        caption_token_dropout_enabled=True,
        caption_token_dropout_max_steps=100,
        caption_token_dropout_p_start=0.10,
        caption_token_dropout_p_end=0.30,
    )
    assert caption_token_dropout_prob(on, 50) == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# unit layer: the masking (input-only, labels intact, structural never dropped)
# ---------------------------------------------------------------------------


def _labels_with_structure() -> torch.Tensor:
    """labels mimicking the post-splice plain layout: positions 0..2 are
    image / BOS / prompt (ignore_index), 3..6 supervised caption, 7 padding
    (ignore_index)."""
    labels = torch.full((2, 8), IGN)
    labels[:, 3:7] = torch.tensor([10, 11, 12, 13])
    return labels


def test_drop_is_input_only_labels_intact() -> None:
    """p=1 drops EVERY supervised input embedding (deterministic), zeroes ONLY
    those rows, and never mutates labels."""
    labels = _labels_with_structure()
    labels_before = labels.clone()
    emb = torch.randn(2, 8, 4)
    out = apply_caption_token_dropout(emb, labels, p=1.0, ignore_index=IGN)

    assert torch.equal(labels, labels_before), "labels were mutated"
    supervised = labels != IGN
    # supervised rows fully zeroed (dropped), ignore rows bit-identical to input.
    assert torch.all(out[supervised] == 0.0)
    assert torch.equal(out[~supervised], emb[~supervised])


def test_structural_and_image_tokens_never_dropped() -> None:
    """Across many random draws, no ignore_index position (image / BOS / prompt
    / padding) is ever zeroed, regardless of p."""
    labels = _labels_with_structure()
    ignore_mask = labels == IGN
    for seed in range(50):
        g = torch.Generator().manual_seed(seed)
        emb = torch.ones(2, 8, 4)
        out = apply_caption_token_dropout(emb, labels, p=0.9, ignore_index=IGN, generator=g)
        # any row that became zero must be a supervised (non-ignore) position.
        zeroed = (out == 0.0).all(dim=-1)
        assert not torch.any(zeroed & ignore_mask), f"ignore position dropped (seed {seed})"


def test_p_zero_is_identity() -> None:
    """p <= 0 returns the SAME tensor (no clone, no RNG consumed) -> the disabled
    path cannot perturb training or desync the global RNG stream."""
    labels = _labels_with_structure()
    emb = torch.randn(2, 8, 4)
    assert apply_caption_token_dropout(emb, labels, p=0.0, ignore_index=IGN) is emb
    assert apply_caption_token_dropout(emb, labels, p=-0.5, ignore_index=IGN) is emb


def test_drop_blocks_gradient_only_at_dropped_positions() -> None:
    """Dropped positions become zero constants (zero grad); kept positions keep
    their autograd graph."""
    labels = _labels_with_structure()
    emb = torch.randn(2, 8, 4, requires_grad=True)
    g = torch.Generator().manual_seed(0)
    out = apply_caption_token_dropout(emb, labels, p=0.5, ignore_index=IGN, generator=g)
    out.sum().backward()
    # masked_fill -> dropped rows get grad 0, every other element grad 1.
    assert torch.all((emb.grad == 0.0) | (emb.grad == 1.0))


# ---------------------------------------------------------------------------
# model layer: tiny CPU encoder-free VLM (integrates into the real forward)
# ---------------------------------------------------------------------------

BASE_LM = "Qwen/Qwen3-0.6B"
VISION_DIALS = dict(patch_size=4, pooling_kernel_size=1, max_soft_tokens=16)
PATCH_DIM = (4 * 1) ** 2 * 3  # 48


def _build_vlm(*, loss_chunk_size: int = 0, **ctd: Any) -> Any:
    """A tiny encoder-free VLM with the caption-token-dropout flat config dials
    (normally set by vlm.vlm). Seeded so two builds share weights."""
    from vlm.models import get_dynamic_vlm

    base_cfg = AutoConfig.from_pretrained(BASE_LM)
    base_cfg.hidden_size = 32
    base_cfg.intermediate_size = 64
    base_cfg.num_hidden_layers = 2
    base_cfg.layer_types = ["full_attention"] * 2
    base_cfg.num_attention_heads = 4
    base_cfg.num_key_value_heads = 2
    base_cfg.head_dim = 8
    base_cfg.max_position_embeddings = 512
    kwargs = dict(
        hf_name=BASE_LM,
        vision_config={**VISION_DIALS, "hf_name": None, "hidden_size": PATCH_DIM},
        connector_config={
            "name": "raw_patch",
            "type": "raw_patch",
            "mm_embed_dim": 32,
            "mm_posemb_size": VISION_DIALS["max_soft_tokens"],
        },
        audio_config=None,
        **base_cfg.to_dict(),
        image_token="<image>",
        image_token_index=-200,
        audio_token="<audio>",
        audio_token_index=-201,
        ignore_index=IGN,
        max_seq_length=512,
        padding_side="left",
        loss_chunk_size=loss_chunk_size,
        caption_token_dropout_enabled=ctd.get("enabled", False),
        caption_token_dropout_p_start=ctd.get("p_start", 0.10),
        caption_token_dropout_p_end=ctd.get("p_end", 0.30),
        caption_token_dropout_max_steps=ctd.get("max_steps", 100),
    )
    _, VLMConfig = get_dynamic_vlm(BASE_LM)
    config = VLMConfig(**kwargs)
    VLMForCausalLM, _ = get_dynamic_vlm(BASE_LM)
    torch.manual_seed(0)
    model = VLMForCausalLM(config)
    model.train()
    return model


def _image_batch(n_patches: int = 4) -> dict[str, Any]:
    """An image-bearing caption training batch (the S1 caption-pretrain case):
    one <image> sentinel (-200) the splice expands to n_patches feature rows
    (labels ignore_index), then 3 supervised caption tokens. The dropout only
    fires once the splice materializes inputs_embeds, i.e. when media is present
    — exactly the S1 regime — so the model tests must carry an image."""
    return dict(
        input_ids=torch.tensor([[-200, 5, 6, 7]]),
        labels=torch.tensor([[IGN, 5, 6, 7]]),
        images=[torch.randn(n_patches, PATCH_DIM)],
        image_position_ids=[torch.zeros(n_patches, 2, dtype=torch.long)],
    )


def test_buffer_registered_only_when_enabled() -> None:
    """The rank-identical step buffer (mirrored by VLMTrainer) is registered ONLY
    when enabled, so a disabled / baseline checkpoint's state_dict is unchanged
    and is non-persistent (never serialized)."""
    assert not hasattr(_build_vlm(enabled=False), "_caption_dropout_step")
    on = _build_vlm(enabled=True)
    assert hasattr(on, "_caption_dropout_step")
    assert "_caption_dropout_step" not in on.state_dict()  # non-persistent


def test_forward_enabled_runs_and_changes_loss() -> None:
    """Enabled (training) the dropout fires inside the real chunked-CE forward
    (the production loss path): finite loss that DIFFERS from the dropout-off
    forward of the same weights — proving the blanking actually reaches the
    loss. (The chunked path returns loss only; it never materializes full
    logits, which is its whole point.)"""
    model = _build_vlm(enabled=True, p_start=1.0, p_end=1.0, loss_chunk_size=8)
    batch = _image_batch()
    model._caption_dropout_step.fill_(0)  # mirrored optimizer step

    loss_on = model(**batch).loss
    assert torch.isfinite(loss_on)

    # Same model + same image, dropout turned off at runtime -> the no-op path.
    model.config.caption_token_dropout_enabled = False
    loss_off = model(**batch).loss
    assert torch.isfinite(loss_off)
    assert not torch.allclose(loss_on, loss_off), "dropout did not affect the loss"


def test_forward_disabled_is_bit_identical_and_deterministic() -> None:
    """Disabled == enabled-at-p=0: neither perturbs the embeddings nor consumes
    RNG, so both yield identical logits to the baseline forward (and the disabled
    forward is reproducible across calls)."""
    batch = _image_batch()  # one shared batch -> identical image for both models
    off = _build_vlm(enabled=False, loss_chunk_size=8)
    p0 = _build_vlm(enabled=True, p_start=0.0, p_end=0.0, loss_chunk_size=8)
    p0._caption_dropout_step.fill_(123)

    loss_off1 = off(**batch).loss
    loss_off2 = off(**batch).loss
    loss_p0 = p0(**batch).loss
    assert torch.equal(loss_off1, loss_off2), "disabled forward is not deterministic"
    assert torch.allclose(loss_off1, loss_p0, atol=1e-6), "p=0 / disabled forwards differ"


def test_image_positions_never_dropped_end_to_end(monkeypatch: Any) -> None:
    """With a real image spliced in, the post-splice image-feature labels are all
    ignore_index, so the dropout (even at p=1) blanks ONLY caption positions and
    never an image row."""
    model = _build_vlm(enabled=True, p_start=1.0, p_end=1.0)
    model._caption_dropout_step.fill_(0)

    n_patches = 4
    images = [torch.randn(n_patches, PATCH_DIM)]
    image_position_ids = [torch.zeros(n_patches, 2, dtype=torch.long)]
    # one <image> sentinel (-200) then 3 supervised caption tokens.
    input_ids = torch.tensor([[-200, 5, 6, 7]])
    labels = torch.tensor([[IGN, 5, 6, 7]])

    captured: dict[str, torch.Tensor] = {}
    real = mvlm.apply_caption_token_dropout

    def spy(inputs_embeds: torch.Tensor, lab: torch.Tensor, **kw: Any) -> torch.Tensor:
        out = real(inputs_embeds, lab, **kw)
        captured["labels"] = lab.clone()
        captured["changed"] = (out != inputs_embeds).any(dim=-1)  # (B, L) rows touched
        return out

    monkeypatch.setattr(mvlm, "apply_caption_token_dropout", spy)
    model(input_ids=input_ids, labels=labels, images=images, image_position_ids=image_position_ids)

    spliced_labels = captured["labels"]
    # the splice expands the sentinel to n_patches feature rows, all ignore_index.
    assert (spliced_labels[0, :n_patches] == IGN).all(), "image rows are supervised"
    assert spliced_labels.shape[1] == n_patches + 3
    # every row the dropout touched must be a supervised (caption) position.
    changed = captured["changed"]
    assert not torch.any(changed & (spliced_labels == IGN)), "an image/ignore row was dropped"
    # at p=1 all 3 caption rows are dropped.
    assert int(changed.sum()) == 3
