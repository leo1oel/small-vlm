"""Cross-modal 4D attention masks for the early-fusion access arms (plan
docs/superpowers/plans/2026-06-10-early-fusion-access-arms.md).

All builders return (B, 1, L, L) bool masks, True = "query row may attend
key column". transformers 5.10 passes prepared 4D masks through
create_causal_mask untouched (masking_utils.py:839), so these reach SDPA
as-is. Added cross-modal edges always target PREFIX columns only (prefix =
non-pad positions strictly before the sample's first supervised label;
labels=None, i.e. generation prefill, makes the whole prompt the prefix) —
answer keys are never exposed non-causally, so training matches inference.
"""

import torch
from torch import Tensor

IGNORE_INDEX_DEFAULT = -100


def _prefix(attn2d: Tensor, labels: Tensor | None, ignore_index: int) -> Tensor:
    """(B, L) bool: non-pad positions before the first supervised label."""
    keep = attn2d.bool()
    if labels is None:
        return keep
    bsz, seq_len = labels.shape
    idx = torch.arange(seq_len, device=labels.device).expand(bsz, seq_len)
    supervised = labels.ne(ignore_index)
    first = torch.where(supervised, idx, torch.full_like(idx, seq_len)).amin(dim=1)
    return keep & (idx < first.unsqueeze(1))


def build_base_mask(attn2d: Tensor) -> Tensor:
    """Plain causal mask with key-side padding, as a 4D bool tensor."""
    keep = attn2d.bool()
    seq_len = keep.shape[1]
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=keep.device).tril()
    return (causal.unsqueeze(0) & keep.unsqueeze(1)).unsqueeze(1)


def build_cross_modal_mask(
    attn2d: Tensor,
    image_block_ids: Tensor | None,
    labels: Tensor | None,
    mode: str,
    ignore_index: int = IGNORE_INDEX_DEFAULT,
) -> Tensor:
    """Base causal mask plus the arm's cross-modal edges.

    prefix_lm     - bidirectional attention within the prefix
                    (system + image + question), causal suffix. PaliGemma
                    masking transplanted; image_block_ids unused.
    img2q_window  - image-position query rows additionally attend to
                    question-text key columns (prefix & not image). The
                    LAYER windowing happens at install time (the windowed
                    layers get this mask, the rest get build_base_mask).
    """
    base = build_base_mask(attn2d)
    prefix = _prefix(attn2d, labels, ignore_index)
    if mode == "prefix_lm":
        extra = prefix.unsqueeze(2) & prefix.unsqueeze(1)
    elif mode == "img2q_window":
        if image_block_ids is None:
            raise ValueError("img2q_window needs image_block_ids")
        is_img = image_block_ids.ge(0)
        q_text = prefix & ~is_img
        extra = is_img.unsqueeze(2) & q_text.unsqueeze(1)
    else:
        raise ValueError(f"unknown cross_modal_mask mode: {mode!r}")
    return base | extra.unsqueeze(1)


from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def sdpa_xmodal_forward(module, query, key, value, attention_mask, **kwargs):
    """Stock SDPA, but a layer can carry its own 4D mask override
    (module._xmodal_mask, stashed by the model before the decoder runs). The
    shape guard makes decode steps (q_len 1) fall back to the passed-in
    causal row automatically, so generation needs no per-step bookkeeping."""
    override = getattr(module, "_xmodal_mask", None)
    if (
        override is not None
        and attention_mask is not None
        and override.shape[-2] == query.shape[-2]
        and override.shape[-1] == attention_mask.shape[-1]
    ):
        attention_mask = override
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


ALL_ATTENTION_FUNCTIONS["sdpa_xmodal"] = sdpa_xmodal_forward

# Mask-builder registry (masking_utils.py): _preprocess_mask_arguments early-
# exits with a None mask when config._attn_implementation is absent from
# ALL_MASK_ATTENTION_FUNCTIONS._global_mapping (5.10.2, masking_utils.py:848),
# so decode under "sdpa_xmodal" would otherwise fall back to is_causal=True
# WITHOUT key-side padding — silently dropping padding on batched padded
# decode. Register the SAME builder stock "sdpa" uses (sdpa_mask): the
# non-4D path (decode steps, q_len 1) then builds the identical padding-aware
# causal mask, while the 4D-override loss path is handled by
# sdpa_xmodal_forward above (it ignores the passed mask when its shape guard
# matches). Prefill 4D bool masks still pass through create_causal_mask
# untouched.
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask

ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa_xmodal", sdpa_mask)
