"""Truth-table tests for the cross-modal 4D mask builder (plan 2026-06-10).

Toy layout, right padding, L=7:
  pos:    0     1     2     3     4     5    6
  role: [sys] [img] [img] [q]  [ans] [ans] [pad]
labels:  -100  -100  -100  -100   7     8   -100
block:    -1    0     0    -1    -1    -1    -1
"""

import torch

from vlm.models.xmodal_mask import build_base_mask, build_cross_modal_mask

ATTN = torch.tensor([[1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool)
LABELS = torch.tensor([[-100, -100, -100, -100, 7, 8, -100]])
BLOCKS = torch.tensor([[-1, 0, 0, -1, -1, -1, -1]])


def allowed(mask, row, col):
    return bool(mask[0, 0, row, col])


def test_base_is_causal_with_key_padding():
    m = build_base_mask(ATTN)
    assert m.shape == (1, 1, 7, 7)
    assert allowed(m, 3, 0) and allowed(m, 3, 3)
    assert not allowed(m, 3, 4)  # causal
    assert not allowed(m, 5, 6)  # pad key always blocked


def test_prefix_lm_bidirectional_over_sys_img_question_only():
    m = build_cross_modal_mask(ATTN, None, LABELS, mode="prefix_lm")
    # prefix = positions 0..3 (before first supervised pos 4)
    assert allowed(m, 1, 3)  # img row sees question col (non-causal!)
    assert allowed(m, 0, 3)  # sys row sees question col
    assert allowed(m, 3, 1)  # question row sees img col (causal anyway)
    assert not allowed(m, 1, 4)  # img row must NOT see answer col
    assert not allowed(m, 3, 4)  # question row must NOT see answer col
    assert allowed(m, 4, 1) and allowed(m, 5, 4)  # answer rows: plain causal
    assert not allowed(m, 4, 5)  # answer rows stay causal


def test_img2q_window_adds_only_img_to_question_edges():
    m = build_cross_modal_mask(ATTN, BLOCKS, LABELS, mode="img2q_window")
    assert allowed(m, 1, 3) and allowed(m, 2, 3)  # img rows -> question col
    assert not allowed(m, 0, 3)  # sys row gets NO forward edge
    assert not allowed(m, 1, 4)  # img row -> answer col still blocked
    assert not allowed(m, 3, 4)  # question row unchanged (pure causal)
    # everything else identical to base
    base = build_base_mask(ATTN)
    diff = (m ^ base)[0, 0]
    rows, cols = diff.nonzero(as_tuple=True)
    assert set(rows.tolist()) <= {1, 2}
    assert set(cols.tolist()) <= {3}


def test_labels_none_treats_whole_prompt_as_prefix():
    # generation prefill: [img][img][q][q] no labels
    attn = torch.tensor([[1, 1, 1, 1]], dtype=torch.bool)
    blocks = torch.tensor([[0, 0, -1, -1]])
    m = build_cross_modal_mask(attn, blocks, None, mode="img2q_window")
    assert allowed(m, 0, 2) and allowed(m, 0, 3) and allowed(m, 1, 3)
    mp = build_cross_modal_mask(attn, None, None, mode="prefix_lm")
    assert allowed(mp, 0, 3) and allowed(mp, 2, 0)


def test_left_padding():
    attn = torch.tensor([[0, 1, 1, 1, 1]], dtype=torch.bool)  # pad at pos 0
    labels = torch.tensor([[-100, -100, -100, -100, 9]])
    blocks = torch.tensor([[-1, 0, 0, -1, -1]])
    m = build_cross_modal_mask(attn, blocks, labels, mode="img2q_window")
    assert allowed(m, 1, 3)  # img -> question forward edge
    assert not allowed(m, 1, 0)  # pad key blocked even for img rows
    assert not allowed(m, 1, 4)  # answer col blocked
    mp = build_cross_modal_mask(attn, None, labels, mode="prefix_lm")
    assert allowed(mp, 1, 3) and not allowed(mp, 1, 0)


def test_unknown_mode_raises():
    import pytest

    with pytest.raises(ValueError):
        build_cross_modal_mask(ATTN, BLOCKS, LABELS, mode="banana")


def test_img2q_window_needs_block_ids():
    import pytest

    with pytest.raises(ValueError):
        build_cross_modal_mask(ATTN, None, LABELS, mode="img2q_window")


def test_batch_mixed_prefix_lengths_and_padding_sides():
    """Two samples in one batch with different prefix boundaries and padding
    sides; vectorized prefix detection must be per-sample correct.

    L=6.
    sample 0 (right pad): [sys][img][q][ans][ans][pad]
        labels: -100 -100 -100  3    4  -100   -> first supervised = pos 3
        block:   -1   0   -1   -1   -1   -1     -> prefix = {0,1,2}
    sample 1 (left pad):  [pad][pad][img][img][q][ans]
        labels: -100 -100 -100 -100 -100  5    -> first supervised = pos 5
        attn:    0    0    1    1    1    1     -> prefix = {2,3,4}
        block:   -1   -1   2    2   -1   -1
    """
    attn = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    labels = torch.tensor(
        [
            [-100, -100, -100, 3, 4, -100],
            [-100, -100, -100, -100, -100, 5],
        ]
    )
    blocks = torch.tensor(
        [
            [-1, 0, -1, -1, -1, -1],
            [-1, -1, 2, 2, -1, -1],
        ]
    )

    # prefix_lm: bidirectional within each sample's own prefix only
    mp = build_cross_modal_mask(attn, None, labels, mode="prefix_lm")
    # sample 0 prefix {0,1,2}: img(1) sees q(2) forward; not answer(3)
    assert bool(mp[0, 0, 1, 2])
    assert not bool(mp[0, 0, 1, 3])
    # sample 0 prefix excludes pos 3+, so sys(0) must not forward-see pos 3
    assert not bool(mp[0, 0, 0, 3])
    # sample 1 prefix {2,3,4}: img(2) sees q(4) forward; not answer(5)
    assert bool(mp[1, 0, 2, 4])
    assert not bool(mp[1, 0, 2, 5])
    # sample 1 left-pad keys (0,1) always blocked even within prefix rows
    assert not bool(mp[1, 0, 4, 0]) and not bool(mp[1, 0, 4, 1])

    # img2q_window: img rows -> question (prefix & non-img) only
    mw = build_cross_modal_mask(attn, blocks, labels, mode="img2q_window")
    # sample 0: img(1) -> q(2)
    assert bool(mw[0, 0, 1, 2])
    assert not bool(mw[0, 0, 1, 3])  # answer blocked
    assert not bool(mw[0, 0, 0, 2])  # sys gets no forward edge
    # sample 1: img(2),img(3) -> q(4)
    assert bool(mw[1, 0, 2, 4]) and bool(mw[1, 0, 3, 4])
    assert not bool(mw[1, 0, 2, 5])  # answer blocked
    assert not bool(mw[1, 0, 2, 0])  # left-pad key blocked


def test_sdpa_xmodal_registered_and_swaps_per_module_mask():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    import vlm.models.xmodal_mask  # noqa: F401  (registration import)

    assert "sdpa_xmodal" in ALL_ATTENTION_FUNCTIONS

    fn = ALL_ATTENTION_FUNCTIONS["sdpa_xmodal"]

    class _Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("C", (), {"_attn_implementation": "sdpa_xmodal"})()
            self.layer_idx = 0
            self.num_key_value_groups = 1
            self.is_causal = True

    torch.manual_seed(0)
    m = _Stub()
    B, H, L, D = 1, 2, 4, 8
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    base = build_base_mask(torch.ones(B, L, dtype=torch.bool))
    full = torch.ones(B, 1, L, L, dtype=torch.bool)

    out_base, _ = fn(m, q, k, v, base, dropout=0.0, scaling=None)
    m._xmodal_mask = full
    out_full, _ = fn(m, q, k, v, base, dropout=0.0, scaling=None)
    assert not torch.allclose(out_base, out_full)  # override took effect

    # decode-step shape guard: q_len 1 != override L -> override ignored
    # (no exception from feeding the (B,1,1,L) row into the unmodified path).
    # sdpa_attention_forward returns (B, q_len, H, D), so q_len is axis 1.
    q1 = torch.randn(B, H, 1, D)
    row = torch.ones(B, 1, 1, L, dtype=torch.bool)
    out_dec, _ = fn(m, q1, k, v, row, dropout=0.0, scaling=None)
    assert out_dec.shape[1] == 1


def test_sdpa_xmodal_mask_builder_registered():
    """The non-4D decode path goes through ALL_MASK_ATTENTION_FUNCTIONS;
    sdpa_xmodal must resolve to the same builder as stock sdpa so batched
    padded decode keeps key-side padding (masking_utils.py:848 early-exits to
    a None mask if the impl is absent)."""
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

    import vlm.models.xmodal_mask  # noqa: F401  (registration import)

    assert "sdpa_xmodal" in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping
    assert ALL_MASK_ATTENTION_FUNCTIONS["sdpa_xmodal"] is ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]


def test_gen_mask_stash_shape_contract():
    import torch

    gen_mask = torch.ones(1, 1, 6, 6, dtype=torch.bool)
    embeds_prefill = torch.zeros(1, 6, 4)
    embeds_decode = torch.zeros(1, 1, 4)
    assert embeds_prefill.shape[1] == gen_mask.shape[-2]
    assert embeds_decode.shape[1] != gen_mask.shape[-2]


def test_install_xmodal_masks_window_stash():
    """install_xmodal_masks stashes the windowed mask on layers lo..hi only."""
    import types

    import torch

    from vlm.models import modeling_vlm

    class Attn:
        pass

    class Layer:
        def __init__(self):
            self.self_attn = Attn()

    fake = types.SimpleNamespace(
        config=types.SimpleNamespace(
            cross_modal_mask_mode="img2q_window",
            cross_modal_mask_window=[1, 2],
            ignore_index=-100,
        ),
        model=types.SimpleNamespace(layers=[Layer(), Layer(), Layer()]),
    )
    attn = torch.ones(1, 5, dtype=torch.bool)
    labels = torch.tensor([[-100, -100, -100, 4, 5]])
    blocks = torch.tensor([[-1, 0, -1, -1, -1]])

    out = modeling_vlm.install_xmodal_masks(fake, attn, blocks, labels)
    assert out.shape == (1, 1, 5, 5)
    assert fake.model.layers[0].self_attn._xmodal_mask is not None
    assert fake.model.layers[1].self_attn._xmodal_mask is not None
    assert fake.model.layers[2].self_attn._xmodal_mask is None
    # prefix_lm returns the merged mask and does not touch layers.
    # labels [-100,-100,-100,4,5] -> first supervised pos = 3, prefix = {0,1,2}.
    fake.config.cross_modal_mask_mode = "prefix_lm"
    out2 = modeling_vlm.install_xmodal_masks(fake, attn, None, labels)
    assert out2.shape == (1, 1, 5, 5)
    assert bool(out2[0, 0, 0, 2])  # genuinely non-causal prefix edge: row 0 sees col 2
    assert not bool(out2[0, 0, 0, 3])  # prefix row does not see the supervised suffix


def test_cross_modal_mask_config_defaults():
    from vlm.config.config_schema import CrossModalMaskConfig

    c = CrossModalMaskConfig()
    assert c.mode == "none" and c.window == [1, 9] and c.bidirectional is False


def _validate(mode, attn, window=(1, 9), bidirectional=False, n_layers=28):
    from vlm.train.train import validate_cross_modal_mask_config

    return validate_cross_modal_mask_config(
        mode,
        list(window),
        bidirectional,
        attn_implementation=attn,
        num_hidden_layers=n_layers,
    )


def test_cross_modal_mask_validation():
    import pytest

    # mode none passes regardless of attn (bit-identical baseline path)
    assert _validate("none", "flash_attention_2") == ("none", [1, 9])
    assert _validate(None, "sdpa") == ("none", [1, 9])

    # unknown mode raises
    with pytest.raises(ValueError, match="must be none"):
        _validate("banana", "sdpa")

    # bidirectional=True is rejected (not implemented in v1)
    with pytest.raises(ValueError, match="bidirectional"):
        _validate("prefix_lm", "sdpa", bidirectional=True)

    # prefix_lm needs sdpa/sdpa_xmodal — flash_attention_2 raises
    with pytest.raises(ValueError, match="prefix_lm needs"):
        _validate("prefix_lm", "flash_attention_2")
    assert _validate("prefix_lm", "sdpa") == ("prefix_lm", [1, 9])
    assert _validate("prefix_lm", "sdpa_xmodal") == ("prefix_lm", [1, 9])

    # img2q_window needs sdpa_xmodal specifically — plain sdpa raises
    with pytest.raises(ValueError, match="img2q_window needs"):
        _validate("img2q_window", "sdpa")

    # window bounds: [0, 9] (lo < 1) and [1, 99] (hi > n_layers) raise
    with pytest.raises(ValueError, match="out of range"):
        _validate("img2q_window", "sdpa_xmodal", window=(0, 9))
    with pytest.raises(ValueError, match="out of range"):
        _validate("img2q_window", "sdpa_xmodal", window=(1, 99), n_layers=28)
    # inverted window (lo > hi) also out of range
    with pytest.raises(ValueError, match="out of range"):
        _validate("img2q_window", "sdpa_xmodal", window=(9, 1))

    # valid img2q_window combos pass and round-trip the window
    assert _validate("img2q_window", "sdpa_xmodal", window=(1, 9)) == ("img2q_window", [1, 9])
    assert _validate("img2q_window", "sdpa_xmodal", window=(1, 28), n_layers=28) == (
        "img2q_window",
        [1, 28],
    )


# ---------------------------------------------------------------------------
# #6 — prefix detection robust to Qwen ChatML delimiter unmasking
# ---------------------------------------------------------------------------

IM_START = 151644  # Qwen <|im_start|>
IM_END = 151645  # Qwen <|im_end|>


def test_prefix_skip_ids_under_qwen_delimiter_unmasking():
    """Qwen `preprocess_qwen` unmasks ChatML delimiters EVERYWHERE, so the
    leading system <|im_start|> at position 0 becomes "supervised". Without
    skip ids the prefix collapses to empty (the arm is a silent no-op);
    `prefix_skip_ids` excludes delimiters so the boundary lands on the first
    real answer token and the prompt (image + question) stays in the prefix.

    layout (L=7): [imS(sys) | img | q | imE | imS(asst) | ans | imE]
    """
    attn = torch.ones(1, 7, dtype=torch.bool)
    labels = torch.tensor([[IM_START, -100, -100, IM_END, IM_START, 42, IM_END]])
    blocks = torch.tensor([[-1, 0, -1, -1, -1, -1, -1]])

    # BUG reproduction: no skip ids -> first supervised label = pos 0 -> empty
    # prefix -> mask identical to plain causal (the cross-modal edges vanish).
    buggy = build_cross_modal_mask(attn, blocks, labels, mode="img2q_window")
    assert torch.equal(buggy, build_base_mask(attn))
    buggy_p = build_cross_modal_mask(attn, None, labels, mode="prefix_lm")
    assert torch.equal(buggy_p, build_base_mask(attn))

    # FIX: skip the delimiters -> first answer = pos 5 -> prefix = {0,1,2,3,4}.
    skip = [IM_START, IM_END]
    fixed = build_cross_modal_mask(attn, blocks, labels, mode="img2q_window", prefix_skip_ids=skip)
    assert allowed(fixed, 1, 2)  # image row -> question key (non-causal edge)
    assert allowed(fixed, 1, 0)  # image row -> system prefix key (prefix kept)
    assert not allowed(fixed, 1, 5)  # answer content key still blocked

    fixed_p = build_cross_modal_mask(attn, None, labels, mode="prefix_lm", prefix_skip_ids=skip)
    assert allowed(fixed_p, 1, 2) and allowed(fixed_p, 2, 1)  # bidirectional img<->q
    assert not allowed(fixed_p, 1, 5)  # answer not in the prefix


def test_prefix_skip_ids_noop_for_clean_templates():
    """With no delimiter pollution the first supervised label already IS the
    first answer token, so passing skip ids changes nothing."""
    no_skip = build_cross_modal_mask(ATTN, BLOCKS, LABELS, mode="img2q_window")
    with_skip = build_cross_modal_mask(
        ATTN, BLOCKS, LABELS, mode="img2q_window", prefix_skip_ids=[IM_START, IM_END]
    )
    assert torch.equal(no_skip, with_skip)


def test_install_xmodal_masks_reads_config_prefix_skip_ids():
    """install_xmodal_masks threads config.cross_modal_prefix_skip_ids into the
    prefix computation, so the Qwen-delimiter collapse is fixed end to end."""
    import types

    from vlm.models import modeling_vlm

    class Attn:
        pass

    class Layer:
        def __init__(self):
            self.self_attn = Attn()

    attn = torch.ones(1, 6, dtype=torch.bool)
    # [imS(sys) | img | q | imE | ans | imE]
    labels = torch.tensor([[IM_START, -100, -100, IM_END, 7, IM_END]])

    def _fake(skip_ids):
        return types.SimpleNamespace(
            config=types.SimpleNamespace(
                cross_modal_mask_mode="prefix_lm",
                cross_modal_mask_window=[1, 2],
                ignore_index=-100,
                cross_modal_prefix_skip_ids=skip_ids,
            ),
            model=types.SimpleNamespace(layers=[Layer(), Layer()]),
        )

    out = modeling_vlm.install_xmodal_masks(_fake([IM_START, IM_END]), attn, None, labels)
    assert bool(out[0, 0, 0, 2])  # row 0 (system) sees question col 2: prefix kept
    out2 = modeling_vlm.install_xmodal_masks(_fake(None), attn, None, labels)
    assert not bool(out2[0, 0, 0, 2])  # without skip ids the prefix collapses


# ---------------------------------------------------------------------------
# #29 — img2q_window must not treat BREEN learnable-query rows as question text
# ---------------------------------------------------------------------------


def test_img2q_window_excludes_query_rows_from_question_keys():
    """BREEN learnable-query rows live in the prefix and are not image, so the
    plain `prefix & ~is_img` question-text set wrongly includes them. Passing
    query_block_ids excludes those columns from the image->question edges.

    layout (L=6): [img | q0 | q1 | query | ans | pad]
      image_block:   0   -1   -1    -1     -1   -1
      query_block:  -1   -1   -1     0     -1   -1
    """
    attn = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    labels = torch.tensor([[-100, -100, -100, -100, 9, -100]])
    image_block = torch.tensor([[0, -1, -1, -1, -1, -1]])
    query_block = torch.tensor([[-1, -1, -1, 0, -1, -1]])

    # Without query ids: image row -> the query column (pos 3) is wrongly added.
    without = build_cross_modal_mask(attn, image_block, labels, mode="img2q_window")
    assert allowed(without, 0, 3)  # the bug: image attends the learnable-query row

    # With query ids: the query column is excluded; real question keys stay.
    fixed = build_cross_modal_mask(
        attn, image_block, labels, mode="img2q_window", query_block_ids=query_block
    )
    assert allowed(fixed, 0, 1) and allowed(fixed, 0, 2)  # real question text kept
    assert not allowed(fixed, 0, 3)  # learnable-query row no longer a question key
