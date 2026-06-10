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
    assert not allowed(m, 3, 4)          # causal
    assert not allowed(m, 5, 6)          # pad key always blocked


def test_prefix_lm_bidirectional_over_sys_img_question_only():
    m = build_cross_modal_mask(ATTN, None, LABELS, mode="prefix_lm")
    # prefix = positions 0..3 (before first supervised pos 4)
    assert allowed(m, 1, 3)      # img row sees question col (non-causal!)
    assert allowed(m, 0, 3)      # sys row sees question col
    assert allowed(m, 3, 1)      # question row sees img col (causal anyway)
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
    attn = torch.tensor([[0, 1, 1, 1, 1]], dtype=torch.bool)   # pad at pos 0
    labels = torch.tensor([[-100, -100, -100, -100, 9]])
    blocks = torch.tensor([[-1, 0, 0, -1, -1]])
    m = build_cross_modal_mask(attn, blocks, labels, mode="img2q_window")
    assert allowed(m, 1, 3)       # img -> question forward edge
    assert not allowed(m, 1, 0)   # pad key blocked even for img rows
    assert not allowed(m, 1, 4)   # answer col blocked
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
    assert not bool(mw[0, 0, 1, 3])      # answer blocked
    assert not bool(mw[0, 0, 0, 2])      # sys gets no forward edge
    # sample 1: img(2),img(3) -> q(4)
    assert bool(mw[1, 0, 2, 4]) and bool(mw[1, 0, 3, 4])
    assert not bool(mw[1, 0, 2, 5])      # answer blocked
    assert not bool(mw[1, 0, 2, 0])      # left-pad key blocked
