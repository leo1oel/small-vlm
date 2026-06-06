"""CPU unit tests for the visual-aux pair construction + target prep helpers
(spec: docs/superpowers/specs/2026-06-06-visual-aux-loss-design.md §2-3)."""

import torch

from vlm.models.modeling_vlm import build_visual_aux_pairs, prepare_visual_aux_targets


def _row(spec: list[tuple[int, int]], length: int) -> torch.Tensor:
    """spec = [(block_id, n_positions), ...] laid out left to right; -1 fills."""
    row = torch.full((length,), -1, dtype=torch.long)
    cursor = 0
    for block_id, n in spec:
        row[cursor : cursor + n] = block_id
        cursor += n
    return row


def test_pairs_within_blocks_only():
    # row 0: text(2) | img0 patches(4) | text(1) | img1 patches(3) | pad(2)
    ids = torch.stack(
        [
            _row([(-1, 2), (0, 4), (-1, 1), (1, 3)], 12),
            _row([(-1, 12)], 12),  # text-only row (dummy image consumed zero-width)
        ]
    )
    flat_pos, segments = build_visual_aux_pairs(ids, num_target_rows=[4, 3])
    # img0: positions 2,3,4 predict rows 1,2,3 (position 5 = last patch, no target)
    # img1: positions 7,8 predict rows 1,2
    assert segments == [(0, 3), (1, 2)]
    assert flat_pos.tolist() == [2, 3, 4, 7, 8]


def test_pairs_second_batch_row_offset():
    L = 8
    ids = torch.stack([_row([(-1, L)], L), _row([(0, 5), (-1, 3)], L)])
    flat_pos, segments = build_visual_aux_pairs(ids, num_target_rows=[5])
    assert segments == [(0, 4)]
    assert flat_pos.tolist() == [L + 0, L + 1, L + 2, L + 3]


def test_truncated_block_keeps_valid_pairs():
    # img0 has 6 target rows but only 2 positions survived truncation:
    # both surviving positions still have real next-row targets (rows 1, 2).
    ids = _row([(-1, 1), (0, 2)], 3).unsqueeze(0)
    flat_pos, segments = build_visual_aux_pairs(ids, num_target_rows=[6])
    assert segments == [(0, 2)]
    assert flat_pos.tolist() == [1, 2]


def test_single_patch_and_empty():
    ids = _row([(0, 1), (-1, 4)], 5).unsqueeze(0)
    flat_pos, segments = build_visual_aux_pairs(ids, num_target_rows=[1])
    assert segments == [] and flat_pos.numel() == 0
    flat_pos, segments = build_visual_aux_pairs(
        torch.full((2, 5), -1, dtype=torch.long), num_target_rows=[]
    )
    assert segments == [] and flat_pos.numel() == 0


def test_left_padding_offsets():
    # left-padded row: pad(3) | img0(4) | text(1)
    ids = _row([(-1, 3), (0, 4), (-1, 1)], 8).unsqueeze(0)
    flat_pos, segments = build_visual_aux_pairs(ids, num_target_rows=[4])
    assert segments == [(0, 3)]
    assert flat_pos.tolist() == [3, 4, 5]


def test_aim_pixel_targets_zscore_and_shift():
    torch.manual_seed(0)
    img = torch.randn(4, 12, dtype=torch.float32) * 3.0 + 1.5
    tgt = prepare_visual_aux_targets("aim_pixel", [img], [(0, 3)])
    assert tgt.shape == (3, 12) and tgt.dtype == torch.float32
    # rows are img[1:4], each z-scored with the MAE formula (unbiased var, eps 1e-6)
    ref = img[1:4]
    ref = (ref - ref.mean(-1, keepdim=True)) / (ref.var(-1, keepdim=True) + 1e-6).sqrt()
    assert torch.allclose(tgt, ref, atol=1e-6)
    # shift guard: target row 0 is img[1], NOT img[0]
    assert not torch.allclose(tgt[0], (img[0] - img[0].mean()) / (img[0].var() + 1e-6).sqrt())


def test_nepa_targets_detached_and_normalized():
    e = torch.randn(5, 8, requires_grad=True)
    tgt = prepare_visual_aux_targets("nepa", [e], [(0, 4)])
    assert tgt.requires_grad is False  # stop-grad: MANDATORY (collapse guard)
    assert tgt.shape == (4, 8)
    assert torch.allclose(tgt.norm(dim=-1), torch.ones(4), atol=1e-5)
    ref = torch.nn.functional.normalize(e[1:5].detach().float(), dim=-1)
    assert torch.allclose(tgt, ref, atol=1e-6)


def test_multi_image_target_concat_order():
    a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    b = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 100
    tgt = prepare_visual_aux_targets("nepa", [a, b], [(0, 2), (1, 1)])
    assert tgt.shape == (3, 4)
    raw = torch.cat([a[1:3], b[1:2]]).float()
    ref = torch.nn.functional.normalize(raw, dim=-1)
    assert torch.allclose(tgt, ref, atol=1e-6)


def test_unknown_objective_raises():
    import pytest

    with pytest.raises(ValueError, match="unknown visual_aux objective"):
        prepare_visual_aux_targets("flow_matching", [torch.ones(2, 4)], [(0, 1)])
