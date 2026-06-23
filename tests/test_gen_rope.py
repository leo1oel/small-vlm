"""Leaf tests for the 2D MRoPE generation rotary (importlib direct-load to skip
the ~90s `import vlm`). Verifies BIT-IDENTITY with stock Qwen3RotaryEmbedding on
1D positions (so understanding/text is unchanged) and correct axial behaviour."""

import importlib.util
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen_rope = _load("gen_rope", "src/vlm/models/gen_rope.py")


def _qwen3_rotary():
    from transformers import AutoConfig
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-1.7B")
    return Qwen3RotaryEmbedding(cfg), cfg


def test_1d_bit_identical_to_stock_qwen3():
    base, _ = _qwen3_rotary()
    mrope = gen_rope.Gen2DRotaryEmbedding(base)
    x = torch.randn(2, 7, 8)  # (B,L,*) dummy for dtype/device
    pos = torch.stack([torch.arange(7), torch.arange(7)])  # (B=2, L=7)
    cos_b, sin_b = base(x, pos)
    cos_m, sin_m = mrope(x, pos)
    assert torch.equal(cos_b, cos_m), (cos_b - cos_m).abs().max()
    assert torch.equal(sin_b, sin_m), (sin_b - sin_m).abs().max()


def test_equal_axes_reduces_to_1d():
    base, _ = _qwen3_rotary()
    mrope = gen_rope.Gen2DRotaryEmbedding(base)
    x = torch.randn(1, 5, 8)
    pos1d = torch.arange(5)[None, :]  # (1,5)
    pos3 = pos1d[None].expand(3, 1, 5).contiguous()  # all axes equal
    cos1, sin1 = mrope(x, pos1d)
    mrope._mrope_positions = pos3
    cos3, sin3 = mrope(x, pos1d)  # passed pos ignored; stash used
    assert torch.equal(cos1, cos3)
    assert torch.equal(sin1, sin3)


def test_axial_distinguishes_h_and_w():
    base, _ = _qwen3_rotary()
    mrope = gen_rope.Gen2DRotaryEmbedding(base)
    x = torch.randn(1, 4, 8)
    # two tokens: (h=5,w=5) vs (h=5,w=9) -> only w differs -> cos must differ
    pos = torch.tensor([[[5, 5]], [[5, 5]], [[5, 9]]])  # (3,B=1,L=2)? build manually
    pos = torch.zeros(3, 1, 2, dtype=torch.long)
    pos[:, 0, 0] = torch.tensor([5, 5, 5])  # token0 t=h=w=5
    pos[:, 0, 1] = torch.tensor([5, 5, 9])  # token1 t=5,h=5,w=9 (w shifted)
    mrope._mrope_positions = pos
    cos, _ = mrope(x[:, :2], torch.zeros(1, 2, dtype=torch.long))
    assert not torch.allclose(cos[0, 0], cos[0, 1]), "w shift must change rotation"


def test_build_mrope_position_ids_shape_and_values():
    pos = gen_rope.build_mrope_position_ids(prefix_len=3, grid_h=2, grid_w=2, batch=2, device=torch.device("cpu"))
    assert pos.shape == (3, 2, 7)  # 3 prefix + 4 image
    # prefix: all axes equal arange
    assert torch.equal(pos[0, 0, :3], torch.tensor([0, 1, 2]))
    assert torch.equal(pos[1, 0, :3], torch.tensor([0, 1, 2]))
    assert torch.equal(pos[2, 0, :3], torch.tensor([0, 1, 2]))
    # image (prefix_len=3): t const=3; h=3+y; w=3+x; row-major k=0..3
    assert torch.equal(pos[0, 0, 3:], torch.tensor([3, 3, 3, 3]))          # t
    assert torch.equal(pos[1, 0, 3:], torch.tensor([3, 3, 4, 4]))          # h = 3 + k//2
    assert torch.equal(pos[2, 0, 3:], torch.tensor([3, 4, 3, 4]))          # w = 3 + k%2


def test_mrope_section_sums_to_d2():
    base, _ = _qwen3_rotary()
    mrope = gen_rope.Gen2DRotaryEmbedding(base)
    assert sum(mrope.mrope_section) == mrope.inv_freq.shape[0]
