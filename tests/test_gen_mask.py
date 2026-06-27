"""Generation attention mask (text->image flow matching).

Layout per sample: [text tokens ... | timestep token | noised image patches].
Required edges (bool, True = attend): text+t bidirectional prefix; image block
bidirectional; image attends the prefix; prefix does NOT attend image (so its KV
is reusable across sampler steps); every row attends itself (pad NaN guard).
"""

import importlib.util
import pathlib

import torch


def _load(relpath, name):
    p = pathlib.Path(__file__).resolve().parents[1] / relpath
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


build_generation_mask = _load(
    "src/vlm/models/xmodal_mask.py", "xmodal_mask_under_test"
).build_generation_mask


def test_generation_mask_edges():
    # L=6: positions 0,1 text; 2 t-token; 3,4,5 image
    prefix = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    image = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.bool)
    m = build_generation_mask(prefix, image)
    assert m.shape == (1, 1, 6, 6)
    a = lambda q, k: bool(m[0, 0, q, k])
    # text <-> text bidirectional
    assert a(0, 1) and a(1, 0)
    # text/prefix does NOT attend image
    assert not a(0, 3) and not a(2, 4)
    # image <-> image bidirectional
    assert a(3, 5) and a(5, 3)
    # image attends prefix (text + t-token)
    assert a(3, 0) and a(4, 2)
    # diagonal everywhere
    for i in range(6):
        assert a(i, i)


def test_generation_mask_pad_row_self_only():
    # L=4: 0,1 text; 2 image; 3 PAD
    prefix = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    image = torch.tensor([[0, 0, 1, 0]], dtype=torch.bool)
    m = build_generation_mask(prefix, image)
    assert bool(m[0, 0, 3, 3])  # self (NaN guard)
    assert not bool(m[0, 0, 3, 0]) and not bool(m[0, 0, 3, 2])
    assert int(m[0, 0, 3].sum()) == 1  # pad row attends only itself
    # nobody attends the pad column (no real query -> pad key)
    assert int(m[0, 0, :, 3].sum()) == 1  # only the pad row's own diagonal


def test_generation_mask_batch():
    prefix = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    image = torch.tensor([[0, 0, 1], [0, 1, 0]], dtype=torch.bool)
    m = build_generation_mask(prefix, image)
    assert m.shape == (2, 1, 3, 3)
    assert bool(m[1, 0, 1, 0])  # sample 1: image(1) attends prefix(0)
    assert not bool(m[1, 0, 0, 1])  # sample 1: prefix(0) !attend image(1)
