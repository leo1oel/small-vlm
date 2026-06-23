"""Fixed-canvas patchify / unpatchify + position grid for generation.

The forward patchify MUST match the repo's `convert_image_to_patches`
(image_processing_raw.py:94) so the connector + factorized 2D pos-embedding see
patches in the order they were trained on. unpatchify is its exact inverse (the
repo has none). Pure torch -> loaded by file path for fast TDD.
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


_gi = _load("src/vlm/models/gen_image.py", "gen_image_under_test")
pixels_to_patches = _gi.pixels_to_patches
patches_to_pixels = _gi.patches_to_pixels
make_position_ids = _gi.make_position_ids
assemble_generation_inputs = _gi.assemble_generation_inputs
# convert_image_to_patches lives in image_processing_raw (only external imports).
_ipr = _load("src/vlm/models/image_processing_raw.py", "ipr_under_test")
convert_image_to_patches = _ipr.convert_image_to_patches


def test_roundtrip_single():
    P = 48
    img = torch.randn(3, 2 * P, 3 * P)  # grid 2x3 -> 6 patches
    patches = pixels_to_patches(img, P)
    assert patches.shape == (6, P * P * 3)
    back = patches_to_pixels(patches, 2, 3, P)
    assert torch.allclose(back, img, atol=1e-6)


def test_roundtrip_batched():
    P = 16
    img = torch.randn(4, 3, 4 * P, 5 * P)  # grid 4x5 -> 20 patches
    patches = pixels_to_patches(img, P)
    assert patches.shape == (4, 20, P * P * 3)
    back = patches_to_pixels(patches, 4, 5, P)
    assert torch.allclose(back, img, atol=1e-6)


def test_forward_layout_matches_processor():
    P = 48
    img = torch.randn(3, 2 * P, 3 * P)
    mine = pixels_to_patches(img, P)
    ref = convert_image_to_patches(img, P)
    assert torch.equal(mine, ref)  # exact same patch ordering + flatten


def test_position_ids_formula():
    pos = make_position_ids(2, 3)  # gh=2, gw=3 -> N=6
    assert pos.shape == (6, 2)
    expected = torch.tensor([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]])
    assert torch.equal(pos, expected)  # k -> (x=k%gw, y=k//gw)


def test_position_ids_matches_processor_meshgrid():
    gh, gw = 3, 4
    pos = make_position_ids(gh, gw)
    xs, ys = torch.meshgrid(torch.arange(gw), torch.arange(gh), indexing="xy")
    ref = torch.stack((xs, ys), dim=-1).reshape(gh * gw, 2)
    assert torch.equal(pos, ref)


def test_assemble_no_pad():
    B, Lt, N, H = 1, 3, 2, 5
    text = torch.randn(B, Lt, H)
    timg = torch.randn(B, 1, H)
    img = torch.randn(B, N, H)
    text_mask = torch.ones(B, Lt)
    emb, prefix, image, pos = assemble_generation_inputs(text, text_mask, timg, img)
    L = Lt + 1 + N
    assert emb.shape == (B, L, H)
    assert torch.equal(prefix[0], torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.bool))
    assert torch.equal(image[0], torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.bool))
    assert torch.equal(pos[0], torch.tensor([0, 1, 2, 3, 4, 5]))  # structural arange
    # the image slice of inputs_embeds is exactly img_embeds
    assert torch.allclose(emb[:, Lt + 1 :], img)
    assert torch.allclose(emb[:, Lt : Lt + 1], timg)


def test_assemble_left_pad():
    B, Lt, N, H = 1, 3, 2, 4
    text = torch.randn(B, Lt, H)
    timg = torch.randn(B, 1, H)
    img = torch.randn(B, N, H)
    text_mask = torch.tensor([[0, 1, 1]])  # one left-pad token
    emb, prefix, image, pos = assemble_generation_inputs(text, text_mask, timg, img)
    # prefix excludes the padded text position (mask), but positions are
    # structural arange (independent of the mask) so CFG cond/uncond align.
    assert torch.equal(prefix[0], torch.tensor([0, 1, 1, 1, 0, 0], dtype=torch.bool))
    assert torch.equal(image[0], torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.bool))
    assert torch.equal(pos[0], torch.tensor([0, 1, 2, 3, 4, 5]))
