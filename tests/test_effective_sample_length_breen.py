"""#4 — token-budget bucketing must count BREEN learnable-query expansion.

Each "<query>" sentinel is ONE input_ids token but the model splice expands it
into (learnable_query_num_fine + learnable_query_num_coarse) rows (one block per
image). effective_sample_length must subtract the 1-token sentinel and add the
expanded rows, else token-budget microbatches overshoot and land in the wrong
bucket (~19-31% over budget on the live BREEN path).
"""

import pytest

torch = pytest.importorskip("torch")

try:
    from vlm.data.data_arguments import DataArguments
    from vlm.data.energon_dataset import effective_sample_length
except ModuleNotFoundError as e:  # pragma: no cover - slim envs
    pytest.skip(f"vlm package not importable here: {e}", allow_module_level=True)

IMG = -200
AUD = -201
QRY = -202


def _img_entry(n_patches: int):
    # encoder-free 4-tuple: (patches (N, patch_dim), positions, size, modality)
    return (torch.zeros(n_patches, 6912), torch.zeros(n_patches, 2), (64, 48), "image")


def _data_dict(input_ids, images=None, audios=None):
    dd = {"input_ids": torch.tensor(input_ids, dtype=torch.long)}
    if images is not None:
        dd["image"] = images
    if audios is not None:
        dd["audio"] = audios
    return dd


def _args(**kw):
    return DataArguments(**kw)


def test_query_expansion_counted_when_enabled():
    # ids: 3 text + 1 image sentinel + 1 query sentinel = 5 tokens
    dd = _data_dict([1, 2, IMG, QRY, 3], images=[_img_entry(10)])
    da = _args(
        learnable_query_enabled=True, learnable_query_num_fine=64, learnable_query_num_coarse=36
    )
    # 5 - 1(img sentinel) - 1(query sentinel) + 10(img rows) + 100(query rows) = 113
    assert effective_sample_length(dd, da) == 113


def test_undercount_without_the_fix_is_avoided():
    # The pre-fix value (query counted as a single token, not expanded) would be
    # 5 - 1 + 10 = 14. The fix must NOT return that.
    dd = _data_dict([1, 2, IMG, QRY, 3], images=[_img_entry(10)])
    da = _args(learnable_query_enabled=True)
    assert effective_sample_length(dd, da) != 14
    assert effective_sample_length(dd, da) == 113  # 64+36 default rows


def test_multi_image_multi_query():
    # 2 images + 2 query blocks
    dd = _data_dict([1, IMG, QRY, 2, IMG, QRY], images=[_img_entry(8), _img_entry(12)])
    da = _args(
        learnable_query_enabled=True, learnable_query_num_fine=64, learnable_query_num_coarse=36
    )
    # 6 - 2(img) - 2(query) + (8+12) + 2*100 = 6 - 4 + 20 + 200 = 222
    assert effective_sample_length(dd, da) == 222


def test_custom_query_row_counts():
    dd = _data_dict([1, IMG, QRY], images=[_img_entry(5)])
    da = _args(
        learnable_query_enabled=True, learnable_query_num_fine=8, learnable_query_num_coarse=4
    )
    # 3 - 1 - 1 + 5 + (8+4) = 18
    assert effective_sample_length(dd, da) == 18


def test_no_query_block_when_disabled():
    # learnable_query disabled -> a stray -202 must NOT be expanded/subtracted.
    dd = _data_dict([1, 2, IMG, 3], images=[_img_entry(10)])
    da = _args(learnable_query_enabled=False)
    # 4 - 1 + 10 = 13 (unchanged baseline behavior)
    assert effective_sample_length(dd, da) == 13


def test_enabled_but_no_query_token_present():
    dd = _data_dict([1, 2, IMG, 3], images=[_img_entry(10)])
    da = _args(learnable_query_enabled=True)
    # no <query> sentinel in ids -> same as baseline: 4 - 1 + 10 = 13
    assert effective_sample_length(dd, da) == 13
