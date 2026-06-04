"""Tests for the encoder-free RawImageProcessor (gemma4_unified-style).

The three budget examples are the canonical fixtures from the pipeline
walkthrough (exp/gemma4_image_pipeline.html): they pin the resize math to
gemma4_unified's reference behavior.
"""

import importlib.util
import pathlib

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
PIL_Image = pytest.importorskip("PIL.Image")


def _load_module():
    """Import the processor module, tolerating partial environments.

    The normal package import (`vlm.models.image_processing_raw`) runs
    `vlm/__init__.py`, which pulls in hydra/deepspeed-era dependencies. The
    processor itself only needs torch/numpy/PIL/transformers, so in slim test
    environments we load it directly by file path instead.
    """
    try:
        from vlm.models import image_processing_raw

        return image_processing_raw
    except ModuleNotFoundError:
        path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "src/vlm/models/image_processing_raw.py"
        )
        spec = importlib.util.spec_from_file_location("image_processing_raw", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


_m = _load_module()
RawImageProcessor = _m.RawImageProcessor
convert_image_to_patches = _m.convert_image_to_patches
get_aspect_ratio_preserving_size = _m.get_aspect_ratio_preserving_size


# (width, height, budget) -> (target_w, target_h, expected_tokens)
BUDGET_EXAMPLES = [
    (800, 600, 280, 912, 672, 266),  # 4:3 — upscaled to fill budget
    (200, 150, 280, 912, 672, 266),  # same aspect, tiny image -> same output
    (1600, 100, 280, 3168, 192, 264),  # 16:1 extreme strip
]


@pytest.mark.parametrize("width,height,budget,exp_w,exp_h,exp_tokens", BUDGET_EXAMPLES)
def test_budget_examples(
    width: int, height: int, budget: int, exp_w: int, exp_h: int, exp_tokens: int
):
    proc = RawImageProcessor(patch_size=16, pooling_kernel_size=3, max_soft_tokens=budget)
    target_h, target_w = proc.get_target_size(height, width)
    assert (target_w, target_h) == (exp_w, exp_h)
    assert proc.get_num_patches(height, width) == exp_tokens
    assert exp_tokens <= budget


def test_end_to_end_shapes_and_positions():
    proc = RawImageProcessor(patch_size=16, pooling_kernel_size=3, max_soft_tokens=280)
    image = PIL_Image.new("RGB", (800, 600), color=(255, 0, 0))
    out = proc.preprocess(image)

    patches = out["pixel_values"][0]
    positions = out["image_position_ids"][0]
    assert patches.shape == (266, 48 * 48 * 3)
    assert positions.shape == (266, 2)
    assert out["num_patches_per_image"][0] == 266

    # Row-major (x, y): grid is 19 wide x 14 tall
    assert positions[0].tolist() == [0, 0]
    assert positions[1].tolist() == [1, 0]  # x advances first
    assert positions[19].tolist() == [0, 1]  # new row
    assert positions[:, 0].max().item() == 18
    assert positions[:, 1].max().item() == 13

    # Rescale-only by default: red pixels -> R channel 1.0, G/B 0.0
    assert torch.isclose(patches.max(), torch.tensor(1.0))
    assert patches.min().item() >= 0.0


def test_patch_content_matches_manual_slice():
    """First patch must equal the top-left 48x48 block flattened (rows, cols, C)."""
    proc = RawImageProcessor(max_soft_tokens=280)
    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(672, 912, 3), dtype=np.uint8)
    image = PIL_Image.fromarray(array)  # already budget-shaped: no resize change
    out = proc.preprocess(image)

    expected = torch.from_numpy(array[:48, :48, :].astype(np.float32) / 255.0).reshape(-1)
    assert torch.allclose(out["pixel_values"][0][0], expected)


def test_direct_cut_equals_teacher_merge_layout():
    """Deviation-3 guarantee: cutting model patches directly == gemma4's
    teacher-patchify + k×k merge, for the flattened element order."""
    k, ps = 3, 16
    side = k * ps
    tensor = torch.arange(3 * side * side, dtype=torch.float32).reshape(3, side, side)

    direct = convert_image_to_patches(tensor, side)  # (1, 6912)

    teacher = convert_image_to_patches(tensor, ps)  # (9, 768)
    # gemma4 merge layout: (y_k, ps_h, x_k, ps_w, c) flatten (see patches_merge L188-194)
    merged = teacher.reshape(k, k, ps, ps, 3).permute(0, 2, 1, 3, 4).reshape(1, -1)
    assert torch.equal(direct, merged)


def test_variable_aspect_batch():
    proc = RawImageProcessor(max_soft_tokens=280)
    images = [PIL_Image.new("RGB", (800, 600)), PIL_Image.new("RGB", (1600, 100))]
    out = proc.preprocess(images)
    assert out["num_patches_per_image"] == [266, 264]
    assert out["pixel_values"][0].shape[0] != out["pixel_values"][1].shape[0]


def test_arbitrary_budget_allowed():
    """No {70,140,280,560,1120} whitelist: any positive int budget works."""
    proc = RawImageProcessor(max_soft_tokens=100)
    n = proc.get_num_patches(600, 800)
    assert 0 < n <= 100


def test_degenerate_sizes():
    proc = RawImageProcessor(max_soft_tokens=280)
    # Extreme thin strip: one side clamps to one patch row, other capped
    n = proc.get_num_patches(10, 10000)
    assert 0 < n <= 280
    # Both dims rounding to zero raises (port of gemma4 L79-83)
    with pytest.raises(ValueError):
        get_aspect_ratio_preserving_size(1, 1, patch_size=16, max_patches=4, pooling_kernel_size=3)


def test_dummy_inputs_contract():
    proc = RawImageProcessor()
    patches, positions = proc.get_dummy_inputs()
    assert patches.shape == (1, proc.patch_dim)
    assert positions.shape == (1, 2)
    assert positions.dtype == torch.long


def test_normalization_dial():
    proc = RawImageProcessor(image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
    image = PIL_Image.new("RGB", (912, 672), color=(255, 255, 255))
    out = proc.preprocess(image)
    # (1.0 - 0.5) / 0.5 = 1.0 for white; black corners would be -1.0
    assert torch.isclose(out["pixel_values"][0].max(), torch.tensor(1.0))
