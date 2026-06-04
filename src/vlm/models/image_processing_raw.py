"""Encoder-free (gemma4_unified-style) raw-patch image processor.

Faithful port of the gemma4_unified image pipeline in transformers 5.10.1
(models/gemma4_unified/image_processing_gemma4_unified.py), with three
deliberate deviations for the LLaVA-splice integration in this repo:

  1. No padding to the token budget. Outputs are variable-length per image —
     the LLaVA-style splice in modeling_vlm handles a variable number of image
     features natively, so gemma4's pad-to-budget + "-1 position sentinel"
     protocol is unnecessary here.
  2. ``max_soft_tokens`` accepts any positive integer (gemma4 enforces the
     whitelist {70, 140, 280, 560, 1120}, tied to its own checkpoint).
  3. Patchification cuts model patches (``patch_size * pooling_kernel_size``
     px) directly instead of gemma4's teacher-patchify + k×k merge two-step.
     This is byte-identical: gemma4's patches_merge permutation
     (image_processing L188-194) flattens each merged patch as
     (rows, cols, channels) row-major over the (k*patch_size)² pixel block —
     exactly what a direct big-block patchify yields. Index check for k=3,
     patch=16: merge gives y_k*2304 + p*144 + x_k*48 + q*3 + c, direct gives
     (y_k*16+p)*144 + (x_k*16+q)*3 + c — identical.

Position-id convention (must stay in sync with RawPatchConnector's factorized
table lookup): positions are integer (x, y) pairs with x = column index and
y = row index of the model-patch grid, row-major order — matching gemma4's
``torch.meshgrid(arange(W_p), arange(H_p), indexing="xy")`` (L329-334).

Resize note: gemma4 uses torchvision's bicubic with antialias; we use PIL
bicubic (already antialiased) to avoid a torchvision dependency. Numerics
differ at the last decimal but the embedder is trained from scratch.
"""

import logging
import math
from typing import Any, override

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from transformers.image_processing_base import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor

log: logging.Logger = logging.getLogger(__name__)


def get_aspect_ratio_preserving_size(
    height: int,
    width: int,
    patch_size: int,
    max_patches: int,
    pooling_kernel_size: int,
) -> tuple[int, int]:
    """Largest (height, width) that fits the patch budget, preserves aspect
    ratio, and is divisible by ``pooling_kernel_size * patch_size``.

    Verbatim port of gemma4_unified's get_aspect_ratio_preserving_size
    (image_processing L54-105), including the degenerate-dimension clamps.
    """
    total_px = height * width
    target_px = max_patches * (patch_size**2)
    factor = math.sqrt(target_px / total_px)
    ideal_height = factor * height
    ideal_width = factor * width
    side_mult = pooling_kernel_size * patch_size

    # Round down to nearest multiple of side_mult
    target_height = int(math.floor(ideal_height / side_mult)) * side_mult
    target_width = int(math.floor(ideal_width / side_mult)) * side_mult

    # Handle edge cases where one or both dimensions round to 0
    if target_height == 0 and target_width == 0:
        raise ValueError(
            "Attempting to resize to a 0 x 0 image. Resized height should be divisible by "
            f"`pooling_kernel_size * patch_size`={side_mult}."
        )

    max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(int(math.floor(width / height)) * side_mult, max_side_length)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(int(math.floor(height / width)) * side_mult, max_side_length)

    if target_height * target_width > target_px:
        raise ValueError(
            f"Resizing [{height}x{width}] to [{target_height}x{target_width}] "
            f"but this exceeds {max_patches} patches with patch_size {patch_size}"
        )

    return target_height, target_width


def convert_image_to_patches(image: Tensor, patch_size: int) -> Tensor:
    """(C, H, W) -> (num_patches_h * num_patches_w, patch_size² * C).

    Row-major grid; each patch flattened as (rows, cols, channels).
    Verbatim port of gemma4_unified's convert_image_to_patches (L108-119).
    """
    num_channels, image_height, image_width = image.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched_image = image.reshape(
        num_channels, num_patches_height, patch_size, num_patches_width, patch_size
    )
    patched_image = patched_image.permute(1, 3, 2, 4, 0)
    return patched_image.reshape(num_patches_height * num_patches_width, -1)


class RawImageProcessor(BaseImageProcessor):
    """Variable-resolution raw-patch processor for the encoder-free VLM path.

    Per image, produces:
      - ``pixel_values``: (N, model_patch_size² * 3) float32 raw-RGB patches
      - ``image_position_ids``: (N, 2) int64 (x, y) model-grid coordinates
    where N varies with aspect ratio (N <= max_soft_tokens). Outputs are
    returned as *lists* (one entry per image) — shapes differ across images,
    so they are never stacked; the collator passes them through as lists.

    Tunables (the "vision dials", all yaml-configurable):
      patch_size:           teacher patch edge, px (gemma4 default 16)
      pooling_kernel_size:  k; model patch = patch_size * k px (gemma4: 3 -> 48px)
      max_soft_tokens:      per-image token budget, any positive int (gemma4: 280)
      image_mean/image_std: optional per-channel normalize AFTER rescale to [0,1];
                            default None/None = rescale-only, like gemma4
                            (image_mean=[0,0,0], image_std=[1,1,1])
    """

    model_input_names: list[str] = ["pixel_values", "image_position_ids"]

    def __init__(
        self,
        patch_size: int = 16,
        pooling_kernel_size: int = 3,
        max_soft_tokens: int = 280,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if patch_size < 1 or pooling_kernel_size < 1 or max_soft_tokens < 1:
            raise ValueError(
                f"RawImageProcessor: patch_size ({patch_size}), pooling_kernel_size "
                f"({pooling_kernel_size}) and max_soft_tokens ({max_soft_tokens}) must all be >= 1."
            )
        self.patch_size: int = patch_size
        self.pooling_kernel_size: int = pooling_kernel_size
        self.max_soft_tokens: int = max_soft_tokens
        self.image_mean: list[float] | None = image_mean
        self.image_std: list[float] | None = image_std
        # Compatibility attribute: dataset.py's text-only branch reads
        # image_processor.crop_size (or .size) to build a dummy input. The raw
        # path uses get_dummy_inputs() instead, but keep the attribute so any
        # generic code path doesn't AttributeError.
        self.crop_size: dict[str, int] = {
            "height": self.model_patch_size,
            "width": self.model_patch_size,
        }

    @property
    def model_patch_size(self) -> int:
        """Model patch edge in pixels (= patch_size * pooling_kernel_size)."""
        return self.patch_size * self.pooling_kernel_size

    @property
    def patch_dim(self) -> int:
        """Flattened model-patch feature dimension (= model_patch_size² * 3)."""
        return self.model_patch_size**2 * 3

    def get_target_size(self, height: int, width: int) -> tuple[int, int]:
        """Resize target for an image of (height, width), per the patch budget."""
        return get_aspect_ratio_preserving_size(
            height=height,
            width=width,
            patch_size=self.patch_size,
            max_patches=self.max_soft_tokens * self.pooling_kernel_size**2,
            pooling_kernel_size=self.pooling_kernel_size,
        )

    def get_num_patches(self, height: int, width: int) -> int:
        """Soft-token count an image of (height, width) will produce."""
        target_h, target_w = self.get_target_size(height, width)
        return (target_h // self.model_patch_size) * (target_w // self.model_patch_size)

    def get_dummy_inputs(self) -> tuple[Tensor, Tensor]:
        """Zero-image stand-in for text-only samples: 1 black patch at (0, 0).

        Keeps the batch's image list non-empty so the encoder-free branch and
        the splice's no-image path (modeling_vlm L386-391, which consumes one
        feature set per sample and appends features[0:0]) work unchanged.
        """
        return (
            torch.zeros(1, self.patch_dim, dtype=torch.float32),
            torch.zeros(1, 2, dtype=torch.long),
        )

    def _process_one(self, image: "Image.Image | Tensor") -> tuple[Tensor, Tensor]:
        """PIL image (or (C, H, W) uint8/float tensor) -> (patches, position_ids)."""
        if isinstance(image, Tensor):
            if image.ndim != 3:
                raise ValueError(
                    f"RawImageProcessor: expected (C, H, W) tensor, got {tuple(image.shape)}"
                )
            # Round-trip through PIL for the resize so both input types share
            # one code path (and one resampling implementation).
            array = image.permute(1, 2, 0).cpu().numpy()
            if array.dtype != np.uint8:
                array = (array.clip(0.0, 1.0) * 255.0).astype(np.uint8)
            image = Image.fromarray(array)
        image = image.convert("RGB")

        width, height = image.size  # PIL convention: (W, H)
        target_h, target_w = self.get_target_size(height, width)
        image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)

        array = np.asarray(image, dtype=np.float32) / 255.0  # (H, W, 3), rescale
        if self.image_mean is not None and self.image_std is not None:
            array = (array - np.array(self.image_mean, dtype=np.float32)) / np.array(
                self.image_std, dtype=np.float32
            )
        tensor = torch.from_numpy(array).permute(2, 0, 1)  # (3, H, W)

        # Direct model-patch cut — byte-identical to gemma4's teacher+merge
        # (see module docstring, deviation 3).
        patches = convert_image_to_patches(tensor, self.model_patch_size)

        grid_h = target_h // self.model_patch_size
        grid_w = target_w // self.model_patch_size
        # (x, y) with x = column, y = row; row-major — matches gemma4's
        # meshgrid(arange(W_p), arange(H_p), indexing="xy") (L329-334) and the
        # RawPatchConnector lookup table[pos[:, 0], 0] + table[pos[:, 1], 1].
        xs, ys = torch.meshgrid(
            torch.arange(grid_w, dtype=torch.long),
            torch.arange(grid_h, dtype=torch.long),
            indexing="xy",
        )
        position_ids = torch.stack((xs, ys), dim=-1).reshape(patches.shape[0], 2)

        return patches, position_ids

    @override
    def preprocess(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        images: "Image.Image | Tensor | list[Image.Image | Tensor]",
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> BatchFeature:
        """Process one image or a list of images.

        ``return_tensors`` is accepted for interface compatibility but per-image
        shapes vary, so outputs are always python lists of tensors (never
        stacked). ``pixel_values[i]`` is (N_i, patch_dim); ``image_position_ids[i]``
        is (N_i, 2); ``num_patches_per_image[i]`` is N_i.
        """
        if not isinstance(images, list):
            images = [images]

        pixel_values: list[Tensor] = []
        position_ids: list[Tensor] = []
        num_patches: list[int] = []
        for image in images:
            patches, positions = self._process_one(image)
            pixel_values.append(patches)
            position_ids.append(positions)
            num_patches.append(patches.shape[0])

        return BatchFeature(
            data={
                "pixel_values": pixel_values,
                "image_position_ids": position_ids,
                "num_patches_per_image": num_patches,
            },
            tensor_type=None,  # variable shapes: never stack
        )


__all__ = ["RawImageProcessor", "get_aspect_ratio_preserving_size", "convert_image_to_patches"]
