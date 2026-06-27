"""Fixed-canvas patchify / unpatchify + position grid for the generation path.

Spec: docs/superpowers/specs/2026-06-20-unified-vlm-design.md . Generation works
in the same flat patch space as the encoder-free understanding connector
(patch_dim = model_patch_size**2 * 3). Unlike understanding (variable aspect,
RawImageProcessor), generation pins a FIXED square grid so the sampler can
unpatchify deterministically.

`pixels_to_patches` reproduces `image_processing_raw.convert_image_to_patches`
EXACTLY (verified by test) so the connector + factorized 2D pos-embedding see
the patch order they were trained on. `patches_to_pixels` is its exact inverse
(the repo has none). Pure torch (no vlm/transformers imports).
"""

from __future__ import annotations

import torch
from torch import Tensor


def pixels_to_patches(image: Tensor, model_patch_size: int) -> Tensor:
    """(C, H, W) -> (N, P*P*C) or (B, C, H, W) -> (B, N, P*P*C).

    Row-major grid; each patch flattened (rows, cols, channels). Identical layout
    to `convert_image_to_patches` (image_processing_raw.py:94)."""
    p = model_patch_size
    if image.dim() == 3:
        c, h, w = image.shape
        gh, gw = h // p, w // p
        x = image.reshape(c, gh, p, gw, p).permute(1, 3, 2, 4, 0)  # (gh,gw,p,p,c)
        return x.reshape(gh * gw, p * p * c)
    if image.dim() == 4:
        b, c, h, w = image.shape
        gh, gw = h // p, w // p
        x = image.reshape(b, c, gh, p, gw, p).permute(0, 2, 4, 3, 5, 1)  # (b,gh,gw,p,p,c)
        return x.reshape(b, gh * gw, p * p * c)
    raise ValueError(f"expected (C,H,W) or (B,C,H,W), got shape {tuple(image.shape)}")


def patches_to_pixels(
    patches: Tensor, grid_h: int, grid_w: int, model_patch_size: int, channels: int = 3
) -> Tensor:
    """Exact inverse of `pixels_to_patches`.

    (N, P*P*C) -> (C, H, W) or (B, N, P*P*C) -> (B, C, H, W), with
    H = grid_h*P, W = grid_w*P."""
    p, c = model_patch_size, channels
    if patches.dim() == 2:
        x = patches.reshape(grid_h, grid_w, p, p, c).permute(4, 0, 2, 1, 3).contiguous()
        return x.reshape(c, grid_h * p, grid_w * p)
    if patches.dim() == 3:
        b = patches.shape[0]
        x = patches.reshape(b, grid_h, grid_w, p, p, c).permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.reshape(b, c, grid_h * p, grid_w * p)
    raise ValueError(f"expected (N,D) or (B,N,D), got shape {tuple(patches.shape)}")


def assemble_generation_inputs(
    text_embeds: Tensor, text_mask: Tensor, t_token: Tensor, img_embeds: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build the generation sequence [text | timestep token | image patches].

    text_embeds (B, Lt, H), text_mask (B, Lt) 0/1 (left-padding honored),
    t_token (B, 1, H), img_embeds (B, N, H). Returns:
      inputs_embeds (B, L, H), prefix_mask (B, L) bool (text + t-token),
      image_mask (B, L) bool, position_ids (B, L) long.  L = Lt + 1 + N.

    position_ids are STRUCTURAL (arange), i.e. independent of the text mask, so
    the classifier-free-guidance conditional and unconditional (dropped-text)
    passes place the timestep token and image block at IDENTICAL RoPE positions
    — required for a valid guidance combination and for training with
    label-dropout. (Use left-padding for the text so there is no position gap
    between the real text and the timestep token.)"""
    bsz, text_len, _ = text_embeds.shape
    n_img = img_embeds.shape[1]
    device = text_embeds.device
    inputs_embeds = torch.cat([text_embeds, t_token, img_embeds], dim=1)
    tm = text_mask.to(torch.bool)
    ones_t = torch.ones(bsz, 1, dtype=torch.bool, device=device)
    zeros_text = torch.zeros(bsz, text_len, dtype=torch.bool, device=device)
    zeros_t = torch.zeros(bsz, 1, dtype=torch.bool, device=device)
    ones_img = torch.ones(bsz, n_img, dtype=torch.bool, device=device)
    prefix_mask = torch.cat(
        [tm, ones_t, torch.zeros(bsz, n_img, dtype=torch.bool, device=device)], dim=1
    )
    image_mask = torch.cat([zeros_text, zeros_t, ones_img], dim=1)
    seq_len = inputs_embeds.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)
    return inputs_embeds, prefix_mask, image_mask, position_ids


def make_position_ids(grid_h: int, grid_w: int, device: torch.device | None = None) -> Tensor:
    """(N, 2) int64 XY coords, row-major: patch k -> (x = k % grid_w, y = k // grid_w).

    Matches the RawImageProcessor meshgrid(arange(W_p), arange(H_p), indexing='xy')
    convention (image_processing_raw.py:227-237) so the connector pos-embedding
    lookup `table[pos[:,0],0] + table[pos[:,1],1]` is correct."""
    idx = torch.arange(grid_h * grid_w, dtype=torch.long, device=device)
    x = idx % grid_w
    y = idx // grid_w
    return torch.stack((x, y), dim=-1)
