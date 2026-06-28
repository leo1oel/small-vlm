"""Perceptual losses for pixel-space generation (PRX / HiDream-O1 recipe).

Adds an auxiliary LPIPS + P-DINO loss between the denoiser's x0 prediction and
the clean target image (both RGB in [-1, 1]), on top of the flow-matching MSE.
Pure flow-matching MSE in pixel space favours low-frequency / blurry outputs;
perceptual supervision is the lever PRX and HiDream-O1 use to get sharp images
(weights from PRX configs: LPIPS 0.1, P-DINO 0.01).

The frozen LPIPS (VGG) and DINOv2 networks are NOT submodules of the VLM: they
are a module-level singleton cache keyed by (device, dtype), so they never enter
named_parameters / state_dict / the optimizer and are never checkpointed. They
are loss functions, not model weights.

Reference (read, not copied — PRX is Composer-coupled, inference-only HiDream has
no loss code): github.com/Photoroom/PRX prx/algorithm/{lpips,perceptual_dino}.py.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# (device, dtype) -> _Perceptual singleton
_CACHE: dict = {}


class _DinoEncoder(nn.Module):
    """Frozen DINOv2 patch-token encoder. Input RGB in [-1, 1]."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        # Cached on the login node into ~/.cache/torch/hub; loads offline.
        self.model = torch.hub.load(
            "facebookresearch/dinov2", model_name, source="github", trust_repo=True
        ).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.patch = 14
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, img: Tensor) -> Tensor:
        img = ((img + 1.0) / 2.0).clamp(0.0, 1.0)
        img = (img - self.mean.to(img.dtype)) / self.std.to(img.dtype)
        return self.model.forward_features(img)["x_norm_patchtokens"]  # (B, P, D)


class _Perceptual(nn.Module):
    def __init__(self, lpips_net: str, dino_model: str, dino_w: float) -> None:
        super().__init__()
        import lpips  # pyright: ignore[reportMissingImports]  # optional dependency

        self.lpips_fn = lpips.LPIPS(net=lpips_net).eval()
        for p in self.lpips_fn.parameters():
            p.requires_grad = False
        self.dino = _DinoEncoder(dino_model) if dino_w > 0.0 else None

    @torch.no_grad()
    def _noop(self) -> None:  # keep type checkers calm; nets are frozen
        pass

    def lpips(self, pred: Tensor, gt: Tensor) -> Tensor:
        # lpips expects [-1, 1]; returns (B, 1, 1, 1) -> per-sample (B,).
        return self.lpips_fn(pred, gt).view(-1)

    def dino_loss(self, pred: Tensor, gt: Tensor, resize: int) -> Tensor:
        assert self.dino is not None
        r = (resize // self.dino.patch) * self.dino.patch  # multiple of 14
        pred = F.interpolate(pred, size=(r, r), mode="bilinear", align_corners=False)
        gt = F.interpolate(gt, size=(r, r), mode="bilinear", align_corners=False)
        f_pred = self.dino(pred)
        with torch.no_grad():
            f_gt = self.dino(gt)
        cos = F.cosine_similarity(f_pred.float(), f_gt.float(), dim=-1)  # (B, P)
        return (1.0 - cos).mean(dim=-1)  # per-sample (B,)


def _get(
    device: torch.device, dtype: torch.dtype, lpips_net: str, dino_model: str, dino_w: float
) -> _Perceptual:
    key = (str(device), str(dtype), lpips_net, dino_model, dino_w > 0.0)
    mod = _CACHE.get(key)
    if mod is None:
        mod = _Perceptual(lpips_net, dino_model, dino_w).to(device=device, dtype=dtype)
        mod.eval()
        _CACHE[key] = mod
    return mod


def perceptual_loss(
    pred_img: Tensor,
    gt_img: Tensor,
    lpips_weight: float,
    dino_weight: float,
    lpips_net: str = "vgg",
    dino_model: str = "dinov2_vitb14_reg",
    resize: int = 256,
    t: Tensor | None = None,
    t_gate: float = 0.0,
) -> dict[str, Tensor]:
    """pred_img / gt_img: (B, 3, H, W) in [-1, 1]. gt is detached internally.

    Noise gating (PixelGen 2602.02493 sec 3.4): perceptual losses are only
    applied to LOW-noise samples, where the x0 prediction is close enough to a
    real image that VGG/DINO features are meaningful. In our convention t=1 is
    clean and t=0 is pure noise, so "low noise" = t > t_gate (PixelGen disables
    the first 30% high-noise steps -> t_gate=0.3). t=None or t_gate<=0 = all.

    Returns {'lpips', 'dino', 'weighted', 'frac'} (scalars over the gated set).
    'weighted' = the term to add to the base loss; pred keeps grad, gt does not.
    """
    device = pred_img.device
    dtype = pred_img.dtype
    net = _get(device, dtype, lpips_net, dino_model, dino_weight)
    pred = pred_img.clamp(-1.0, 1.0)
    gt = gt_img.detach().clamp(-1.0, 1.0)
    bsz = pred.shape[0]

    zeros_b = pred.new_zeros(bsz)
    lp = net.lpips(pred.to(dtype), gt.to(dtype)).float() if lpips_weight > 0.0 else zeros_b
    dn = (
        net.dino_loss(pred.to(dtype), gt.to(dtype), resize).float()
        if dino_weight > 0.0
        else zeros_b
    )

    if t is not None and t_gate > 0.0:
        mask = (t.to(device).float() > t_gate).float()  # 1 on low-noise samples
    else:
        mask = pred.new_ones(bsz)
    denom = mask.sum().clamp(min=1.0)
    lp_m = (lp * mask).sum() / denom
    dn_m = (dn * mask).sum() / denom
    weighted = lpips_weight * lp_m + dino_weight * dn_m
    return {
        "lpips": lp_m.detach(),
        "dino": dn_m.detach(),
        "weighted": weighted,
        "frac": (mask.mean()).detach(),
    }
