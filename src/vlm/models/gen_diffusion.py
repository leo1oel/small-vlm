"""Flow-matching objective + Euler sampler for text->image generation.

Spec: docs/superpowers/specs/2026-06-20-unified-vlm-design.md . Faithful to
minit2i / MM-JiT (mini_t2i/diffusion.py), adapted for our patch space.

Convention (IMPORTANT): **t = 1 is the CLEAN image, t = 0 is pure noise.**
    x_t      = x1 * t + noise * (1 - t)            # noise pre-scaled by noise_scale
    velocity = (x1 - x_t) / max(1 - t, eps)        # == x1 - noise away from t->1
The network predicts the CLEAN image (x-prediction); the loss is the MSE between
the predicted and target velocities (v-loss), with an eps floor on (1 - t) to
avoid blow-up near t -> 1. Pure torch (no vlm/transformers imports) so it is
cheap to unit-test in isolation.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def sample_timesteps(
    batch_size: int,
    mu: float,
    sigma: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Logit-normal timestep sampling: t = sigmoid(N(mu, sigma)), clamped to
    (0, 1). mu < 0 biases toward small t (noisier samples). Returns (B,)."""
    z = torch.randn(batch_size, device=device, generator=generator) * sigma + mu
    return torch.sigmoid(z).clamp(1e-5, 1.0 - 1e-5)


def _expand_t(t: Tensor, ndim: int) -> Tensor:
    """Reshape a per-sample (B,) tensor to (B, 1, 1, ...) for broadcasting over
    a feature tensor of rank `ndim`."""
    return t.reshape(t.shape[0], *([1] * (ndim - 1)))


def add_noise(
    x1: Tensor, t: Tensor, noise_scale: float, noise: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Interpolate clean image x1 (..) and noise toward x_t at timestep t.

    x_t = x1 * t + noise * (1 - t) with t broadcast per-sample. If `noise` is
    None, samples `randn_like(x1) * noise_scale`. Returns (x_t, noise)."""
    if noise is None:
        noise = torch.randn_like(x1) * noise_scale
    tb = _expand_t(t.to(x1.dtype), x1.ndim)
    x_t = x1 * tb + noise * (1.0 - tb)
    return x_t, noise


def to_velocity(pred_x0: Tensor, x_t: Tensor, t: Tensor, eps: float = 0.05) -> Tensor:
    """Convert an x-prediction (clean-image estimate) to a velocity:
    v = (pred_x0 - x_t) / max(1 - t, eps)."""
    tb = _expand_t(t.to(pred_x0.dtype), pred_x0.ndim)
    return (pred_x0 - x_t) / (1.0 - tb).clamp_min(eps)


def flow_matching_loss(
    pred_x0: Tensor, x_t: Tensor, x1: Tensor, t: Tensor, eps: float = 0.05
) -> Tensor:
    """Velocity-space MSE: mean over feature dims, then over batch. Zero when
    pred_x0 == x1 (perfect x-prediction)."""
    v_pred = to_velocity(pred_x0, x_t, t, eps)
    v_target = to_velocity(x1, x_t, t, eps)
    per_sample = (v_pred - v_target).pow(2).flatten(1).mean(dim=1)
    return per_sample.mean()


def euler_step(x: Tensor, v: Tensor, dt: float) -> Tensor:
    """One explicit Euler ODE step: x_{next} = x + v * dt."""
    return x + v * dt


def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """Sinusoidal embedding of timesteps t (B,) in [0, 1] -> (B, dim).
    minit2i convention (mini_t2i/model.py:36). Odd dim is zero-padded by one."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class GenTimestepEmbedder(nn.Module):
    """Sinusoidal timestep -> 2-layer MLP -> a single (B, hidden_size) token
    embedding, spliced into the sequence between text and image (in-context
    conditioning; no adaLN, keeping the Qwen3 backbone a plain pre-norm
    transformer)."""

    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: Tensor) -> Tensor:
        emb = timestep_embedding(t, self.freq_dim)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))
