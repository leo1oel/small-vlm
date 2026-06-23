"""2D (axial / interleaved-MRoPE) rotary embedding for generation image tokens.

Ported from Qwen3-VL's `Qwen3VLTextRotaryEmbedding` (the reference "online
implementation" for our backbone family; HiDream-O1 uses exactly this). The
stock Qwen3 RoPE is 1D: it rotates every band by a token's 1D sequence index,
so two vertically-adjacent image patches (1D index differs by a full grid row)
are treated as far apart. FLUX / minit2i / Qwen3-VL all instead use 2D RoPE so
spatial neighbours are close in rotary space.

Design (minimal-intrusion): this module REPLACES `model.rotary_emb`. When called
with a plain (B,L) position_ids — text / understanding / any non-generation
forward — it is BIT-IDENTICAL to the stock Qwen3RotaryEmbedding (it reuses the
base module's inv_freq + attention_scaling, and equal-axis MRoPE reduces exactly
to 1D). For generation, `forward_generation` stashes a (3,B,L) position tensor
(t,h,w axes) on `_mrope_positions`; the prefix tokens get equal axes (1D) and the
image block gets distinct h/w coordinates, giving them true 2D spatial RoPE.
The interleaving (apply_interleaved_mrope) spreads each axis across the frequency
spectrum so h and w are isotropic. mrope_section sums to head_dim//2.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def default_mrope_section(d2: int) -> list[int]:
    """Split head_dim//2 frequency bands across (t, h, w). Matches Qwen3-VL
    [24,20,20] at d2=64; generic otherwise. Any split keeps TEXT bit-identical
    (equal axes -> 1D); it only sets the h/w band budget for image tokens."""
    if d2 == 64:
        return [24, 20, 20]
    hw = d2 // 3
    return [d2 - 2 * hw, hw, hw]


class Gen2DRotaryEmbedding(nn.Module):
    def __init__(self, base_rotary: nn.Module, mrope_section: list[int] | None = None) -> None:
        super().__init__()
        # Reuse the stock module's exact frequencies -> guaranteed 1D identity.
        self.register_buffer("inv_freq", base_rotary.inv_freq.clone(), persistent=False)
        self.attention_scaling: float = float(getattr(base_rotary, "attention_scaling", 1.0))
        d2 = int(self.inv_freq.shape[0])
        self.mrope_section = list(mrope_section) if mrope_section else default_mrope_section(d2)
        if sum(self.mrope_section) != d2:
            raise ValueError(f"mrope_section {self.mrope_section} must sum to head_dim//2={d2}")
        # Set by forward_generation (try/finally) to inject (3,B,L) positions.
        self._mrope_positions: Tensor | None = None

    def apply_interleaved_mrope(self, freqs: Tensor) -> Tensor:
        """freqs (3, B, L, d2) -> (B, L, d2). Interleave t/h/w across bands
        ([THTHWH...] not chunked [TTT..HHH..WWW..]) to keep frequency continuity
        and make h/w isotropic. Verbatim logic from Qwen3-VL."""
        ms = self.mrope_section
        freqs_t = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):  # H (dim1), W (dim2)
            length = ms[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        pos = self._mrope_positions if self._mrope_positions is not None else position_ids
        if pos.ndim == 2:  # (B,L) -> equal axes -> exactly 1D
            pos = pos[None, ...].expand(3, pos.shape[0], -1)
        pos = pos.to(x.device)
        inv = self.inv_freq[None, None, :, None].float().expand(3, pos.shape[1], -1, 1).to(x.device)
        pos_exp = pos[:, :, None, :].float()  # (3, B, 1, L)
        dev = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=dev, enabled=False):
            freqs = (inv.float() @ pos_exp.float()).transpose(2, 3)  # (3, B, L, d2)
            freqs = self.apply_interleaved_mrope(freqs)  # (B, L, d2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def build_mrope_position_ids(
    prefix_len: int, grid_h: int, grid_w: int, batch: int, device: torch.device
) -> Tensor:
    """(3, B, L) positions for a [prefix | image-grid] generation sequence.

    Prefix tokens (text + timestep) get equal t=h=w=arange -> 1D, structural
    (padding-independent, for CFG cond/uncond consistency). Each image patch at
    row-major index k -> (y=k//grid_w, x=k%grid_w) gets t=prefix_len (constant),
    h=prefix_len+y, w=prefix_len+x: a 2D block placed right after the prefix.
    """
    n = grid_h * grid_w
    total = prefix_len + n
    pos = torch.zeros(3, 1, total, dtype=torch.long, device=device)
    pr = torch.arange(prefix_len, device=device)
    pos[:, 0, :prefix_len] = pr  # broadcast over the 3 axes
    k = torch.arange(n, device=device)
    pos[0, 0, prefix_len:] = prefix_len
    pos[1, 0, prefix_len:] = prefix_len + (k // grid_w)
    pos[2, 0, prefix_len:] = prefix_len + (k % grid_w)
    return pos.expand(3, batch, total).contiguous()
