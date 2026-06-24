"""Visual-encoder distillation for the encoder-free (native) VLM
(spec: docs/superpowers/specs/2026-06-21-visual-distill-design.md).

Motivation. A native VLM (raw patches -> linear -> decoder, no vision tower)
left to plain SFT rides the language prior and never learns to condition on
pixels (FDI probe R0 ~ 0; POPE degenerate). The linear-predictivity probe shows
the model DOES build encoder-grade features internally, but slowly and late, and
only at its own pace. This module adds an explicit teacher signal: align the
LLM's hidden states at image positions to a frozen vision encoder's per-patch
features, so the visual pathway is supervised directly instead of having to be
discovered through next-token loss alone.

One mechanism, seven methods (config.method):
  repa       single mid-layer (~0.3 depth), MLP head, per-patch neg-cosine to
             CLIP's final patch features. = REPA (arXiv:2410.06940), which aligns
             a denoising-transformer layer to DINOv2 for *generation*; here the
             same idea for *understanding* (LLM layer -> CLIP).
  eve        align the FINAL hidden state (post-norm) to CLIP final features
             (EVE arXiv:2406.11832 Patch-Aligning Layer, last-layer variant).
  vora       block-wise: the first N LLM layers each align 1:1 to a relative-
             depth-matched CLIP block, RMSNorm+Linear head per layer
             (VoRA arXiv:2503.20680, minus the LoRA — we train the trunk).
  softdepth  OURS: a learned softmax over a pool of layers picks WHICH depth
             serves as the internal encoder (the model self-selects placement,
             vs VoRA hard-coding the first N). One MLP head on the mixed hidden.
  relational OURS: align the token-token similarity (Gram) matrix instead of the
             features themselves — no projection head, dimension-free, transfers
             relational structure (CKA/Gram alignment).
  vae        low-level control: teacher = a frozen VAE encoder's latent grid
             (pixels/texture, no semantics) instead of CLIP — isolates how much
             of the gain is semantic vs reconstructive.
  breen      BREEN (arXiv:2503.12446): distill the LLM's hidden at the LEARNABLE
             QUERY positions (not image patches) to a dual-granularity avg-pooled
             CLIP grid — first num_fine query rows <-> 8x8 fine pool, last
             num_coarse rows <-> 6x6 coarse pool. The CLIP(1024) target is
             projected UP to LLM-hidden via a LayerNorm+Linear `norm_layer` and
             1-cos is taken in LLM-hidden space (faithful to BREEN's geometry).

The teacher sees the SAME pixels as the model: we reconstruct the RGB image from
the raw patches already in the batch (RawImageProcessor is lossless, rescale-only
[0,1], known 48x48x3 row-major layout) — so distillation needs ZERO dataloader
changes. The teacher is held off the module tree (list-wrapped on the model) so
it is invisible to .parameters()/optimizer/state_dict (it must never be trained
nor bloat checkpoints); only the small projection head is a real submodule.
"""

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

log: logging.Logger = logging.getLogger(__name__)

# OpenAI CLIP image normalization (applied to [0,1] RGB). Hardcoded so we never
# need to load a CLIPImageProcessor just for three constants.
CLIP_IMAGE_MEAN: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

DISTILL_METHODS: tuple[str, ...] = (
    "repa",
    "eve",
    "vora",
    "softdepth",
    "relational",
    "vae",
    "breen",
)


def resolve_distill_layers(method: str, layers_cfg: list[int] | None, num_layers: int) -> list[int]:
    """1-based decoder-output layer indices this method aligns.

    `eve` and `breen` return [] — they use the final post-norm hidden state
    directly (breen at query positions), not a captured intermediate layer. The
    others default sensibly when layers_cfg is None: repa/relational/vae a single
    ~0.3-depth layer; vora the first half (block-wise); softdepth every
    intermediate layer as the selection pool.
    """
    if method in ("eve", "breen"):
        return []
    if layers_cfg:
        layers = sorted({int(k) for k in layers_cfg})
        bad = [k for k in layers if not 1 <= k <= num_layers]
        if bad:
            raise ValueError(
                f"visual_distill layers {bad} out of range [1, {num_layers}] "
                f"(1-based decoder-output index)"
            )
        return layers
    if method in ("repa", "relational", "vae"):
        return [max(1, round(0.3 * num_layers))]
    if method == "vora":
        return list(range(1, max(2, round(0.5 * num_layers)) + 1))
    if method == "softdepth":
        return list(range(1, num_layers))  # all intermediate layers as the pool
    raise ValueError(f"unknown visual_distill method {method!r} (one of {DISTILL_METHODS})")


def reconstruct_image_from_patches(
    patches: Tensor,
    positions: Tensor,
    patch_px: int,
    out_size: int,
    src_mean: list[float] | None,
    src_std: list[float] | None,
) -> Tensor:
    """Lossless inverse of RawImageProcessor: raw patches -> (3, out_size,
    out_size) RGB in [0,1], resized for the teacher.

    `patches` (N, patch_px**2 * 3) row-major (row, col, channel) per patch (the
    convert_image_to_patches layout); `positions` (N, 2) integer (x=col, y=row)
    model-grid coordinates. If the model normalized (src_mean/std set), undo it
    to recover [0,1] before the teacher's own normalization.
    """
    n = patches.shape[0]
    grid_w = int(positions[:, 0].max().item()) + 1
    grid_h = int(positions[:, 1].max().item()) + 1
    p = patch_px
    blocks = patches.reshape(n, p, p, 3).float()  # (N, row, col, ch)
    canvas = blocks.new_zeros((grid_h, p, grid_w, p, 3))
    canvas[positions[:, 1], :, positions[:, 0], :, :] = blocks
    img = canvas.permute(4, 0, 1, 2, 3).reshape(3, grid_h * p, grid_w * p)
    if src_mean is not None and src_std is not None:
        mean = torch.tensor(src_mean, device=img.device).view(3, 1, 1)
        std = torch.tensor(src_std, device=img.device).view(3, 1, 1)
        img = img * std + mean
    img = img.clamp(0.0, 1.0)
    img = F.interpolate(
        img.unsqueeze(0), size=(out_size, out_size), mode="bicubic", align_corners=False
    ).squeeze(0)
    return img.clamp(0.0, 1.0)


class VisualDistillTeacher(nn.Module):
    """Frozen vision encoder that turns reconstructed [0,1] RGB into per-patch
    target features. CLIP (semantic) or a VAE encoder (low-level latent).

    Held list-wrapped on the model so it is NOT a registered submodule — never
    trained, never saved. Device/dtype are managed manually (see `.align_to`).
    Always eval + no_grad: targets are detached.
    """

    def __init__(self, kind: str, name: str, out_size: int = 224) -> None:
        super().__init__()
        self.kind: str = kind
        self.name: str = name
        self.out_size: int = out_size
        if kind == "clip":
            from transformers import CLIPVisionModel

            self.model = CLIPVisionModel.from_pretrained(name)
            self.feature_dim: int = int(self.model.config.hidden_size)
            self.num_blocks: int = int(self.model.config.num_hidden_layers)
            mean, std = CLIP_IMAGE_MEAN, CLIP_IMAGE_STD
        elif kind == "siglip":
            # SigLIP: the native models' MOST-predictable target per the
            # predictivity probe (R² ~0.5-0.65 vs CLIP ~0.31-0.47) — the
            # data-optimal dense teacher. No CLS token (mean-pool head), so all
            # last_hidden_state rows are patch tokens. Norm = [-1, 1] (mean/std .5).
            from transformers import SiglipVisionModel

            self.model = SiglipVisionModel.from_pretrained(name)
            self.feature_dim = int(self.model.config.hidden_size)
            self.num_blocks = int(self.model.config.num_hidden_layers)
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        elif kind == "vae":
            from diffusers import AutoencoderKL

            self.model = AutoencoderKL.from_pretrained(name)
            self.feature_dim = int(self.model.config.latent_channels)
            self.num_blocks = 0
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)  # VAE expects [-1, 1]
        else:
            raise ValueError(f"unknown teacher kind {kind!r} (clip|siglip|vae)")
        self.register_buffer("norm_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("norm_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, images01: Tensor, want_blocks: bool = False) -> dict[str, Any]:
        """images01: (B, 3, out_size, out_size) in [0,1]. Returns a dict with
        `grid` (B, Ht, Wt, C) final per-patch features and, when want_blocks,
        `blocks` (list over teacher layers of (B, Ht, Wt, C)) for vora matching.
        """
        x = (images01.to(self.norm_mean.dtype) - self.norm_mean) / self.norm_std
        if self.kind in ("clip", "siglip"):
            # CLIP prepends a CLS token (drop it); SigLIP has none — all rows are
            # patch tokens. cls = 1 for clip, 0 for siglip.
            cls = 1 if self.kind == "clip" else 0
            out = self.model(pixel_values=x, output_hidden_states=want_blocks)
            tok = out.last_hidden_state[:, cls:, :]
            side = int(round(tok.shape[1] ** 0.5))
            grid = tok.reshape(tok.shape[0], side, side, tok.shape[-1])
            result: dict[str, Any] = {"grid": grid}
            if want_blocks:
                blocks = []
                for h in out.hidden_states[1:]:  # skip embeddings
                    t = h[:, cls:, :]
                    blocks.append(t.reshape(t.shape[0], side, side, t.shape[-1]))
                result["blocks"] = blocks
            return result
        # VAE: latent mean grid (B, C, h, w) -> (B, h, w, C)
        latent = self.model.encode(x).latent_dist.mean
        grid = latent.permute(0, 2, 3, 1).contiguous()
        return {"grid": grid}


def _proj_head(in_dim: int, out_dim: int, hidden: int, kind: str) -> nn.Module:
    """Projection head from LLM hidden to teacher space.
    mlp:     RMSNorm -> Linear -> GELU -> Linear   (repa/eve/softdepth/vae)
    linear:  RMSNorm -> Linear                     (vora, faithful AuxHead)
    """
    if kind == "linear":
        return nn.Sequential(nn.RMSNorm(in_dim), nn.Linear(in_dim, out_dim))
    return nn.Sequential(
        nn.RMSNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim)
    )


class VisualDistillHead(nn.Module):
    """Trainable projection head(s) + (softdepth) layer-mixing weights.

    A real submodule on the ForCausalLM (`visual_distill_head`); its params fall
    through to the `language_model` optimizer group (trained with the LM, which
    every native-SFT distillation run trains). The teacher is held separately
    and is NOT part of this module.

    The loss math lives here (`compute`) so the model's chunked_ce_forward stays
    a one-line call, mirroring how the visual-aux loss is structured.
    """

    def __init__(
        self,
        method: str,
        llm_dim: int,
        teacher_dim: int,
        layers: list[int],
        head_hidden: int,
        loss_type: str,
        num_fine: int = 64,
        num_coarse: int = 36,
    ) -> None:
        super().__init__()
        self.method: str = method
        self.layers: list[int] = layers
        self.loss_type: str = loss_type
        self.teacher_dim: int = teacher_dim
        self.num_fine: int = num_fine
        self.num_coarse: int = num_coarse
        hidden = head_hidden if head_hidden > 0 else llm_dim
        if method == "breen":
            # BREEN `norm_layer`: project the CLIP target UP to LLM-hidden
            # (LayerNorm(1024) -> Linear(1024 -> hidden, bias=False)); cosine is
            # taken in LLM-hidden space. No projection on the LLM (query) side —
            # the queries are aligned directly. (vision_tokenizer.py:433-436)
            self.proj = None
            self.norm_layer = nn.Sequential(
                nn.LayerNorm(teacher_dim), nn.Linear(teacher_dim, llm_dim, bias=False)
            )
        elif method == "relational":
            self.proj = None  # dimension-free: align Gram matrices, no projection
        elif method == "vora":
            # One RMSNorm+Linear AuxHead per aligned block (VoRA: each early
            # block has its own head onto its matched teacher block).
            self.proj = nn.ModuleList(
                [_proj_head(llm_dim, teacher_dim, hidden, "linear") for _ in layers]
            )
        else:
            self.proj = _proj_head(llm_dim, teacher_dim, hidden, "mlp")
        # softdepth: learned softmax over the layer pool (self-selected depth).
        if method == "softdepth":
            if len(layers) < 2:
                raise ValueError(
                    f"softdepth needs a pool of >= 2 layers to select among, got {layers}; "
                    "a length-1 softmax is constant (no gradient to depth_logits)"
                )
            self.depth_logits = nn.Parameter(torch.zeros(len(layers)))

    def _align(self, pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        """Per-token alignment loss + the bare cosine (for logging). pred/target
        are (M, C). cosine: 1 - cos; smoothl1: huber on raw vectors (vae)."""
        pred = pred.float()
        target = target.float()
        if self.loss_type == "smoothl1":
            cos = F.cosine_similarity(pred, target, dim=-1).mean()
            return F.smooth_l1_loss(pred, target), cos
        p = F.normalize(pred, dim=-1)
        t = F.normalize(target, dim=-1)
        cos = (p * t).sum(dim=-1)
        return (1.0 - cos).mean(), cos.mean()

    def compute(self, samples: list[dict[str, Any]]) -> tuple[Tensor, dict[str, Tensor]]:
        """samples: per-image dicts. Each carries:
            'native': dict[layer_idx -> (m, d)] gathered LLM hidden at this
                      image's surviving patch positions (layer 0 key = final
                      post-norm hidden, used by eve).
            'target': (m, teacher_dim) teacher features at the matched patches.
            'target_blocks': (vora) dict[layer_idx -> (m, teacher_dim)].
        Returns (loss, components). Token-weighted mean across images.
        """
        device = samples[0]["target"].device
        loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        cos_sum = torch.zeros((), dtype=torch.float32, device=device)
        n_tok = 0
        for s in samples:
            m = s["target"].shape[0]
            if self.method == "relational":
                loss_i, cos_i = self._relational(s)
            elif self.method == "vora":
                loss_i, cos_i = self._vora(s)
            elif self.method == "softdepth":
                loss_i, cos_i = self._softdepth(s)
            else:  # repa / eve / vae : single projector on a single layer
                lyr = self.layers[0] if self.method != "eve" else 0
                pred = self.proj(s["native"][lyr])
                loss_i, cos_i = self._align(pred, s["target"])
            loss_sum = loss_sum + loss_i * m
            cos_sum = cos_sum + cos_i * m
            n_tok += m
        loss = loss_sum / max(n_tok, 1)
        components: dict[str, Tensor] = {
            "distill": loss.detach(),
            "distill_cos": (cos_sum / max(n_tok, 1)).detach(),
        }
        if self.method == "softdepth":
            w = torch.softmax(self.depth_logits.detach().float(), dim=0)
            # Effective selected depth = sum_l w_l * layer_l (a single readable
            # number: which depth the model put its internal encoder at).
            layer_idx = torch.tensor([float(x) for x in self.layers], device=w.device)
            sel = float((w * layer_idx).sum())
            components["distill_sel_depth"] = torch.tensor(sel, device=device)
            components["distill_sel_max"] = w.max().to(device)
        return loss, components

    def compute_breen(self, samples: list[dict[str, Any]]) -> tuple[Tensor, dict[str, Tensor]]:
        """BREEN query distillation. Each per-image sample carries:
            'query_hidden': (num_fine+num_coarse, llm_dim) LLM final hidden at
                            this image's query positions, in row order.
            'target_fine':  (num_fine, teacher_dim)  8x8 avg-pool of the CLIP grid.
            'target_coarse':(num_coarse, teacher_dim) 6x6 avg-pool of the CLIP grid.
        Projects the CLIP targets UP to LLM-hidden via `norm_layer`, then
        1-cos(fine_queries, fine_target) + 1-cos(coarse_queries, coarse_target).
        Token-weighted mean across images (weight = number of queries)."""
        device = samples[0]["query_hidden"].device
        loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        cos_sum = torch.zeros((), dtype=torch.float32, device=device)
        n_tok = 0
        for s in samples:
            qh = s["query_hidden"]
            fine_pred = qh[: self.num_fine]
            coarse_pred = qh[self.num_fine :]
            fine_tgt = self.norm_layer(s["target_fine"].to(qh.dtype))
            coarse_tgt = self.norm_layer(s["target_coarse"].to(qh.dtype))
            l_f, c_f = self._align(fine_pred, fine_tgt)
            l_c, c_c = self._align(coarse_pred, coarse_tgt)
            m = qh.shape[0]
            loss_sum = loss_sum + (l_f + l_c) * m
            cos_sum = cos_sum + 0.5 * (c_f + c_c) * m
            n_tok += m
        loss = loss_sum / max(n_tok, 1)
        return loss, {
            "distill": loss.detach(),
            "distill_cos": (cos_sum / max(n_tok, 1)).detach(),
        }

    def _softdepth(self, s: dict[str, Any]) -> tuple[Tensor, Tensor]:
        # Magnitude-normalize each pooled layer, mix by softmax weights, project.
        stack = torch.stack([s["native"][k] for k in self.layers], dim=0)  # (L, m, d)
        stack = F.normalize(stack.float(), dim=-1) * (stack.shape[-1] ** 0.5)
        w = torch.softmax(self.depth_logits, dim=0).view(-1, 1, 1)
        mixed = (w * stack).sum(dim=0).to(s["native"][self.layers[0]].dtype)
        return self._align(self.proj(mixed), s["target"])

    def _vora(self, s: dict[str, Any]) -> tuple[Tensor, Tensor]:
        loss = torch.zeros((), dtype=torch.float32, device=s["target"].device)
        cos = torch.zeros((), dtype=torch.float32, device=s["target"].device)
        for j, lyr in enumerate(self.layers):
            tgt = s["target_blocks"][lyr]
            li, ci = self._align(self.proj[j](s["native"][lyr]), tgt)
            loss = loss + li
            cos = cos + ci
        n = len(self.layers)
        return loss / n, cos / n

    def _relational(self, s: dict[str, Any]) -> tuple[Tensor, Tensor]:
        # Gram (token-token cosine) alignment — no projection, dimension-free.
        h = F.normalize(s["native"][self.layers[0]].float(), dim=-1)
        t = F.normalize(s["target"].float(), dim=-1)
        gram_h = h @ h.t()
        gram_t = t @ t.t()
        loss = F.mse_loss(gram_h, gram_t)
        # "cos" here = mean off-diagonal agreement, a comparable monitor.
        return loss, (1.0 - (gram_h - gram_t).abs().mean())


__all__ = [
    "DISTILL_METHODS",
    "resolve_distill_layers",
    "reconstruct_image_from_patches",
    "VisualDistillTeacher",
    "VisualDistillHead",
]
