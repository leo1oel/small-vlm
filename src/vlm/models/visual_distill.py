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
            from diffusers import AutoencoderKL  # pyright: ignore[reportMissingImports]

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
        debias_target: bool = False,
        debias_momentum: float = 0.9,
        debias_std: bool = False,
        rkd_dist_weight: float = 0.0,
        rkd_angle_weight: float = 0.0,
        rkd_angle_triplets: int = 512,
        vicreg_var_weight: float = 0.0,
        vicreg_cov_weight: float = 0.0,
        vicreg_gamma: float = 1.0,
        ac_warmup_steps: int = 0,
        mgd_weight: float = 0.0,
        mgd_beta: float = 0.5,
        sigreg_weight: float = 0.0,
        sigreg_dirs: int = 256,
        sigreg_knots: int = 17,
        sigreg_warmup_steps: int = 0,
    ) -> None:
        super().__init__()
        self.method: str = method
        self.layers: list[int] = layers
        self.loss_type: str = loss_type
        self.teacher_dim: int = teacher_dim
        self.num_fine: int = num_fine
        self.num_coarse: int = num_coarse
        # Anti-collapse recipe (spec anticollapse-ablation.md); all default off.
        self.debias_target: bool = debias_target
        self.debias_momentum: float = debias_momentum
        self.debias_std: bool = debias_std
        self.rkd_dist_weight: float = rkd_dist_weight
        self.rkd_angle_weight: float = rkd_angle_weight
        self.rkd_angle_triplets: int = rkd_angle_triplets
        self.vicreg_var_weight: float = vicreg_var_weight
        self.vicreg_cov_weight: float = vicreg_cov_weight
        self.vicreg_gamma: float = vicreg_gamma
        self.ac_warmup_steps: int = ac_warmup_steps
        # Round-2 levers (spec round2-ablation.md).
        self.mgd_weight: float = mgd_weight
        self.mgd_beta: float = mgd_beta
        self.sigreg_weight: float = sigreg_weight
        self.sigreg_dirs: int = sigreg_dirs
        self.sigreg_knots: int = sigreg_knots
        self.sigreg_warmup_steps: int = sigreg_warmup_steps
        if ac_warmup_steps > 0 or sigreg_warmup_steps > 0:
            # Optimizer-step counter for the RKD/VICReg AND SIGReg warmups
            # (persisted so a resume keeps ramping). Registered only when some
            # warmup is on, so non-warmup checkpoints are unchanged.
            self.register_buffer("_ac_step", torch.zeros((), dtype=torch.long))
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
        # Anti-collapse (A): EMA per-channel mean/var of the teacher target (the
        # shared "mean-image" constant). Buffers (no grad, persisted) so a reload
        # resumes the running estimate. Registered iff debias_target (serialized).
        if self.debias_target:
            self.register_buffer("debias_mean", torch.zeros(teacher_dim))
            self.register_buffer("debias_var", torch.ones(teacher_dim))
            self.register_buffer("debias_inited", torch.zeros((), dtype=torch.bool))
        # Round-2 MGD (masked generative): a TRAINING-ONLY decoder that regenerates
        # the debiased teacher target from the channel-masked projected student.
        # Linear -> GELU -> Linear with NO OUTPUT BIAS (so it cannot emit a pure
        # constant = the mean target, "mean-coasting"). Lives on the head (trains
        # with it; unused at inference -> the L_mgd term simply isn't computed).
        self.mgd_decoder: nn.Module | None = None
        if self.mgd_weight > 0.0:
            self.mgd_decoder = nn.Sequential(
                nn.Linear(teacher_dim, teacher_dim),
                nn.GELU(),
                nn.Linear(teacher_dim, teacher_dim, bias=False),
            )

    def _align(self, pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        """Per-token alignment loss + the bare cosine (for logging). pred/target
        are (M, C). cosine: 1 - cos; smoothl1: huber on raw vectors (vae)."""
        pred = pred.float()
        target = target.float()
        if self.loss_type == "smoothl1":
            cos = F.cosine_similarity(pred, target, dim=-1).mean()
            return F.smooth_l1_loss(pred, target), cos
        if self.loss_type == "mse":
            # PHI-S (round-2 arm 3): MSE in the STANDARDIZED target space (debias +
            # per-channel var-equalize upstream). MSE — unlike scale-blind cosine —
            # forces the residual structure, not just the dominant direction. cos
            # is still returned for the logged/probed discrimination metric.
            cos = F.cosine_similarity(pred, target, dim=-1).mean()
            return F.mse_loss(pred, target), cos
        # Eps-INSIDE-sqrt normalize (RMSNorm-style): x*rsqrt(sum(x^2)+eps). A
        # near-zero row (de-meaned target near the EMA mean, or a tiny projected
        # patch) makes F.normalize / .norm() — both sqrt(sum x^2) — have a 1/||x||
        # backward that explodes to NaN (anomaly-traced to this Mul). Putting eps
        # under the sqrt gives a finite backward EVERYWHERE; identical for
        # unit-scale rows. (Additive eps AFTER the norm does NOT fix it: the
        # sqrt inside .norm() still has the inf backward at zero.)
        p = pred * torch.rsqrt(pred.pow(2).sum(dim=-1, keepdim=True) + 1e-12)
        t = target * torch.rsqrt(target.pow(2).sum(dim=-1, keepdim=True) + 1e-12)
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
        if self._anticollapse_on() and self.method in ("eve", "repa", "softdepth", "vae"):
            return self._compute_anticollapse(samples)
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
        1-cos(fine_queries, fine_target) + 1-cos(coarse_queries, coarse_target),
        where each 1-cos is already a mean over its granularity's rows (_align).
        Every query block contributes exactly num_fine+num_coarse rows (the caller
        admits a block only when all nq rows are present), so the m-weighting is
        uniform — this reduces to a PLAIN per-image mean of (fine-mean + coarse-
        mean). The * m / n_tok form is kept only for symmetry with compute()
        (where per-image patch counts genuinely vary); here it cancels."""
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

    # ---- Anti-collapse recipe (spec anticollapse-ablation.md) ----------------
    def _anticollapse_on(self) -> bool:
        return bool(
            self.debias_target
            or self.rkd_dist_weight > 0.0
            or self.rkd_angle_weight > 0.0
            or self.vicreg_var_weight > 0.0
            or self.vicreg_cov_weight > 0.0
            or self.mgd_weight > 0.0
            or self.sigreg_weight > 0.0
        )

    def _pred_for(self, s: dict[str, Any]) -> Tensor:
        """Projected-student per-patch features (m, teacher_dim) for the single-
        projector methods — same selection compute() / _softdepth use."""
        if self.method == "eve":
            return self.proj(s["native"][0])
        if self.method == "softdepth":
            stack = torch.stack([s["native"][k] for k in self.layers], dim=0)  # (L,m,d)
            stack = F.normalize(stack.float(), dim=-1) * (stack.shape[-1] ** 0.5)
            w = torch.softmax(self.depth_logits, dim=0).view(-1, 1, 1)
            mixed = (w * stack).sum(dim=0).to(s["native"][self.layers[0]].dtype)
            return self.proj(mixed)
        return self.proj(s["native"][self.layers[0]])  # repa / vae

    def _compute_anticollapse(
        self, samples: list[dict[str, Any]]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Per-patch cosine on a (optionally) DE-MEANED target [A], plus an
        inter-image RKD relational term over per-image POOLED features [B], plus a
        VICReg variance/covariance floor on the student patches [C]. Each term is
        gated by its weight; all-off would be the plain cosine (but compute() only
        dispatches here when at least one is on)."""
        device = samples[0]["target"].device
        z = torch.zeros((), dtype=torch.float32, device=device)
        preds = [self._pred_for(s).float() for s in samples]  # list of (m_i, d)
        targets = [s["target"].float() for s in samples]  # list of (m_i, d)
        # Sanitize the cosine-path inputs the SAME way _relational_gram sanitizes
        # its Gram inputs. The frozen-LM bf16 forward can emit a non-finite patch
        # hidden on an occasional microbatch (overflow); proj() then carries the
        # inf into _align, whose rsqrt(inf^2) -> NaN poisons BOTH the cosine loss
        # (-> the whole microbatch is non-finite -> the trainer skips it, losing
        # ~10-20% of data) AND the logged distill_cos metric (NaN reduces across
        # ranks -> unreadable). nan_to_num here clamps inf->0 with a 0 gradient at
        # those elements (finite gradient elsewhere), so the cosine is finite, the
        # metric is readable, and no microbatch is dropped. MUST precede the debias
        # EMA update below (an inf target would otherwise poison the running mean).
        preds = [torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0) for p in preds]
        targets = [torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0) for t in targets]

        # (A) debias: subtract the EMA per-channel mean (across images x patches)
        # from the frozen target. Update the running estimate in training only.
        if self.debias_target:
            allt = torch.cat(targets, dim=0)  # (N, d)
            if self.training:
                bmean = allt.mean(dim=0).detach()
                bvar = allt.var(dim=0, unbiased=False).detach()
                # Re-init on first step OR if the running stats are non-finite
                # (defense-in-depth against an uninitialized `to_empty` buffer that
                # init_visual_distill_buffers somehow missed: a garbage-Inf mean
                # would subtract to NaN and skip every microbatch forever).
                if not bool(self.debias_inited) or not torch.isfinite(self.debias_mean).all():
                    self.debias_mean.copy_(bmean)
                    self.debias_var.copy_(bvar)
                    self.debias_inited.fill_(True)  # type: ignore[arg-type]
                else:
                    m = self.debias_momentum
                    self.debias_mean.mul_(m).add_(bmean * (1.0 - m))
                    self.debias_var.mul_(m).add_(bvar * (1.0 - m))
            c = self.debias_mean.to(device=device, dtype=torch.float32)
            targets = [t - c for t in targets]
            if self.debias_std:
                sd = (self.debias_var.to(device=device, dtype=torch.float32) + 1e-6).sqrt()
                targets = [t / sd for t in targets]

        # Per-patch cosine (token-weighted), the base distill.
        loss_sum = z.clone()
        cos_sum = z.clone()
        n_tok = 0
        for p, t in zip(preds, targets):
            li, ci = self._align(p, t)
            m = t.shape[0]
            loss_sum = loss_sum + li * m
            cos_sum = cos_sum + ci * m
            n_tok += m
        cos_loss = loss_sum / max(n_tok, 1)
        total = cos_loss
        comps: dict[str, Tensor] = {
            "distill": cos_loss.detach(),
            "distill_cos": (cos_sum / max(n_tok, 1)).detach(),
        }
        # Initialize the always-emitted anti-collapse keys to 0 (cross-rank key
        # set must be step-deterministic; see compute_distill_loss `keys`).
        if self.rkd_dist_weight > 0.0:
            comps["distill_rkd_d"] = z.clone()
        if self.rkd_angle_weight > 0.0:
            comps["distill_rkd_a"] = z.clone()
        if self.vicreg_var_weight > 0.0:
            comps["distill_vic_var"] = z.clone()
        if self.vicreg_cov_weight > 0.0:
            comps["distill_vic_cov"] = z.clone()
        if self.mgd_weight > 0.0:
            comps["distill_mgd"] = z.clone()
        if self.sigreg_weight > 0.0:
            comps["distill_sigreg"] = z.clone()

        # nan_to_num insurance: any non-finite auxiliary term is zeroed (value AND
        # gradient) so one pathological batch can't poison the weights over a long
        # unattended run — on top of the cdist-backward root-cause fix below.
        def _safe(x: Tensor) -> Tensor:
            return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Linear warmup of the aux weights (0 -> 1). The RKD/VICReg gradients are
        # ill-conditioned on the untrained connector and diverge it in a few steps;
        # ramp them in after the cosine+debias has shaped a sane connector. SIGReg
        # gets its OWN (typically longer) ramp. One shared step counter increments
        # once per training forward when EITHER ramp is active.
        if (self.ac_warmup_steps > 0 or self.sigreg_warmup_steps > 0) and self.training:
            self._ac_step += 1
        step = float(self._ac_step) if hasattr(self, "_ac_step") else 0.0
        wf = min(1.0, step / float(self.ac_warmup_steps)) if self.ac_warmup_steps > 0 else 1.0
        wf_sig = (
            min(1.0, step / float(self.sigreg_warmup_steps))
            if self.sigreg_warmup_steps > 0
            else 1.0
        )

        # (B) RKD relational over per-image POOLED features (B = #images >= 3).
        if (self.rkd_dist_weight > 0.0 or self.rkd_angle_weight > 0.0) and len(preds) >= 3:
            # EPS-FLOORED unit normalization of the pooled mean. With a random
            # connector the projected patches average toward ~0, and F.normalize's
            # backward (1/||x||) -> inf at a near-zero mean -> NaN gradient (the
            # actual root cause; arm A never pools, so it is unaffected). The
            # clamp_min makes the denominator constant where small -> finite grad.
            def _unit(v: Tensor) -> Tensor:
                # Eps-INSIDE-sqrt (RMSNorm-style): finite backward everywhere,
                # including ||v||->0. (clamp_min and additive-after-norm both leave
                # the sqrt-at-zero inf backward intact.)
                return v * torch.rsqrt(v.pow(2).sum() + 1e-12)

            sp = torch.stack([_unit(p.mean(dim=0)) for p in preds])  # (B,d)
            tp = torch.stack([_unit(t.mean(dim=0)) for t in targets])  # (B,d)
            if self.rkd_dist_weight > 0.0:
                # Bounded Gram relational (RKD distance/angle have inf gradients
                # that diverge the connector — see _relational_gram).
                ld = _safe(self._relational_gram(sp, tp))
                comps["distill_rkd_d"] = ld.detach()
                total = total + (wf * self.rkd_dist_weight) * ld
            if self.rkd_angle_weight > 0.0:
                la = _safe(self._rkd_angle(sp, tp))
                comps["distill_rkd_a"] = la.detach()
                total = total + (wf * self.rkd_angle_weight) * la

        # (C) VICReg variance + covariance on the student projected patches.
        if self.vicreg_var_weight > 0.0 or self.vicreg_cov_weight > 0.0:
            zc = torch.cat(preds, dim=0)  # (N, d)
            lv, lc = self._vicreg(zc)
            if self.vicreg_var_weight > 0.0:
                lv = _safe(lv)
                comps["distill_vic_var"] = lv.detach()
                total = total + (wf * self.vicreg_var_weight) * lv
            if self.vicreg_cov_weight > 0.0:
                lc = _safe(lc)
                comps["distill_vic_cov"] = lc.detach()
                total = total + (wf * self.vicreg_cov_weight) * lc

        # Round-2 MGD (masked generative): channel-mask each projected-student
        # patch, regenerate the DEBIASED target through the train-only decoder,
        # 1 - cos(G(p*mask), t̃). Forces the student to carry reconstructable info.
        if self.mgd_weight > 0.0 and self.mgd_decoder is not None:
            keep = 1.0 - self.mgd_beta
            # The decoder is a real (bf16) submodule, but preds are fp32 (_pred_for
            # .float()) and this trainer forward is NOT under autocast — feed the
            # decoder its own param dtype to avoid a Float/BFloat16 matmul error.
            mgd_dtype = self.mgd_decoder[0].weight.dtype
            mgd_sum = z.clone()
            mtok = 0
            for p, t in zip(preds, targets):
                mask = (torch.rand_like(p) < keep).to(p.dtype)  # per-patch channel mask
                g = self.mgd_decoder((p * mask).to(mgd_dtype))
                gl, _ = self._align(g, t)
                mm = t.shape[0]
                mgd_sum = mgd_sum + gl * mm
                mtok += mm
            l_mgd = _safe(mgd_sum / max(mtok, 1))
            comps["distill_mgd"] = l_mgd.detach()
            total = total + self.mgd_weight * l_mgd

        # Round-2 SIGReg (sliced Epps-Pulley isotropy): push the student per-patch
        # features toward N(0, I). Bounded gradient + small weight + long warmup so
        # it regularizes WITHOUT starving the alignment (the round-1 VICReg failure).
        if self.sigreg_weight > 0.0:
            ls = _safe(self._sigreg(torch.cat(preds, dim=0)))
            comps["distill_sigreg"] = ls.detach()
            total = total + (wf_sig * self.sigreg_weight) * ls

        if self.method == "softdepth":
            w = torch.softmax(self.depth_logits.detach().float(), dim=0)
            layer_idx = torch.tensor([float(x) for x in self.layers], device=w.device)
            comps["distill_sel_depth"] = torch.tensor(float((w * layer_idx).sum()), device=device)
            comps["distill_sel_max"] = w.max().to(device)
        return total, comps

    def _relational_gram(self, s: Tensor, t: Tensor) -> Tensor:
        """Inter-image relational via the Manifold/Gram (2107.01378) cosine-
        structure: match the B×B cosine-similarity matrix of the pooled STUDENT to
        the pooled TEACHER's, MSE over the off-diagonal. s, t are L2-normalized
        (B, d), so the Gram is the cosine matrix in [-1, 1] — BOUNDED, no sqrt /
        no division, so gradients stay finite (unlike RKD distance/angle, whose
        1/||edge|| terms produce inf gradients that diverge the untrained
        connector — 0*inf=NaN even under a warmup). A mean-collapsed student has a
        near-all-ones Gram it cannot match to the teacher's diverse one, so the
        term still penalizes collapse — the property we need."""
        # Sanitize the Gram INPUTS (a pathological pooled mean can still slip a
        # non-finite row through _unit; nan_to_num here gives finite forward AND
        # zero-gradient at those rows, so the MSE backward 2*(gs-gt)/N can never be
        # NaN — guarding the OUTPUT alone doesn't, since the NaN is born inside the
        # MSE backward). clamp keeps gs/gt in the valid cosine range.
        s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        mask = ~torch.eye(s.shape[0], dtype=torch.bool, device=s.device)
        gs = (s @ s.t()).clamp(-1.0, 1.0)
        gt = (t @ t.t()).clamp(-1.0, 1.0)
        return F.mse_loss(gs[mask], gt[mask])

    def _rkd_distance(self, s: Tensor, t: Tensor) -> Tensor:
        """RKD distance-wise (1904.05068): match pairwise L2 distances, each side
        normalized by its OWN batch-mean pairwise distance (cancels the shared
        constant exactly), Huber loss over the off-diagonal. (Superseded by
        _relational_gram for stability; kept for reference.)"""
        mask = ~torch.eye(s.shape[0], dtype=torch.bool, device=s.device)

        def norm_pdist(x: Tensor) -> Tensor:
            # Gram-based squared distance + EPS-FLOORED sqrt. torch.cdist's
            # backward is NaN at coincident rows (||xi-xj||=0 -> grad/0); in the
            # early-collapse regime many pooled features are near-identical, so
            # that singularity fires and poisons the connector weights. The
            # clamp_min before sqrt keeps the gradient finite at d=0.
            g = x @ x.t()
            diag = g.diagonal()
            sq = (diag.unsqueeze(0) + diag.unsqueeze(1) - 2.0 * g).clamp_min(1e-8)
            d = sq.sqrt()
            mu = d[mask].mean().clamp_min(1e-6)
            return d / mu

        return F.smooth_l1_loss(norm_pdist(s)[mask], norm_pdist(t)[mask])

    def _rkd_angle(self, s: Tensor, t: Tensor) -> Tensor:
        """RKD angle-wise (1904.05068): match the angle potential cos(e_ij, e_kj)
        over sampled triplets (j the vertex), Huber. Scale-invariant. Subsampled
        to bound the O(B^3) cost."""
        b = s.shape[0]
        n = min(self.rkd_angle_triplets, b * b * b)
        # Sample triplets; vary per call by drawing fresh indices (no global RNG
        # dependence on Math.random — torch.randint is fine here).
        idx = torch.randint(0, b, (n, 3), device=s.device)
        i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]

        def edge(a: Tensor, b: Tensor) -> Tensor:
            # eps-FLOOR the edge norm (not F.normalize's 1e-12): near-collapsed /
            # duplicate in-batch images give ~0-length edges whose normalize
            # Jacobian (~1/||edge||) would explode the gradient to ~1e11.
            d = a - b
            return d / d.norm(dim=-1, keepdim=True).clamp_min(1e-4)

        def angle(x: Tensor) -> Tensor:
            return (edge(x[i], x[j]) * edge(x[k], x[j])).sum(dim=-1)

        return F.smooth_l1_loss(angle(s), angle(t))

    def _vicreg(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """VICReg variance hinge + off-diagonal covariance penalty on the student
        projected per-patch features z (N, d)."""
        z = z.float()
        # RMS-normalize each patch feature to O(1) components BEFORE the variance/
        # covariance. The cosine base loss constrains only DIRECTION (it is scale-
        # invariant), so the projected-feature MAGNITUDE is an unconstrained gauge;
        # a raw variance hinge `max(0, gamma - std)` keeps pushing that magnitude up
        # every step until the bf16 forward overflows (-> inf features -> NaN loss
        # -> the step is skipped -> the connector is frozen mid-divergence; this is
        # exactly how the +C arm got stuck). RMS-normalization bounds the features
        # (no overflow), keeps gamma=1 meaningful (components are O(1)), uses the
        # eps-inside-sqrt rsqrt for a finite backward at ||z||->0, and aligns the
        # variance/covariance floor with the DIRECTIONAL structure the cosine cares
        # about (anti-collapse on directions, not on the irrelevant magnitude).
        z = z * torch.rsqrt(z.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
        var_loss = torch.clamp(self.vicreg_gamma - std, min=0.0).mean()
        n, d = z.shape
        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.t() @ zc) / max(n - 1, 1)  # (d, d)
        off = cov - torch.diag(torch.diagonal(cov))
        cov_loss = (off.pow(2).sum()) / d
        return var_loss, cov_loss

    def _sigreg(self, z: Tensor) -> Tensor:
        """Sliced Epps-Pulley isotropy regularizer (round-2 arm 2). Push the
        student per-patch features z (N, d) toward an isotropic standard normal
        N(0, I) by a characteristic-function (CF) goodness-of-fit test along M
        random unit directions (resampled each step). For direction u and the 1-D
        projection x = z·u, the empirical CF phi_emp(t) = mean_n exp(i t x_n) is
        compared to the N(0,1) CF phi_ref(t) = exp(-t^2/2) at K quadrature knots on
        [-5, 5] under a Gaussian window w(t) = exp(-t^2/2):
            stat = mean_u  sum_k w_k [ (Re phi_emp - phi_ref)^2 + (Im phi_emp)^2 ].
        Do NOT standardize x first — the scale/shape deviation IS the signal, so
        the gradient drives mean->0, var->1, and decorrelates. CF values are
        bounded and the gradient magnitude is O(t_max/N), so this cannot blow up
        the way an unbounded variance hinge can — the whole point of choosing it
        over VICReg."""
        z = z.float()
        n, d = z.shape
        # Subsample patches: the CF statistic needs only a few thousand samples,
        # and the (N, M, K) intermediate below would otherwise scale with the full
        # batch's patch count (OOM at large batch). Resampled each step (stochastic).
        cap = 8192
        if n > cap:
            idx = torch.randint(0, n, (cap,), device=z.device)
            z = z[idx]
            n = cap
        # Random unit directions, resampled every call (no global RNG state; the
        # stochasticity is the slicing, like sliced-Wasserstein). (d, M)
        dirs = torch.randn(d, self.sigreg_dirs, device=z.device, dtype=z.dtype)
        dirs = dirs * torch.rsqrt(dirs.pow(2).sum(dim=0, keepdim=True) + 1e-12)
        proj = z @ dirs  # (N, M) — 1-D projections
        t = torch.linspace(-5.0, 5.0, self.sigreg_knots, device=z.device, dtype=z.dtype)  # (K,)
        arg = proj.unsqueeze(-1) * t  # (N, M, K)
        cos_e = arg.cos().mean(dim=0)  # (M, K) Re phi_emp
        sin_e = arg.sin().mean(dim=0)  # (M, K) Im phi_emp
        ref = torch.exp(-0.5 * t * t)  # (K,) phi_ref = window w (both = exp(-t^2/2))
        stat = (ref * ((cos_e - ref) ** 2 + sin_e**2)).sum(dim=-1)  # (M,)
        return stat.mean()


__all__ = [
    "DISTILL_METHODS",
    "resolve_distill_layers",
    "reconstruct_image_from_patches",
    "VisualDistillTeacher",
    "VisualDistillHead",
]
