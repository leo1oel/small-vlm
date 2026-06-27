import logging
import re
from abc import ABC, abstractmethod
from typing import Any, override

import torch
import torch.nn as nn
from torch import Tensor

log: logging.Logger = logging.getLogger(__name__)


class Connector(nn.Module, ABC):
    """
    Abstract base class for all connector modules.
    Connectors are responsible for projecting visual features to a space
    compatible with text features.
    """

    def __init__(
        self,
        config: Any,
        image_hidden_size: int,
        text_hidden_size: int,
    ) -> None:
        super().__init__()
        self.config: Any = config
        self.name: str = self.config.name
        self.image_hidden_size: int = image_hidden_size
        self.text_hidden_size: int = text_hidden_size
        self.projection_layer: nn.Module = self._build_projection_layer()

    @abstractmethod
    def _build_projection_layer(self) -> nn.Module:
        pass

    @override
    def forward(self, visual_features: Tensor) -> Tensor:
        return self.projection_layer(visual_features)


class IdentityConnector(Connector):
    def __init__(
        self,
        config: Any,
        image_hidden_size: int,
        text_hidden_size: int,
    ) -> None:
        if image_hidden_size != text_hidden_size:
            log.warning(
                f"IdentityConnector initialized with image_hidden_size ({image_hidden_size}) "
                f"!= text_hidden_size ({text_hidden_size}). Features will pass through unchanged."
            )
        super().__init__(config, image_hidden_size, text_hidden_size)

    @override
    def _build_projection_layer(self) -> nn.Module:
        return nn.Identity()


class LinearConnector(Connector):
    def __init__(
        self,
        config: Any,
        image_hidden_size: int,
        text_hidden_size: int,
    ) -> None:
        super().__init__(config, image_hidden_size, text_hidden_size)

    @override
    def _build_projection_layer(self) -> nn.Module:
        return nn.Linear(
            self.image_hidden_size,
            self.text_hidden_size,
        )


class MLPConnector(Connector):
    ACTIVATION_MAP: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,  # Swish/SiLU
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }

    @override
    def __init__(
        self,
        config: Any,
        image_hidden_size: int,
        text_hidden_size: int,
    ) -> None:
        self.num_layers: int = 2
        self.activation_name: str = "gelu"

        # Parse num_layers and activation_name from the connector's name string
        self._parse_config_name(config.name)

        super().__init__(config, image_hidden_size, text_hidden_size)

    def _parse_config_name(self, name: str) -> None:
        pattern = r"mlp_(\d+)_(\w+)"  # e.g., mlp_2_gelu, mlp_3_relu
        match = re.match(pattern, name)
        if match:
            try:
                self.num_layers = int(match.group(1))
                self.activation_name = match.group(2).lower()
                if self.activation_name not in self.ACTIVATION_MAP:
                    log.warning(
                        f"MLPConnector: Activation '{self.activation_name}' from name '{name}' is not recognized. "
                        f"Falling back to default activation '{MLPConnector.activation_name}'. "
                        f"Supported: {list(self.ACTIVATION_MAP.keys())}"
                    )
                    self.activation_name = "gelu"  # Fallback to default if parsed name is invalid
            except ValueError:
                log.warning(
                    f"MLPConnector: Could not parse num_layers from '{match.group(1)}' in name '{name}'. "
                    f"Using default num_layers: {self.num_layers}."
                )
        else:
            log.warning(
                f"MLPConnector name '{name}' does not match pattern 'mlp_NUMLAYERS_ACTIVATION'. "
                f"Using defaults: num_layers={self.num_layers}, activation_name='{self.activation_name}'."
            )

    @override
    def _build_projection_layer(self) -> nn.Module:
        if self.num_layers < 1:
            raise ValueError(
                f"MLPConnector: Number of layers must be at least 1, got {self.num_layers}"
            )

        activation_class = self.ACTIVATION_MAP.get(self.activation_name)
        if activation_class is None:
            # This case should ideally be handled by _parse_config_name fallback,
            # but as a safeguard:
            log.error(
                f"MLPConnector: Unsupported activation function '{self.activation_name}'. "
                f"Supported activations: {list(self.ACTIVATION_MAP.keys())}. "
                f"Defaulting to GELU."
            )
            activation_class = nn.GELU  # Fallback

        layers: list[nn.Module] = []

        for i in range(self.num_layers):
            # The first layer maps from image_hidden_size to text_hidden_size.
            # Subsequent hidden layers map from text_hidden_size to text_hidden_size.
            # The final layer also outputs text_hidden_size.
            input_dim = self.image_hidden_size if i == 0 else self.text_hidden_size
            output_dim = (
                self.text_hidden_size
            )  # All layers in the MLP project towards/maintain text_hidden_size

            layers.append(nn.Linear(input_dim, output_dim))

            # Add activation function for all layers except the last one
            if i < self.num_layers - 1:
                layers.append(activation_class())

        return nn.Sequential(*layers)


class RMSNormNoScale(nn.Module):
    """Gemma-style RMSNorm without a learnable scale.

    Faithful to gemma4_unified's `Gemma4UnifiedRMSNorm(with_scale=False)`
    (modeling_gemma4_unified.py L167-186): computed in float32, uses
    `pow(mean_sq + eps, -0.5)` rather than rsqrt, no weight parameter.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim: int = dim
        self.eps: float = eps

    @override
    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x = x.float()
        x = x * torch.pow(x.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        return x.to(input_dtype)


class _RawPatchEmbedder(nn.Module):
    """Encoder-free vision embedder, faithful to gemma4_unified's
    Gemma4UnifiedVisionEmbedder + Gemma4UnifiedMultimodalEmbedder
    (modeling_gemma4_unified.py L775-861):

        pre-patchified raw RGB patches (N, patch_dim) + XY coords (N, 2)
          -> LayerNorm(patch_dim) -> Linear(patch_dim -> mm_embed_dim) -> LayerNorm
          -> + factorized XY position embedding -> LayerNorm
          -> RMSNorm(no scale) -> Linear(mm_embed_dim -> text_hidden, bias=False)

    Patchification lives in RawImageProcessor (dataloader side, like
    gemma4_unified's image processor), so this module is pure projection.
    Variable-resolution by construction: any grid shape works as long as both
    grid sides are < posemb_size.
    """

    def __init__(
        self,
        patch_dim: int,
        mm_embed_dim: int,
        posemb_size: int,
        text_hidden_size: int,
        bottleneck_dim: int | None = None,
        patch_stem: str | None = None,
        patch_stem_kernel: int = 16,
        patch_mlp_hidden: int | None = None,
    ) -> None:
        super().__init__()
        # Optional pretrained-conv patch stem (encoder-free "warm tokenizer",
        # spec 2026-06-22): re-encode each raw model-patch with a transplanted
        # ViT conv patch-embed (e.g. SigLIP-B/16's Conv2d(3,768,16,16)) BEFORE
        # the learned projection. The conv is a per-16px-sub-patch linear with
        # NO attention across sub-patches or across the image, so the model
        # stays encoder-free — it only swaps random-init pixel features for
        # pretrained low-level visual features. Output dim is kept == patch_dim
        # (out_ch * subgrid^2 == 3 * side^2, e.g. 768*3*3 == 3*48*48 == 6912) so
        # the rest of the embedder is structurally unchanged and bit-identical
        # when off. Weights are transplanted post-build in vlm.load_model.
        self.patch_stem: nn.Conv2d | None = None
        self._stem_side: int = 0
        if patch_stem:
            side = int(round((patch_dim // 3) ** 0.5))  # model patch edge, px (48)
            if side * side * 3 != patch_dim:
                raise ValueError(
                    f"patch_stem requires a square RGB patch; patch_dim {patch_dim} "
                    f"is not 3*side^2 (got side={side})."
                )
            if side % int(patch_stem_kernel) != 0:
                raise ValueError(
                    f"patch_stem_kernel {patch_stem_kernel} must divide the model "
                    f"patch side {side}px."
                )
            sub = side // int(patch_stem_kernel)  # sub-patch grid per model patch (3)
            if patch_dim % (sub * sub) != 0:
                raise ValueError(
                    f"patch_stem: patch_dim {patch_dim} not divisible by subgrid^2 "
                    f"{sub * sub}; cannot keep output dim == patch_dim."
                )
            out_ch = patch_dim // (sub * sub)  # 6912 // 9 == 768 (== teacher hidden)
            self.patch_stem = nn.Conv2d(
                3, out_ch, kernel_size=int(patch_stem_kernel), stride=int(patch_stem_kernel)
            )
            self._stem_side = side
            # Per-channel input normalization the transplanted conv was trained
            # with (default SigLIP/ImageNet-standard 0.5 -> (x-0.5)/0.5 == 2x-1;
            # init_patch_stem_from_encoder overwrites with the teacher's true
            # stats, e.g. OpenAI-CLIP mean/std). Persistent so reloads restore
            # the CLIP stats (the transplant runs on fresh build only).
            self.register_buffer("stem_mean", torch.full((1, 3, 1, 1), 0.5), persistent=True)
            self.register_buffer("stem_std", torch.full((1, 3, 1, 1), 0.5), persistent=True)
            # Forward-level freeze enforcement (the connector LR group / delta
            # tuning re-enable requires_grad after build, so a plain
            # requires_grad=False is not enough): when set, the conv weight is
            # detached in forward so no gradient ever reaches it. Persistent bool.
            self.register_buffer("_stem_frozen", torch.zeros((), dtype=torch.bool), persistent=True)
        self.patch_ln1: nn.LayerNorm = nn.LayerNorm(patch_dim)
        # patch projection: a single Linear by default (gemma4 understanding
        # path, bit-identical). When bottleneck_dim is set, factor it through a
        # low-rank intermediate (patch_dim -> bottleneck -> mm_embed_dim, no bias
        # on the first, no activation between) — the minit2i/PRX/HiDream-O1
        # "BottleneckPatchEmbed" trick: a raw image patch's intrinsic dimension
        # is far below patch_dim, so denoising in pixel space benefits from
        # projecting onto the signal subspace before the transformer width.
        self.patch_dense: nn.Module
        if patch_mlp_hidden and int(patch_mlp_hidden) > 0:
            # 2-layer GELU MLP head (LLaVA-1.5 arXiv:2310.03744): a real
            # nonlinear projection, strictly stronger than a single linear (the
            # single linear must collapse the 9 transplanted SigLIP sub-patch
            # features into one token with no nonlinearity). Takes precedence
            # over the linear bottleneck when both are set.
            self.patch_dense = nn.Sequential(
                nn.Linear(patch_dim, int(patch_mlp_hidden)),
                nn.GELU(),
                nn.Linear(int(patch_mlp_hidden), mm_embed_dim),
            )
        elif bottleneck_dim and int(bottleneck_dim) > 0:
            self.patch_dense = nn.Sequential(
                nn.Linear(patch_dim, int(bottleneck_dim), bias=False),
                nn.Linear(int(bottleneck_dim), mm_embed_dim),
            )
        else:
            self.patch_dense = nn.Linear(patch_dim, mm_embed_dim)
        self.patch_ln2: nn.LayerNorm = nn.LayerNorm(mm_embed_dim)
        # Factorized per-axis (X/Y) position table, shape (posemb_size, 2, D):
        # embedding = table[x, 0] + table[y, 1]  (gemma4_unified L798, L823-827).
        # O(side) parameters instead of O(side^2); supports arbitrary grid shapes.
        self.pos_embedding: nn.Parameter = nn.Parameter(torch.empty(posemb_size, 2, mm_embed_dim))
        nn.init.normal_(self.pos_embedding, std=0.02)
        self.pos_norm: nn.LayerNorm = nn.LayerNorm(mm_embed_dim)
        self.pre_projection_norm: RMSNormNoScale = RMSNormNoScale(mm_embed_dim)
        self.projection: nn.Linear = nn.Linear(mm_embed_dim, text_hidden_size, bias=False)

    @override
    def forward(self, patches: Tensor, position_ids: Tensor) -> Tensor:
        """patches (N, patch_dim) + position_ids (N, 2) int XY -> (N, text_hidden).

        N may be the concatenation of several images' patches (the caller packs
        variable-length per-image patch lists and splits the output back).
        """
        if position_ids.numel() > 0 and int(position_ids.max()) >= self.pos_embedding.shape[0]:
            raise ValueError(
                f"RawPatchConnector: patch coordinate {int(position_ids.max())} exceeds "
                f"position table size {self.pos_embedding.shape[0]}; increase connector "
                f"mm_posemb_size (it must cover the largest possible grid side, i.e. "
                f">= visual_encoder.max_soft_tokens for worst-case aspect ratios)."
            )
        patches = patches.to(self.patch_ln1.weight.dtype)
        if self.patch_stem is not None:
            # (N, side*side*3) flattened row-major as (row, col, channel)
            # [RawImageProcessor.convert_image_to_patches] -> (N, 3, side, side).
            n = patches.shape[0]
            s = self._stem_side
            img = patches.view(n, s, s, 3).permute(0, 3, 1, 2).to(self.patch_stem.weight.dtype)
            # Raw patches are rescale-only ([0,1]); renorm to the teacher conv's
            # training distribution (SigLIP 0.5/0.5 -> 2x-1; CLIP -> OpenAI stats).
            img = (img - self.stem_mean.to(img.dtype)) / self.stem_std.to(img.dtype)
            w, b = self.patch_stem.weight, self.patch_stem.bias
            if bool(self._stem_frozen):  # robust freeze: no grad reaches the conv
                w = w.detach()
                b = b.detach() if b is not None else None
            feat = nn.functional.conv2d(img, w, b, stride=self.patch_stem.stride)  # (N,C,sub,sub)
            patches = feat.flatten(1).to(self.patch_ln1.weight.dtype)  # (N, patch_dim)
        hidden = self.patch_ln2(self.patch_dense(self.patch_ln1(patches)))
        pos = self.pos_embedding[position_ids[:, 0], 0] + self.pos_embedding[position_ids[:, 1], 1]
        hidden = self.pos_norm(hidden + pos.to(hidden.dtype))
        return self.projection(self.pre_projection_norm(hidden))


class RawPatchConnector(Connector):
    """Encoder-free 'vision tower as a connector' (gemma4_unified-style).

    No SigLIP/CLIP/ViT: consumes raw flattened RGB patches produced by
    RawImageProcessor (variable resolution, aspect-ratio preserving) and projects
    them into LM space with a factorized 2D position embedding.
    Use with `visual_encoder.hf_name: null`.

    Interface note: forward takes (patches, position_ids) — it deviates from the
    single-argument base Connector.forward on purpose. The only call site is the
    encoder-free branch in modeling_vlm, which packs per-image variable-length
    patch tensors, calls this connector once, and splits the result by per-image
    patch counts. Spatial information exists ONLY here (the LM uses plain 1D
    RoPE), so position_ids are mandatory.

    Config contract (auto-wired by vlm.load_model from the model yaml):
      - image_hidden_size == (patch_size * pooling_kernel_size)**2 * 3,
        derived by load_model into vision_config.hidden_size
      - mm_embed_dim: internal embed dim (default: text_hidden_size)
      - mm_posemb_size: per-axis position-table size; REQUIRED — load_model
        auto-fills it with visual_encoder.max_soft_tokens (worst-case grid side)
    """

    def __init__(
        self,
        config: Any,
        image_hidden_size: int,
        text_hidden_size: int,
    ) -> None:
        self.mm_embed_dim: int = getattr(config, "mm_embed_dim", None) or text_hidden_size
        posemb_size: int | None = getattr(config, "mm_posemb_size", None)
        if not posemb_size:
            raise ValueError(
                "RawPatchConnector: connector mm_posemb_size is not set. It is "
                "auto-filled from visual_encoder.max_soft_tokens by vlm.load_model; "
                "set it explicitly in the connector config if constructing manually."
            )
        self.posemb_size: int = posemb_size
        # Optional low-rank bottleneck inside the shared patch embedding
        # (minit2i/PRX/HiDream-O1 trick). 0 = off -> single Linear, bit-identical
        # to the original understanding connector. Shared by understanding AND
        # generation (encoder-free: one patch embedding replaces the ViT).
        self.bottleneck_dim: int = int(getattr(config, "bottleneck_dim", 0) or 0)
        # Pretrained-conv "warm tokenizer" (spec 2026-06-22): null = off.
        self.patch_stem: str | None = getattr(config, "patch_stem", None) or None
        self.patch_stem_kernel: int = int(getattr(config, "patch_stem_kernel", 16) or 16)
        self.patch_mlp_hidden: int = int(getattr(config, "patch_mlp_hidden", 0) or 0)
        super().__init__(config, image_hidden_size, text_hidden_size)

    @override
    def _build_projection_layer(self) -> nn.Module:
        return _RawPatchEmbedder(
            patch_dim=self.image_hidden_size,
            mm_embed_dim=self.mm_embed_dim,
            posemb_size=self.posemb_size,
            text_hidden_size=self.text_hidden_size,
            bottleneck_dim=self.bottleneck_dim if self.bottleneck_dim > 0 else None,
            patch_stem=self.patch_stem,
            patch_stem_kernel=self.patch_stem_kernel,
            patch_mlp_hidden=self.patch_mlp_hidden if self.patch_mlp_hidden > 0 else None,
        )

    @override
    def forward(self, visual_features: Tensor, position_ids: Tensor) -> Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self.projection_layer(visual_features, position_ids)


class _RawWaveformEmbedder(nn.Module):
    """The entire learned audio path of gemma4_unified
    (Gemma4UnifiedMultimodalEmbedder, modeling_gemma4_unified.py L836-861):

        raw 16kHz waveform frames (..., samples_per_token)
          -> RMSNorm(no scale) -> Linear(samples_per_token -> text_hidden, bias=False)

    Each 640-sample frame (40ms @ 16kHz) becomes one audio soft token. There is
    no spectrogram and no conv frontend — the LM learns acoustic representations
    itself. Validity masking of padded frames is the caller's responsibility
    (modeling layer), exactly as in gemma4_unified (modeling L1078-1090).
    """

    def __init__(self, feature_dim: int, text_hidden_size: int) -> None:
        super().__init__()
        self.pre_projection_norm: RMSNormNoScale = RMSNormNoScale(feature_dim)
        self.projection: nn.Linear = nn.Linear(feature_dim, text_hidden_size, bias=False)

    @override
    def forward(self, features: Tensor) -> Tensor:
        """(..., samples_per_token) -> (..., text_hidden).

        Leading dims are free: works for a padded batch (B, T, 640) or for
        packed per-audio frames (ΣT_i, 640) — the modeling layer chooses.
        """
        features = features.to(self.projection.weight.dtype)
        return self.projection(self.pre_projection_norm(features))


class RawWaveformConnector(Connector):
    """Audio 'tower' for the encoder-free unified model (gemma4_unified-style).

    The whole learned audio pathway is this one connector: with
    samples_per_token=640 and Qwen3-1.7B hidden 2048 it is a single
    640x2048 matrix (~1.3M params).

    Interface note: `image_hidden_size` (base-class naming) carries the audio
    frame size here — samples_per_token, i.e. 640 for gemma4-compatible
    40ms @ 16kHz frames produced by Gemma4UnifiedAudioFeatureExtractor
    (feature_size == audio_samples_per_token == 640; the wiring derives it
    from the audio config, never hand-set).
    """

    @override
    def _build_projection_layer(self) -> nn.Module:
        return _RawWaveformEmbedder(
            feature_dim=self.image_hidden_size,
            text_hidden_size=self.text_hidden_size,
        )


class _VisualPrefixLayer(nn.Module):
    """One bidirectional pre-norm SwiGLU transformer block — a set-encoder over
    an image's connector tokens. No RoPE: 2D spatial position already lives in
    the connector's factorized XY posemb, so the prefix is a permutation-aware
    set encoder, not a sequence model. Standard Qwen-style sublayers (RMSNorm,
    separate Q/K/V/O, SwiGLU) so the geometry matches the backbone."""

    def __init__(self, dim: int, n_heads: int, intermediate: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"VisualPrefix dim {dim} not divisible by n_heads {n_heads}")
        self.n_heads: int = n_heads
        self.head_dim: int = dim // n_heads
        self.input_norm: nn.RMSNorm = nn.RMSNorm(dim, eps=eps)
        self.q_proj: nn.Linear = nn.Linear(dim, dim, bias=False)
        self.k_proj: nn.Linear = nn.Linear(dim, dim, bias=False)
        self.v_proj: nn.Linear = nn.Linear(dim, dim, bias=False)
        self.o_proj: nn.Linear = nn.Linear(dim, dim, bias=False)
        self.post_norm: nn.RMSNorm = nn.RMSNorm(dim, eps=eps)
        self.gate_proj: nn.Linear = nn.Linear(dim, intermediate, bias=False)
        self.up_proj: nn.Linear = nn.Linear(dim, intermediate, bias=False)
        self.down_proj: nn.Linear = nn.Linear(intermediate, dim, bias=False)
        for lin in (
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.o_proj,
            self.gate_proj,
            self.up_proj,
            self.down_proj,
        ):
            nn.init.normal_(lin.weight, std=0.02)

    @override
    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        """x (B, n, dim); attn_mask (B, 1, 1, n) additive (-inf on padded keys).
        Bidirectional within each image's valid tokens."""
        b, n, dim = x.shape
        h = self.input_norm(x)
        q = self.q_proj(h).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        o = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.o_proj(o.transpose(1, 2).reshape(b, n, dim))
        h = self.post_norm(x)
        x = x + self.down_proj(nn.functional.silu(self.gate_proj(h)) * self.up_proj(h))
        return x


class VisualPrefix(nn.Module):
    """K bidirectional layers applied to each image's connector tokens BEFORE
    they enter the shared LLM (NEO pre-Buffer / spec 2026-06-14 "early-capacity"
    arm). Grows a data-trained visual encoder *inside* the model with dedicated
    params, instead of importing a ViT. Operates per-image (tokens of one image
    attend only among themselves), batched via padding + a key-padding mask."""

    def __init__(self, dim: int, depth: int, n_heads: int, intermediate: int) -> None:
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList(
            [_VisualPrefixLayer(dim, n_heads, intermediate) for _ in range(depth)]
        )

    @override
    def forward(self, features: list[Tensor]) -> list[Tensor]:
        """features: list of per-image (N_i, dim). Returns the same layout."""
        if not features:
            return features
        sizes = [int(f.shape[0]) for f in features]
        m, maxn, dim = len(features), max(sizes), int(features[0].shape[1])
        x = features[0].new_zeros((m, maxn, dim))
        keep = torch.zeros((m, maxn), dtype=torch.bool, device=features[0].device)
        for i, f in enumerate(features):
            x[i, : sizes[i]] = f
            keep[i, : sizes[i]] = True
        # (B,1,1,n) additive mask: a valid query attends to all valid keys of its
        # own image, never to padding; padded query rows are computed then dropped.
        attn_mask = torch.zeros((m, 1, 1, maxn), dtype=x.dtype, device=x.device)
        attn_mask = attn_mask.masked_fill(~keep[:, None, None, :], float("-inf"))
        for layer in self.layers:
            x = layer(x, attn_mask)
        return [x[i, : sizes[i]] for i in range(m)]


# --- Connector Mapping and Exports ---

# This map is used by your _build_connector function to instantiate the correct connector type.
# The keys ('identity', 'linear', 'mlp', ...) should match the `connector_config.type` values.
connector_map: dict[str, type[Connector]] = {
    "identity": IdentityConnector,
    "linear": LinearConnector,
    "mlp": MLPConnector,
    "raw_patch": RawPatchConnector,
    "raw_waveform": RawWaveformConnector,
}
