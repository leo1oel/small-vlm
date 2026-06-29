from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING  # pyright: ignore


@dataclass
class VisualEncoderConfig:
    # `hf_name: null` in the model yaml selects the encoder-free
    # (gemma4_unified-style) raw-patch path: no vision tower at all; the dials
    # below configure RawImageProcessor. When hf_name is set, the dials are
    # ignored and the classic HF-vision-tower path is used unchanged.
    hf_name: str | None = MISSING
    output_layer: int | None = None
    use_cls_token: bool = False
    use_all_tokens: bool = False
    # --- encoder-free (raw_patch) dials ---
    patch_size: int = 16  # teacher patch edge, px (gemma4 default)
    pooling_kernel_size: int = 3  # k; model patch = patch_size * k px (gemma4: 48px)
    max_soft_tokens: int = 280  # per-image token budget, any positive int
    image_mean: list[float] | None = (
        None  # post-rescale normalize; None = rescale-only (gemma4-style)
    )
    image_std: list[float] | None = None


@dataclass
class LanguageModelConfig:
    hf_name: str = MISSING
    max_seq_length: int | None = None
    # Causal control for the text-prior hypothesis (E1, spec 2026-06-18): after
    # building the model, re-initialize ONLY the LM backbone (embeddings/layers/
    # norm/untied lm_head) to the config initializer, destroying the pretrained
    # language prior while keeping the connector + vision params. Tests whether,
    # with no cheap prior to ride, the native model is forced to use the image.
    # Fresh-build only; False = normal pretrained load (bit-identical baseline).
    random_init: bool = False
    use_start_end_tokens: bool = False
    use_image_patch_token: bool = False
    image_start_token: str = "<im_start>"
    image_end_token: str = "<im_end>"
    image_patch_token: str = "<im_patch>"
    image_token: str = "<image>"
    ignore_index: int = -100
    image_token_index: int = -200
    # Audio placeholder, symmetric with the image one: "<audio>" in the sample
    # text is tokenized then replaced by the (non-vocab) sentinel index, which
    # the splice swaps for audio features.
    audio_token: str = "<audio>"
    audio_token_index: int = -201
    # Learnable-query placeholder (BREEN arXiv:2503.12446 port, spec 2026-06-24):
    # "<query>" in the text is tokenized then replaced by this (non-vocab) sentinel
    # index, which the splice swaps for the model's learnable query Parameter (the
    # CLIP-distillation site). -202 is the next free id after image (-200) /
    # audio (-201). Inert unless model.learnable_query.enabled.
    query_token: str = "<query>"
    query_token_index: int = -202
    padding_side: str = "left"


@dataclass
class ConnectorConfig:
    name: str = MISSING
    type: str = MISSING
    # --- raw_patch dials (ignored by other connector types) ---
    mm_embed_dim: int | None = None  # embedder internal width; None = LM hidden_size
    mm_posemb_size: int | None = None  # per-axis posemb rows; None = max_soft_tokens
    # Low-rank bottleneck inside the shared patch embedding (minit2i/PRX/
    # HiDream-O1): patch_dim -> bottleneck -> mm_embed_dim. 0 = off (single
    # Linear, bit-identical). Shared by understanding + generation.
    bottleneck_dim: int = 0
    # Pretrained-conv "warm tokenizer" (spec 2026-06-22, encoder-free catch-up):
    # re-encode each raw model-patch with a transplanted ViT conv patch-embed
    # (the conv is a per-16px-sub-patch linear; NO cross-patch attention, so the
    # model stays encoder-free) BEFORE the learned projection — swapping
    # random-init pixel features for pretrained low-level visual features so the
    # from-scratch tokenizer isn't from scratch. null = off (bit-identical).
    # "siglip"|"clip" select the source family; weights are transplanted in
    # vlm.load_model on fresh build only (reloads carry the trained stem).
    patch_stem: str | None = None
    patch_stem_name: str = "google/siglip-base-patch16-224"
    patch_stem_kernel: int = 16  # teacher patch edge px; must divide model patch side
    patch_stem_freeze: bool = False  # keep the transplanted conv frozen (preserve features)
    # 2-layer GELU MLP patch head (LLaVA-1.5): patch_dim -> hidden -> GELU ->
    # mm_embed_dim. 0 = off (single Linear). Gives the connector nonlinear
    # capacity to combine the transplanted sub-patch features. Takes precedence
    # over bottleneck_dim when both are set.
    patch_mlp_hidden: int = 0


@dataclass
class AudioConfig:
    """Encoder-free audio pathway (gemma4_unified-style). Disabled by default —
    vision-only configs need not mention this section at all."""

    enabled: bool = False
    # Connector (connector_map key + display name)
    name: str = "raw_waveform"
    type: str = "raw_waveform"
    # Frame size: samples per audio soft token. 640 @ 16kHz = 40ms/token,
    # gemma4-compatible. Changing either requires retraining the audio connector.
    samples_per_token: int = 640
    sampling_rate: int = 16000
    # Per-audio token cap for the dataset side (gemma4: 750 = 30s). None = no cap.
    max_audio_tokens: int | None = 750


@dataclass
class VisualAuxConfig:
    """Visual auxiliary prediction loss at image positions (spec:
    docs/superpowers/specs/2026-06-06-visual-aux-loss-design.md). Structural
    dials only — they size the head module, so they live on the model config
    (the loss weight/layer are trainer dials). "none" = no head built,
    bit-identical baseline path."""

    # none | aim_pixel (next-patch z-scored pixel MSE, AIM/AIMv2-style)
    #      | nepa (next-patch connector-embedding cosine, stop-grad target)
    objective: str = "none"
    # Head MLP: depth 1 = single Linear; depth d = (d-1) x [Linear, GELU] + Linear.
    head_depth: int = 2
    # Internal width of the head MLP (input is always the LM hidden size).
    # Default matches the Qwen3-1.7B hidden size — set explicitly for other
    # backbone sizes (the head does not auto-scale).
    head_hidden: int = 2048


@dataclass
class VisualExpertConfig:
    """Per-decoder-layer modality-routed visual experts (EVEv2 "divide-and-conquer"
    / Mono-InternVL arXiv:2410.08202 / BREEN arXiv:2503.12446 style; spec:
    docs/superpowers/specs/2026-06-14-native-vlm-capacity-readiness-design.md).
    Image (and BREEN <query>) tokens flow through vision-specific copies of the
    selected transformer sublayers, while text tokens use the shared ones, routed
    per-token by the same image+query mask. Three independently-toggleable experts
    share one routing / `layers` / `init_from_text` / `gate` mechanism:
      - ffn:       sibling `mlp.mlp_visual` per layer (Mono-InternVL FFN expert).
      - norm:      sibling RMSNorms `input_layernorm.norm_visual` /
                   `post_attention_layernorm.norm_visual` (EVEv2 modality norms).
      - attention: sibling q/k/v/o projections
                   `self_attn.{q,k,v,o}_proj.proj_visual` (EVEv2 modality
                   attention) — only the projection WEIGHTS are split per-token;
                   RoPE, q_norm/k_norm, the causal / cross-modal mask and the
                   KV-cache are untouched, so the attention pattern is unchanged.
    Structural — the modules attach at build time and serialize into checkpoint
    config.json (visual_aux pattern), so inference rebuilds the same experts
    (train/infer parity). enabled=False = bit-identical baseline (nothing
    attached, no routing); with enabled=True each expert is on/off independently."""

    # Master switch. False = bit-identical baseline (no experts, no routing).
    # Back-compat: this was the FFN-expert toggle and `ffn` defaults True, so an
    # existing `enabled: true` config (no ffn/norm/attention keys) still builds
    # exactly the FFN expert and nothing else.
    enabled: bool = False
    # FFN expert (Mono-InternVL): sibling `mlp.mlp_visual` per selected layer.
    ffn: bool = True
    # Norm expert (EVEv2): sibling input/post-attention RMSNorms per selected layer.
    norm: bool = False
    # Attention expert (EVEv2): sibling q/k/v/o projections per selected layer.
    attention: bool = False
    # null = every decoder layer; else explicit 0-based layer indices. Shared by
    # all enabled experts.
    layers: list[int] | None = None
    # Fresh-build only: initialize each visual sibling by copying the matching
    # text weights (so training starts identical, then diverges). Shared by all
    # enabled experts; makes enabling an expert on a pretrained ckpt a step-0 no-op
    # (modulo the gate below).
    init_from_text: bool = True
    # Per-expert sigmoid gate (BREEN arXiv:2503.12446: F.sigmoid(gate(x))*expert(x)
    # on both the text and visual paths). A localized fidelity upgrade over the
    # hard 0/1 mask, applied UNIFORMLY to every enabled expert (FFN/norm/attention):
    # each routed sublayer gets its own Linear(in_dim, 1) gate pair initialized
    # near-identity (zero weight, bias 4 -> sigmoid(4)≈0.982) — close to identity
    # but NOT a literal t=0 no-op: it attenuates each sublayer by ~1.8% from step 0.
    # False = bit-identical to the hard-mask routing.
    gate: bool = False


@dataclass
class VisualPrefixConfig:
    """Dedicated internal "visual prefix" stack (spec 2026-06-14, early-capacity
    arm; NEO pre-Buffer style). K bidirectional layers process each image's
    connector tokens BEFORE they enter the shared LLM — a data-grown visual
    encoder with its own params, no imported ViT. Structural: built next to the
    connector and serialized into checkpoint config.json. enabled=False = no
    module (bit-identical baseline)."""

    enabled: bool = False
    # Number of bidirectional prefix layers (NEO 2B uses ~0.3*depth).
    depth: int = 6
    # 0 -> inherit the LM's num_attention_heads / intermediate_size.
    heads: int = 0
    intermediate: int = 0


@dataclass
class GroundingConfig:
    """Image-grounding margin loss (spec 2026-06-18): the gold answer must be
    more likely WITH the real image than with a blanked image. A pure training
    loss (no module/head), so all dials live on the model config and are read in
    chunked_ce_forward; weight=0.0 = bit-identical baseline (loss never built).
    Directly optimizes the R0 (intact - swap) quantity the FDI probe measures —
    pushes the model out of the language-prior basin so it conditions on pixels."""

    enabled: bool = False
    # Loss weight λ_g: L = L_CE + λ_g · L_ground. 0.0 -> off.
    weight: float = 0.0
    # Hinge margin m (nats): the gold-token logp with the real image must beat
    # the blanked-image logp by at least m, else a per-token penalty applies.
    margin: float = 1.0
    # How to build the negative ("no usable image") pass. "blank": zero the
    # image-position embeddings (shape-preserving ablation; the robust default).
    corruption: str = "blank"


@dataclass
class VisualDistillConfig:
    """Visual-encoder distillation for the native VLM (spec 2026-06-21): align
    the LLM's hidden states at image positions to a frozen vision encoder's
    per-patch features, so the visual pathway is supervised directly instead of
    discovered through next-token loss alone. Structural dials (they size the
    projection head + select the teacher, and serialize into checkpoint
    config.json — visual_aux pattern) live here; the loss WEIGHT is a trainer
    dial. enabled=False = no head built, bit-identical baseline."""

    enabled: bool = False
    # repa | eve | vora | softdepth | relational | vae | breen
    # (see models/visual_distill.py). "breen" distills the learnable queries
    # (not image patches) to a dual-granularity avg-pooled CLIP grid.
    method: str = "repa"
    # Teacher: "clip" loads a CLIPVisionModel, "vae" a diffusers AutoencoderKL.
    teacher_kind: str = "clip"
    teacher_name: str = "openai/clip-vit-base-patch16"
    # Square edge (px) the reconstructed RGB is resized to before the teacher.
    # 224 (default) -> a 16x16 CLIP-B/16 grid; BREEN uses 336 with
    # clip-vit-large-patch14-336 -> a 24x24 grid (= 8x8 fine + 6x6 coarse pools).
    teacher_out_size: int = 224
    # 1-based decoder-output layer index/indices to align. null = method default
    # (repa/relational/vae: ~0.3 depth; vora: first half block-wise; softdepth:
    # all intermediate layers as the selection pool; eve: ignored, uses final).
    layers: list[int] | None = None
    # Internal width of the MLP projection head (input = LM hidden). 0 = LM hidden.
    head_hidden: int = 0
    # Per-token alignment: "cosine" (neg cosine, CLIP) or "smoothl1" (huber, vae).
    # "" = method default (cosine for clip, smoothl1 for vae).
    loss: str = ""
    # --- Anti-collapse recipe (see AGENTS.md "Anti-collapse distill port (ST-2)") ---
    # All default OFF -> bit-identical to the plain per-patch cosine distill.
    # Applied (eve/repa/softdepth/vae single-projector path) in compute().
    # (A) debias_target: subtract a running EMA per-CHANNEL mean (across the
    # batch's images x patches) from the frozen TEACHER target before the cosine,
    # removing the shared "mean-image" constant the plain cosine collapses onto.
    # debias_std also divides by the EMA per-channel std (standardize). This is
    # ACROSS-SAMPLE per-channel centering (an EMA over training), NOT the within-
    # vector centering cosine already does.
    debias_target: bool = False
    debias_momentum: float = 0.9
    debias_std: bool = False
    # (B) RKD relational (1904.05068) on per-image POOLED features over the batch:
    # distance-wise (rkd_dist) + angle-wise (rkd_angle) structure match. Shift/
    # scale-invariant -> a mean-collapsed student structurally cannot satisfy it.
    # 0 = off; paper's RKD-only weights are dist=1, angle=2.
    rkd_dist_weight: float = 0.0
    rkd_angle_weight: float = 0.0
    rkd_angle_triplets: int = 512
    # (C) VICReg variance+covariance floor on the STUDENT projected per-patch
    # features (over the patch axis). Backstop against dimensional collapse. 0=off.
    vicreg_var_weight: float = 0.0
    vicreg_cov_weight: float = 0.0
    vicreg_gamma: float = 1.0
    # Linear warmup (in optimizer steps) for the RKD + VICReg weights, ramped
    # 0 -> full. The relational/variance terms are ill-conditioned on the
    # untrained connector (random pooled features -> huge early gradients that
    # diverge the connector in a few steps at the connector LR); the warmup lets
    # the cosine+debias establish a sane connector first. 0 = no warmup.
    ac_warmup_steps: int = 0
    # --- Round-2 anti-collapse levers (see AGENTS.md "Anti-collapse distill port (ST-2)"), each on TOP of
    # A+B (debias + RKD). All default OFF. ---
    # MGD (masked generative distillation): per-patch channel-mask the projected
    # student, regenerate the DEBIASED teacher target through a tiny train-only
    # decoder (Linear->GELU->Linear, no output bias -> denies mean-coasting),
    # cosine to the target. Forces the student to encode enough to reconstruct.
    # mgd_weight = lambda on L_mgd (target ~0.1-0.3x the A-cosine); 0 = off.
    mgd_weight: float = 0.0
    mgd_beta: float = 0.5  # channel drop prob (keep = 1 - beta)
    # SIGReg (sliced isotropy regularizer, "C done right"): a sliced Epps-Pulley
    # characteristic-function test pushing the student per-patch features toward
    # N(0, I) over M random unit directions (resampled each step), bounded
    # gradient. Small weight + long warmup so it does NOT starve alignment the
    # way VICReg (round-1 arm C) did. 0 = off.
    sigreg_weight: float = 0.0
    sigreg_dirs: int = 256  # M random projection directions per step
    sigreg_knots: int = 17  # quadrature knots on [-5, 5]
    sigreg_warmup_steps: int = 0  # linear 0->full ramp (separate from ac_warmup)


@dataclass
class LearnableQueryConfig:
    """Learnable CLIP-distillation queries (BREEN arXiv:2503.12446 port, spec
    2026-06-24). A trainable nn.Parameter(num_fine+num_coarse, hidden) on the
    ForCausalLM, spliced in at each "<query>" placeholder (one block per image),
    routed to the visual FFN expert, and label-masked (excluded from CE). The
    breen distill method aligns the first num_fine query rows to the 8x8 fine
    avg-pool of the CLIP grid and the last num_coarse rows to the 6x6 coarse
    pool. Structural — it sizes the Parameter, so it lives on the model config
    and serializes into checkpoint config.json (visual_aux pattern).
    enabled=False = no Parameter built, bit-identical baseline."""

    enabled: bool = False
    # 8x8 fine pool of a 24x24 CLIP-L/14-336 grid = 64; 6x6 coarse pool = 36.
    num_fine: int = 64
    num_coarse: int = 36
    # Where the data path emits the "<query>" placeholder relative to the image
    # and the user text: "after_image" (BREEN pretrain: image then queries) |
    # "after_text" (BREEN SFT: queries attend the question, placed after it).
    placement: str = "after_image"


@dataclass
class CrossModalMaskConfig:
    # "none" (default, bit-identical baseline) | "prefix_lm" | "img2q_window".
    # prefix_lm: bidirectional attention over [system+image+question], causal
    # suffix, loss unchanged (PaliGemma masking). img2q_window: image-query
    # rows attend question keys in decoder layers window[0]..window[1] only
    # (1-based, inclusive) — the forced-early-fusion arm.
    mode: str = "none"
    window: list[int] = field(default_factory=lambda: [1, 9])
    # Mutual windowing (also confining text->image attention to the window)
    # is NOT implemented in v1: it requires per-layer decode-step masking.
    # The field exists so the config surface is stable; True is rejected.
    bidirectional: bool = False


@dataclass
class GenerationConfig:
    """Text->image generation pathway: continuous flow matching in patch space
    (Transfusion / minit2i MM-JiT style; spec 2026-06-20). Structural dials size
    the x-prediction head + in-context timestep token and set the noise
    schedule; they serialize into checkpoint config.json (visual_aux pattern) so
    the sampler self-describes. enabled=False = understanding-only (no module
    built, no generation forward — bit-identical baseline)."""

    enabled: bool = False
    # Square target canvas edge (px). Patch grid = (resolution // patch_size)^2.
    resolution: int = 384
    # Generation patch edge (px). Generation REUSES the encoder-free connector's
    # patch space, so this MUST equal the connector model patch
    # (visual_encoder.patch_size * pooling_kernel_size, default 16*3 = 48).
    # resolution % patch_size must be 0.
    patch_size: int = 48
    # Flow-matching noise scale (minit2i: 2.0) and logit-normal timestep
    # sampling t = sigmoid(N(t_mu, t_sigma))  (t=1 clean, t=0 pure noise).
    noise_scale: float = 2.0
    t_mu: float = -0.8
    t_sigma: float = 0.8
    # Sampler defaults (inference only): Euler steps + classifier-free guidance.
    sample_steps: int = 100
    cfg_scale: float = 1.0
    # Train-time probability of dropping the text condition (CFG training).
    label_drop: float = 0.1
    # Independent generation embedder (decouples generation from the 48px
    # understanding connector). When True, generation builds its own raw-patch
    # embedder + x-head at embed_patch_size (e.g. 16px -> finer grid, smaller
    # per-patch target -> sharper images; minit2i 32px/256-token mechanism).
    # When False, generation reuses the connector at patch_size (48px, legacy).
    independent_embed: bool = False
    embed_patch_size: int = 16
    # Per-axis position-table size for the independent embedder (>= grid side =
    # resolution // embed_patch_size).
    embed_posemb_size: int = 64
    # Low-rank bottleneck in the generation patch embedder (minit2i/PRX/
    # HiDream-O1 "BottleneckPatchEmbed"): patch_dim -> bottleneck -> hidden. 0 =
    # off (single Linear). HiDream-O1 uses hidden//4 (=512 at hidden 2048).
    embed_bottleneck_dim: int = 0
    # Perceptual losses on the unpatchified x0 prediction vs the clean image
    # (PRX/HiDream-O1 pixel-space recipe; weights from PRX configs). Off by
    # default — pure flow-matching MSE baseline is bit-identical.
    perceptual_enabled: bool = False
    perceptual_lpips_weight: float = 0.1
    perceptual_dino_weight: float = 0.01
    perceptual_lpips_net: str = "vgg"
    perceptual_dino_model: str = "dinov2_vitb14_reg"
    # Resize the x0/gt images to this square edge before the perceptual nets
    # (memory; PRX uses 256 for DINO).
    perceptual_resize: int = 256
    # Noise gating (PixelGen sec 3.4): only apply perceptual losses to LOW-noise
    # samples (t > gate; our convention t=1 clean). PixelGen disables the first
    # 30% high-noise steps -> 0.3. 0 = apply to all samples (PRX behaviour).
    perceptual_t_gate: float = 0.3
    # Warmup: train flow-MSE ONLY for the first N steps, then add perceptual.
    # The zero-init x-head makes step-0 predictions a constant (grey) image; the
    # LPIPS/DINO feature-normalization gradient is singular there and overflows
    # DeepSpeed's bf16 loss scaler (collapses -> training stalls). Warming up
    # lets the flow loss produce a non-degenerate x0 first (PixelGen-style staged
    # training). Counted in micro-batch forwards (== optimizer steps at
    # grad_accum=1); 0 = no warmup.
    perceptual_warmup_steps: int = 1000
    # 2D (axial/interleaved-MRoPE) rotary for the image tokens (FLUX/minit2i/
    # HiDream-O1 via Qwen3-VL): vertically-adjacent patches become close in
    # rotary space. Bit-identical for text/understanding (replaces model.
    # rotary_emb with an MRoPE that reduces to 1D on equal-axis positions).
    rope_2d: bool = True


@dataclass
class ModelConfig:
    name: str = MISSING
    visual_encoder: VisualEncoderConfig = field(default_factory=VisualEncoderConfig)
    language_model: LanguageModelConfig = field(default_factory=LanguageModelConfig)
    connector: ConnectorConfig = field(default_factory=ConnectorConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    visual_aux: VisualAuxConfig = field(default_factory=VisualAuxConfig)
    visual_expert: VisualExpertConfig = field(default_factory=VisualExpertConfig)
    learnable_query: LearnableQueryConfig = field(default_factory=LearnableQueryConfig)
    visual_prefix: VisualPrefixConfig = field(default_factory=VisualPrefixConfig)
    cross_modal_mask: CrossModalMaskConfig = field(default_factory=CrossModalMaskConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    visual_distill: VisualDistillConfig = field(default_factory=VisualDistillConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


@dataclass
class DatasetConfig:
    name: str = MISSING
    # type "json": local LLaVA-style json/jsonl/yaml-mixture (path + image_folder
    #   required — validated at load time, not by the schema, because the
    #   "energon" type legitimately omits them).
    # type "energon": stream samples from Azure Blob via Megatron-Energon; data
    #   location comes from `folders` instead of path/image_folder.
    type: str = "json"
    path: str | None = None
    lazy_preprocess: bool = True
    is_multimodal: bool = True
    early_mix_text: bool = False
    image_folder: str | None = None
    audio_folder: str | None = None  # root for samples' relative "audio" paths
    image_aspect_ratio: str = "square"
    image_token: str = "<image>"
    # --- streaming dials (type: "energon" only) ---
    # blob folder -> blend weight; a single entry means no blending. Each folder
    # must contain a prepared <jsonl_name> (auto-prepared on first use).
    # Mutually exclusive with `wds_path` (the prepared-WebDataset layout below).
    folders: dict[str, float] | None = None
    jsonl_name: str = "train.jsonl"
    # Prepared-WebDataset layout (type "energon" only; mutually exclusive with
    # `folders`). Point this at an `energon prepare` output directory — tar
    # shards (`{00000..NNNNN}.tar`) + a `.nv-meta/` dir holding a
    # `CrudeWebdataset` dataset.yaml — to stream samples whose image bytes are
    # bundled IN the tar (one sequential GET per ~10k samples) instead of the
    # loose-file jsonl layout (one Azure GET per image). Selects the WDS cooker
    # + loader branch (vlm.data.energon_wds). Accepts a full
    # `msc://<profile>/<container>/...` URL or a container-relative path
    # (resolved through the same MSC profile/container as `folders`). The in-tar
    # `.json` carries the SAME {id, source, messages} schema as the jsonl path.
    wds_path: str | None = None
    # Conversation structure of this dataset's samples, used to validate the
    # trainer template choice (#5). "instruct" = multi-turn chat data, which
    # needs a chat template (e.g. trainer.version=qwen_2_5); pairing it with the
    # 2-turn `plain` caption template silently drops all human text except media
    # placeholders. "caption" = 2-turn caption data, compatible with `plain`.
    # "auto" (default) skips the check (back-compat for json datasets and
    # unclassified mixtures).
    conversation_kind: str = "auto"
    shuffle_buffer_size: int = 10000
    max_samples_per_sequence: int | None = 100
    # energon owns DataLoader workers AND rank sharding; the HF trainer's
    # dataloader_num_workers must stay 0 for this dataset type.
    num_workers: int = 4
    # Length-grouped batching (type "energon" only): upper edges of the
    # effective-length buckets (post-splice tokens: text + per-image patches
    # + per-audio frames + per-query BREEN rows when learnable_query is on).
    # Samples batch only within their bucket, so padding
    # is bounded by bucket width. None = no bucketing. Buckets are
    # worker-local — keep them few and wide so each fills promptly.
    length_buckets: list[int] | None = None
    # Token-budget batching (needs length_buckets): each bucket flushes
    # batch_token_budget // bucket_edge samples per batch, giving every
    # micro-batch ~constant effective tokens — uniform GPU memory and large
    # batches on short buckets. Samples-per-step then VARIES (tokens stay
    # ~constant); trainer.per_device_train_batch_size becomes the loader
    # default only. None = fixed batch size per bucket.
    batch_token_budget: int | None = None
    use_local_jsonl: bool | None = None  # None = prefer a local jsonl copy if present
    # Strip the empty `<think>\n\n</think>` prefix that distilled caption sets
    # (e.g. Bee Stage-1) prepend to every assistant turn, so the boilerplate
    # never enters the loss. Whitespace-bodied blocks only — real reasoning
    # text is preserved. Energon path only.
    strip_empty_think: bool = False
    # Image-placeholder layout inside human turns (plan 2026-06-10, access
    # arms): "keep" preserves today's layout on both paths; "question_first"
    # / "sandwich" / "random" rewrite single-image human turns. Applied on
    # the energon path after placeholder injection and on the local-json
    # path after preprocess_multimodal's image-first normalization.
    image_position: str = "keep"
    # --- generation task (text->image flow matching; spec 2026-06-20) ---
    # "understanding" (default) = image->text chat encoder; "generation" =
    # caption->image: each record yields (caption prompt, target image patches
    # at a fixed canvas). Selects VLMGenTaskEncoder in build_energon_train_loader.
    task: str = "understanding"
    # Generation target canvas edge (px); must be a multiple of the connector
    # model patch (visual_encoder.patch_size * pooling_kernel_size, default 48).
    gen_resolution: int = 384
    # Max caption (prompt) tokens for the generation condition.
    gen_caption_max_len: int = 128
    # Generation target patch edge (px). null -> use the connector's 48px model
    # patch (legacy, reuse-the-connector path). Set to model.generation.
    # embed_patch_size (e.g. 16) when model.generation.independent_embed is on,
    # so the dataloader patchifies targets at the SAME granularity the model's
    # independent gen embedder + x-head expect (else 768-vs-6912 dim mismatch).
    gen_patch_size: int | None = None


@dataclass
class UnfreezeConfig:
    train_vision_model: bool = True
    train_language_model: bool = True
    train_connector: bool = True


@dataclass
class LearningRateConfig:
    visual_encoder_learning_rate: float = 1e-4
    language_model_learning_rate: float = 1e-4
    connector_learning_rate: float = 1e-4
    default_lr: float = 1e-4


@dataclass
class WeightDecayConfig:
    visual_encoder_weight_decay: float = 0.0
    language_model_weight_decay: float = 0.0
    connector_weight_decay: float = 0.0
    default_wd: float = 0.0


@dataclass
class TrainerConfig:
    name: str = MISSING
    output_dir: str = "."
    unfreeze: UnfreezeConfig = field(default_factory=UnfreezeConfig)
    learning_rate: LearningRateConfig = field(default_factory=LearningRateConfig)
    weight_decay: WeightDecayConfig = field(default_factory=WeightDecayConfig)
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 4
    # Precision flags: None means auto-detect; True/False means user override
    bf16: bool | None = None
    fp16: bool = False
    tf32: bool | None = None
    deepspeed: str | None = None
    num_train_epochs: int = 1
    # Required (> 0) for dataset.type="energon": the streaming loader has no
    # epoch length, so scheduling/stopping must be step-based.
    max_steps: int = -1
    save_strategy: str = "steps"
    save_steps: int = 5000
    save_total_limit: int | None = 20  # None = keep every checkpoint
    save_only_model: bool = False
    logging_steps: int = 1
    # transformers v5 deprecated `warmup_ratio` in favor of `warmup_steps`, which
    # accepts a float < 1 interpreted as a ratio of total training steps.
    warmup_steps: float = 0.0
    lr_scheduler_type: str = "linear"
    gradient_accumulation_steps: int = 1
    # transformers v5 default is the string "none"; passing None yields [None] in
    # post_init (not []), which breaks reporting-integration resolution.
    report_to: str = "none"
    dataloader_num_workers: int = 4
    dataloader_prefetch_factor: int | None = None
    version: str = "v0"
    group_by_length: bool = False
    sequential_sampling: bool = False
    group_by_modality_length: bool = False
    gradient_checkpointing: bool = False
    run_name: str = "small-vlm"
    resume_from_checkpoint: str | None = None
    from_pretrained: str | None = None
    seed: int = 42
    attn_implementation: str | None = "flash_attention_2"
    optim: str = "adamw_torch_fused"
    # Training-only chunked cross-entropy (0 = off): drop ignore_index
    # positions before the lm_head matmul, then compute fp32 CE over hidden
    # chunks of this many tokens — never materializing the full
    # (batch*seq, vocab) logits (~25GB fp32 at bs4/seq4k with the 152k vocab).
    # Numerically replicates transformers' ForCausalLMLoss mean reduction.
    loss_chunk_size: int = 0
    # Aux-exit deep supervision for the early-fusion ablation (spec:
    # docs/superpowers/specs/2026-06-05-aux-exit-loss-design.md): at each
    # listed decoder layer k (1-based output index, valid [1, n_layers-1])
    # decode through the SHARED final RMSNorm + lm_head and add the CE to
    # the main loss: L = L_final + aux_exit_weight * sum_k L_k. Empty = off
    # (bit-identical baseline path). Requires loss_chunk_size > 0.
    aux_exit_layers: list[int] = field(default_factory=list)
    # EE-LLM's validated few-exit weight range is 0.1-0.5 (arXiv:2312.04916);
    # only read when aux_exit_layers is non-empty.
    aux_exit_weight: float = 0.25
    # True = detach the shared norm/lm_head weights inside the aux branch so
    # its gradient flows only into layers <= k (fuse for the tied-embedding
    # gradient coupling, arXiv:2603.26663); default False follows the
    # LayerSkip shared-with-grad recipe (arXiv:2404.16710).
    aux_exit_detach: bool = False
    # Visual-aux loss weight λ (spec 2026-06-06): only read when
    # model.visual_aux.objective != "none". L = L_CE + λ·L_visual.
    # AIMv2's literature prior is α=0.4 (arXiv:2411.14402); 0.5 is the
    # user-set value for both v1 arms (spec decision 2026-06-06).
    visual_aux_weight: float = 0.5
    # null = attach the head to the post-final-norm last hidden state;
    # k = decode layer k's output through the shared final RMSNorm first
    # (aux-exit capture mechanism; valid [1, n_layers-1]).
    visual_aux_layer: int | None = None
    # Optimizer dials for the (always-trainable) head; None falls back to
    # default_lr / language_model weight decay.
    visual_aux_head_lr: float | None = None
    visual_aux_head_wd: float | None = None
    # Visual-distill loss weight λ_d (spec 2026-06-21): only read when
    # model.visual_distill.enabled. L = L_CE + λ_d·L_distill. REPA/VoRA use an
    # equal-ish weighting (~0.5-1.0); the head trains with the LM (it falls
    # through to the language_model optimizer group, so no separate lr dial).
    visual_distill_weight: float = 1.0
    # Native transformers token accounting, surfaced per log step in wandb
    # (num_input_tokens_seen + train tokens/sec): "non_padding" sums
    # attention_mask across ranks (small per-step gather), "all" counts
    # padding too, "no" disables. Counts input_ids-level tokens — media
    # sentinels count as 1 (the FLOPs metric uses spliced length instead).
    include_num_input_tokens_seen: str = "no"
    # Collective-op watchdog timeout (seconds). Streaming + token-budget
    # bucketing makes batch-to-batch latency fat-tailed (a rank can wait on a
    # slow bucket flush / Azure stall while peers sit in allreduce) — the
    # 10-min NCCL default killed a run at step 4848; 1h tolerates stalls and
    # lets the run continue instead of paying a full requeue+restore.
    ddp_timeout: int = 3600
    # torch.compile via the HF Trainer (inductor). The multimodal splice and
    # chunked CE graph-break, but the decoder stack (the compute bulk) still
    # compiles; variable spliced lengths settle into dynamic-shape graphs
    # after the first recompile. Validate on a 50-step trial before enabling
    # in production.
    torch_compile: bool = False


@dataclass
class InferenceConfig:
    checkpoint_path: str = MISSING
    num_inference_samples: int | None = None
    chat_template: str = "plain"


@dataclass
class AppConfig:
    is_training: bool = True
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def validate_dataset_config(
    dataset: Any,
    model: Any = None,
    trainer: Any = None,
) -> None:
    """Fail loud on dataset/model/trainer config combinations that are otherwise
    silently mis-trained. Pure attribute reads (works on both the dataclasses
    here and the OmegaConf structured config), so it is callable from
    validate_config (full cfg, fails fast in main()) and from
    build_energon_train_loader (dataset only, guards direct callers/tests).

    Checks:
      * #27 — `batch_token_budget` set without `length_buckets`: the loader only
        builds the bucketed (token-budget) encoder when `length_buckets` is set,
        so the budget would be silently ignored.
      * #14 — generation patch-size mismatch: when the model runs an INDEPENDENT
        generation embedder (`model.generation.independent_embed`), the dataset
        must patchify targets at the embedder's `embed_patch_size`, else the
        per-patch target dim (psz**2*3) mismatches the gen embedder / x-head.
      * #5  — `plain` (2-turn caption) template composed with `instruct`
        (multi-turn) data: `preprocess_plain` keeps only media placeholders + the
        final answer and drops all other human text, silently corrupting samples.
    """
    # --- #27: batch_token_budget needs length_buckets -----------------------
    batch_token_budget = getattr(dataset, "batch_token_budget", None)
    length_buckets = getattr(dataset, "length_buckets", None)
    if batch_token_budget is not None and not length_buckets:
        raise ValueError(
            "dataset.batch_token_budget is set but dataset.length_buckets is empty "
            "— token-budget batching only takes effect together with length "
            "buckets (the loader builds the bucketed encoder only when "
            "length_buckets is set). Set dataset.length_buckets, or clear "
            "dataset.batch_token_budget."
        )

    # --- #14: generation data/model patch-size must match -------------------
    if model is not None:
        gen = getattr(model, "generation", None)
        if (
            gen is not None
            and bool(getattr(gen, "enabled", False))
            and bool(getattr(gen, "independent_embed", False))
            and str(getattr(dataset, "task", "understanding")) == "generation"
        ):
            embed_patch_size = int(getattr(gen, "embed_patch_size", 16))
            gen_patch_size = getattr(dataset, "gen_patch_size", None)
            if gen_patch_size is None or int(gen_patch_size) != embed_patch_size:
                raise ValueError(
                    "dataset.gen_patch_size must equal model.generation."
                    f"embed_patch_size ({embed_patch_size}) when model.generation."
                    "independent_embed is enabled — the dataloader patchifies "
                    "generation targets at dataset.gen_patch_size while the model's "
                    "independent gen embedder / x-head expect embed_patch_size, so "
                    "a mismatch makes the per-patch target dim (psz**2*3) wrong "
                    f"(got dataset.gen_patch_size={gen_patch_size!r})."
                )

    # --- #5: plain template requires caption (2-turn) data -----------------
    if trainer is not None:
        version = str(getattr(trainer, "version", "v0"))
        kind = str(getattr(dataset, "conversation_kind", "auto"))
        if version in {"plain", "v0_plain"} and kind == "instruct":
            raise ValueError(
                f"trainer template version={version!r} (2-turn caption preprocessing) "
                "is composed with an instruct (multi-turn) dataset "
                f"(dataset.conversation_kind='instruct', dataset.name="
                f"{getattr(dataset, 'name', '?')!r}). The plain preprocessor keeps "
                "only media placeholders + the final answer and drops all other "
                "human text. Use a caption-only dataset, or select a chat template "
                "(e.g. trainer.version=qwen_2_5)."
            )


def register_configs() -> None:
    cs: ConfigStore = ConfigStore.instance()
    cs.store(name="cfg", node=AppConfig)
