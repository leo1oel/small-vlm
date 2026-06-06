from dataclasses import dataclass, field

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
    padding_side: str = "left"


@dataclass
class ConnectorConfig:
    name: str = MISSING
    type: str = MISSING
    # --- raw_patch dials (ignored by other connector types) ---
    mm_embed_dim: int | None = None  # embedder internal width; None = LM hidden_size
    mm_posemb_size: int | None = None  # per-axis posemb rows; None = max_soft_tokens


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
class ModelConfig:
    name: str = MISSING
    visual_encoder: VisualEncoderConfig = field(default_factory=VisualEncoderConfig)
    language_model: LanguageModelConfig = field(default_factory=LanguageModelConfig)
    connector: ConnectorConfig = field(default_factory=ConnectorConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    visual_aux: VisualAuxConfig = field(default_factory=VisualAuxConfig)


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
    folders: dict[str, float] | None = None
    jsonl_name: str = "train.jsonl"
    shuffle_buffer_size: int = 10000
    max_samples_per_sequence: int | None = 100
    # energon owns DataLoader workers AND rank sharding; the HF trainer's
    # dataloader_num_workers must stay 0 for this dataset type.
    num_workers: int = 4
    # Length-grouped batching (type "energon" only): upper edges of the
    # effective-length buckets (post-splice tokens: text + per-image patches
    # + per-audio frames). Samples batch only within their bucket, so padding
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


def register_configs() -> None:
    cs: ConfigStore = ConfigStore.instance()
    cs.store(name="cfg", node=AppConfig)
