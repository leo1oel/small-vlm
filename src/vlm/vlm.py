import logging
import sys
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer, set_seed

from vlm.config.config_schema import LanguageModelConfig

from .config import (
    AppConfig,
    ModelConfig,
    TrainerConfig,
    register_configs,
    validate_dataset_config,
)
from .data import get_data_args, make_supervised_data_module
from .models import VLMProcessor, get_dynamic_vlm
from .models.image_processing_raw import RawImageProcessor
from .train import get_training_args, train, validate_energon_args
from .utils import conversation as conversation_lib
from .utils.precision import resolve_precision

log: logging.Logger = logging.getLogger(name=__name__)
CONFIG_PATH: Path = Path(__file__).resolve().parent / "config"


def _cross_modal_prefix_skip_ids(version: str, tokenizer: Any) -> list[int] | None:
    """Chat-delimiter token ids the conversation preprocessor unmasks into the
    labels. The xmodal prefix detector must skip them, else the leading delimiter
    at position 0 makes the first "supervised" label position 0 and collapses the
    prefix to empty (xmodal_mask._prefix). Both the Qwen ChatML template
    (<|im_start|>/<|im_end|>/newline) and the llama3 template
    (<|begin_of_text|>/<|start_header_id|>/<|end_header_id|>/<|eot_id|>/\\n\\n)
    blanket-unmask their delimiters everywhere; gemma and the rest supervise
    answer tokens only, so return None there (prefix-from-labels is already
    correct). Key off the ACTIVE conversation template (resolved the same way the
    preprocess dispatch does in train.py) so the skip set always matches the
    preprocessor, not the raw version string. Drops unk/negative/non-int ids and
    collapses an empty result to None."""
    template = conversation_lib.conv_templates.get(version)
    template_version = template.version if template is not None else None
    unk = tokenizer.unk_token_id
    cand: list[int] = []
    if template_version == "qwen":
        cand = [
            tokenizer.convert_tokens_to_ids("<|im_start|>"),
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
            *tokenizer.encode("\n", add_special_tokens=False),
        ]
    elif template_version == "llama_v3":
        cand = [
            tokenizer.convert_tokens_to_ids("<|begin_of_text|>"),
            tokenizer.convert_tokens_to_ids("<|start_header_id|>"),
            tokenizer.convert_tokens_to_ids("<|end_header_id|>"),
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            *tokenizer.encode("\n\n", add_special_tokens=False),
        ]
    return [int(t) for t in cand if isinstance(t, int) and t >= 0 and t != unk] or None


def add_special_tokens(tokenizer: PreTrainedTokenizer, config: LanguageModelConfig) -> None:
    """Adds special tokens to the tokenizer if they don't exist."""
    # Create a mapping of tokens to their attribute names

    token_mapping = []
    if config.use_image_patch_token:
        token_mapping.append(config.image_patch_token)
    if config.use_start_end_tokens:
        token_mapping.append(config.image_start_token)
        token_mapping.append(config.image_end_token)

    # Identify which tokens need to be added
    tokens_to_add: list[str] = []
    for token in token_mapping:
        if token is None:
            continue
        if tokenizer.convert_tokens_to_ids(token) == tokenizer.unk_token_id:
            tokens_to_add.append(token)
            log.info(f"Token '{token}' does not exist in tokenizer, will be added")
        else:
            token_id = tokenizer.convert_tokens_to_ids(token)
            log.info(f"Token '{token}' exists in tokenizer, ID: {token_id}")

    # Add all new tokens at once if any
    log.info(f"Tokens to add: {tokens_to_add}")
    if tokens_to_add:
        log.info(f"Adding tokens: {tokens_to_add}")
        tokenizer.add_tokens(tokens_to_add, special_tokens=True)


def attach_audio_feature_extractor(processor: VLMProcessor, audio_config: Any) -> None:
    """Attach (or null out) the audio feature extractor on the processor.

    The extractor is fully derivable from the model's audio_config (it is just
    "chunk 16kHz waveform into samples_per_token frames"), so it is rebuilt
    here on every load instead of being persisted with the processor — this
    also keeps it out of ProcessorMixin's attribute machinery. We reuse
    transformers' Gemma4UnifiedAudioFeatureExtractor verbatim (zero code:
    feature_size == audio_samples_per_token frames + bool validity mask).
    Accepts either the yaml dict (fresh build) or the AudioConfig object from a
    checkpoint's model config (reload path).
    """

    def get(key: str, default: Any) -> Any:
        if isinstance(audio_config, dict):
            return audio_config.get(key, default)
        return getattr(audio_config, key, default)

    samples_per_token, sampling_rate = 640, 16000
    enabled = False
    if audio_config is not None:
        enabled = bool(get("enabled", False))
        samples_per_token = int(get("samples_per_token", samples_per_token))
        sampling_rate = int(get("sampling_rate", sampling_rate))
    if not enabled:
        processor.feature_extractor = None
        return
    from transformers.models.gemma4_unified.feature_extraction_gemma4_unified import (
        Gemma4UnifiedAudioFeatureExtractor,
    )

    processor.feature_extractor = Gemma4UnifiedAudioFeatureExtractor(
        feature_size=samples_per_token,
        sampling_rate=sampling_rate,
        padding_value=0.0,
        audio_samples_per_token=samples_per_token,
    )


def attach_distill_teacher(model: Any) -> None:
    """Build and attach the frozen distillation teacher (spec 2026-06-21).

    Stored list-wrapped (`model._distill_teacher = [teacher]`) so it is NOT a
    registered submodule: invisible to .parameters()/optimizer/state_dict — it
    must never be trained (set_trainable would otherwise sweep it into the
    language_model group) nor bloat checkpoints. Covers both the fresh-build and
    from_pretrained reload paths; the projection head itself is built by the
    causal __init__ from the (serialized) config fields. No-op when disabled.
    """
    if not bool(getattr(model.config, "visual_distill", False)):
        return
    if getattr(model, "visual_distill_head", None) is None:
        raise ValueError(
            "model.visual_distill.enabled is set but the loaded checkpoint has no "
            "visual_distill_head — retrofit from a non-distill checkpoint is not "
            "supported; train from scratch or load a checkpoint trained with it"
        )
    from .models.visual_distill import VisualDistillTeacher

    teacher = VisualDistillTeacher(
        kind=str(getattr(model.config, "visual_distill_teacher_kind", "clip")),
        name=str(
            getattr(model.config, "visual_distill_teacher_name", "openai/clip-vit-base-patch16")
        ),
        out_size=int(getattr(model.config, "visual_distill_teacher_out_size", 224) or 224),
    )
    want_dim = int(getattr(model.config, "visual_distill_teacher_dim", 0) or 0)
    if want_dim and teacher.feature_dim != want_dim:
        raise ValueError(
            f"distill teacher feature_dim {teacher.feature_dim} != head output width "
            f"{want_dim} on config — teacher changed since the head was built"
        )
    model._distill_teacher = [teacher]
    log.info(
        f"Attached frozen distill teacher: {teacher.kind}:{teacher.name} "
        f"(feature_dim={teacher.feature_dim}, off the module tree)"
    )


def init_patch_stem_from_encoder(model: Any, model_cfg: ModelConfig) -> None:
    """Transplant a pretrained ViT conv patch-embed into the connector's warm
    tokenizer stem (spec 2026-06-22). Fresh-build only — the conv is a registered
    submodule that trains and serializes normally, so reloads carry the trained
    stem and must NOT be re-transplanted. No-op unless connector.patch_stem is set.

    The stem stays encoder-free: it is a per-16px-sub-patch linear (Conv2d with
    kernel==stride) applied independently to each raw model-patch, with no
    attention across sub-patches or the image — it only seeds the from-scratch
    tokenizer with pretrained low-level visual features instead of random init.
    """
    kind = str(getattr(model_cfg.connector, "patch_stem", None) or "") or None
    if not kind:
        return
    embedder = getattr(getattr(model.model, "connector", None), "projection_layer", None)
    stem = getattr(embedder, "patch_stem", None)
    if stem is None:
        raise ValueError(
            "connector.patch_stem is set but the connector has no patch_stem conv — "
            "the warm tokenizer is only built by RawPatchConnector (encoder-free)."
        )
    name = str(getattr(model_cfg.connector, "patch_stem_name", "google/siglip-base-patch16-224"))
    # Rescale-only inputs are assumed by the stem's [-1,1] renorm; a non-default
    # image_mean/std would feed the transplanted conv a distribution it never saw.
    if (
        model_cfg.visual_encoder.image_mean is not None
        or model_cfg.visual_encoder.image_std is not None
    ):
        raise ValueError(
            "connector.patch_stem assumes rescale-only patches ([0,1] -> [-1,1] "
            "renorm inside the stem); set visual_encoder.image_mean/std to null."
        )
    if kind == "siglip":
        from transformers import SiglipVisionModel

        vit = SiglipVisionModel.from_pretrained(name)
    elif kind == "clip":
        from transformers import CLIPVisionModel

        vit = CLIPVisionModel.from_pretrained(name)
    else:
        raise ValueError(f"connector.patch_stem must be 'siglip'|'clip'|null, got {kind!r}")
    # The conv patch-embed is the single Conv2d in either vision model; locate it
    # by type so the transplant is robust to module-path differences across
    # transformers versions (e.g. the .vision_model wrapper was flattened in 5.x).
    convs = [m for m in vit.modules() if isinstance(m, torch.nn.Conv2d)]
    if len(convs) != 1:
        raise ValueError(
            f"expected exactly one Conv2d patch-embed in {kind} {name}, found {len(convs)}"
        )
    src = convs[0]
    if tuple(src.weight.shape) != tuple(stem.weight.shape):
        raise ValueError(
            f"patch_stem conv shape mismatch: teacher {tuple(src.weight.shape)} vs "
            f"stem {tuple(stem.weight.shape)} — teacher hidden/patch must match the "
            f"model patch geometry (out_ch == patch_dim/subgrid^2, kernel == "
            f"patch_stem_kernel)."
        )
    with torch.no_grad():
        stem.weight.copy_(src.weight.to(stem.weight.dtype))
        if stem.bias is not None:
            if src.bias is not None:
                stem.bias.copy_(src.bias.to(stem.bias.dtype))
            else:  # CLIP conv has no bias; leave the stem's at zero-init
                stem.bias.zero_()
        # Set the input-normalization stats to the ones the teacher conv was
        # trained with, so its pretrained features are evaluated on-distribution.
        # SigLIP uses ImageNet-standard 0.5/0.5 (== the embedder's default);
        # CLIP uses the OpenAI stats. Buffers live on the embedder (stem_mean/std).
        if kind == "clip":
            from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

            dev = embedder.stem_mean.device
            embedder.stem_mean.copy_(torch.tensor(OPENAI_CLIP_MEAN, device=dev).view(1, 3, 1, 1))
            embedder.stem_std.copy_(torch.tensor(OPENAI_CLIP_STD, device=dev).view(1, 3, 1, 1))
    if bool(getattr(model_cfg.connector, "patch_stem_freeze", False)):
        stem.weight.requires_grad_(False)
        if stem.bias is not None:
            stem.bias.requires_grad_(False)
        # Robust freeze (set_trainable / delta-tuning re-enable connector grads
        # after this): the embedder also detaches the conv weight in forward.
        embedder._stem_frozen.fill_(True)
    log.info(
        f"Transplanted {kind} conv patch-embed ({name}) into connector warm "
        f"tokenizer stem {tuple(stem.weight.shape)}"
        + (" [FROZEN]" if bool(getattr(model_cfg.connector, "patch_stem_freeze", False)) else "")
    )


def require_generation_modules(model: Any, requested: bool) -> None:
    """Load-time guard (spec 2026-06-20): enabling generation training on a
    reloaded understanding-only checkpoint (which carries no gen_x_head /
    gen_t_embed) would route any target_patches batch into forward_generation
    and call those None modules. Fail loud here instead, mirroring the
    visual-aux retrofit guard."""
    if requested and getattr(model, "gen_x_head", None) is None:
        raise ValueError(
            "model.generation.enabled is set but the loaded checkpoint has no "
            "generation modules (gen_x_head/gen_t_embed) — enabling generation "
            "on an understanding-only checkpoint is not supported; train from "
            "scratch with generation enabled or load a generation checkpoint"
        )


def load_model(model_cfg: ModelConfig, trainer_cfg: TrainerConfig):
    log.info(
        f"Loading model: [bold red][link=file://{CONFIG_PATH / 'model' / f'{model_cfg.name}.yaml'}]{model_cfg.name}[/link][/bold red]"
    )

    if trainer_cfg.from_pretrained:
        log.info(f"Loading processor from pretrained: {trainer_cfg.from_pretrained}")
        processor = VLMProcessor.from_pretrained(
            trainer_cfg.from_pretrained,
        )
        log.info(f"Loading model from pretrained: {trainer_cfg.from_pretrained}")
        add_special_tokens(processor.tokenizer, model_cfg.language_model)
        VLMForCausalLM, VLMConfig = get_dynamic_vlm(model_cfg.language_model.hf_name)
        model: VLMForCausalLM = VLMForCausalLM.from_pretrained(
            trainer_cfg.from_pretrained,
            dtype=torch.bfloat16
            if trainer_cfg.bf16
            else torch.float16
            if trainer_cfg.fp16
            else torch.float32,
            attn_implementation=trainer_cfg.attn_implementation,
        )
        if model_cfg.language_model.max_seq_length is not None:
            processor.tokenizer.model_max_length = model_cfg.language_model.max_seq_length
            model.config.max_seq_length = model_cfg.language_model.max_seq_length
        if model.config.vocab_size < len(processor.tokenizer):
            model.model.resize_token_embeddings(len(processor.tokenizer))
        # Audio pathway follows the checkpoint's model config on reload.
        attach_audio_feature_extractor(processor, getattr(model.config, "audio_config", None))
        # Visual-aux retrofit guard: enabling the head on an understanding-only
        # checkpoint needs post-hoc init wiring that is deliberately out of v1
        # scope (spec §7) — fail loud instead of silently training without it.
        if (
            str(model_cfg.visual_aux.objective) != "none"
            and getattr(model, "visual_aux_head", None) is None
        ):
            raise ValueError(
                "model.visual_aux.objective is set but the loaded checkpoint has "
                "no visual_aux_head — retrofit from understanding-only checkpoints "
                "is not supported yet; train from scratch (sft-unified-aimpixel / "
                "sft-unified-nepa) or load a checkpoint trained with the head"
            )
        # Generation retrofit guard (spec 2026-06-20): a composed config can turn
        # on generation training, but a reload rebuilds the x-prediction head /
        # in-context timestep embedder ONLY when the checkpoint config already had
        # generation enabled. Without them, a batch carrying target_patches routes
        # to forward_generation and calls gen_t_embed(...)/gen_x_head(...) on None.
        # Mirror the visual-aux guard: fail loud at load instead of crashing.
        require_generation_modules(model, bool(model_cfg.generation.enabled))
    else:
        if model_cfg.visual_encoder.hf_name is None:
            # Encoder-free (gemma4_unified-style raw-patch) path: no vision tower,
            # so there is no HF vision config to inherit — vision_config is just
            # the yaml dials plus one derived field.
            vision_config = cast(dict[str, Any], OmegaConf.to_container(model_cfg.visual_encoder))
            # Single source of truth for the connector input dim:
            # (patch_size * pooling_kernel_size)^2 * 3. Derived, never hand-set.
            model_patch = (
                model_cfg.visual_encoder.patch_size * model_cfg.visual_encoder.pooling_kernel_size
            )
            vision_config["hidden_size"] = model_patch**2 * 3
        else:
            hf_config = AutoConfig.from_pretrained(model_cfg.visual_encoder.hf_name)
            if getattr(hf_config, "vision_config", None):
                hf_config = hf_config.vision_config
            vision_config_dict = hf_config if isinstance(hf_config, dict) else hf_config.to_dict()
            vision_config = vision_config_dict | OmegaConf.to_container(model_cfg.visual_encoder)
        connector_config = cast(dict[str, Any], OmegaConf.to_container(model_cfg.connector))
        if connector_config.get("type") == "raw_patch" and not connector_config.get(
            "mm_posemb_size"
        ):
            # The factorized position table must cover the worst-case grid side
            # (a budget-long single-row strip), so default to the token budget.
            connector_config["mm_posemb_size"] = model_cfg.visual_encoder.max_soft_tokens
        # Audio pathway: only materialize an audio_config when enabled, so
        # vision-only models keep audio_config=None (no audio_connector built).
        audio_config = OmegaConf.to_container(model_cfg.audio) if model_cfg.audio.enabled else None
        hf_config = AutoConfig.from_pretrained(model_cfg.language_model.hf_name)
        if model_cfg.language_model.max_seq_length is None:
            model_cfg.language_model.max_seq_length = hf_config.max_position_embeddings
        language_config = hf_config.to_dict() | OmegaConf.to_container(model_cfg.language_model)
        if model_cfg.visual_encoder.hf_name is None:
            # Encoder-free path: our own RawImageProcessor (variable resolution,
            # raw patches) + the LM's tokenizer. No AutoImageProcessor involved.
            image_processor = RawImageProcessor(
                patch_size=model_cfg.visual_encoder.patch_size,
                pooling_kernel_size=model_cfg.visual_encoder.pooling_kernel_size,
                max_soft_tokens=model_cfg.visual_encoder.max_soft_tokens,
                image_mean=OmegaConf.to_container(model_cfg.visual_encoder.image_mean)
                if model_cfg.visual_encoder.image_mean is not None
                else None,
                image_std=OmegaConf.to_container(model_cfg.visual_encoder.image_std)
                if model_cfg.visual_encoder.image_std is not None
                else None,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_cfg.language_model.hf_name,
                trust_remote_code=True,
                use_fast=True,
                model_max_length=model_cfg.language_model.max_seq_length,
                padding_side=model_cfg.language_model.padding_side,
            )
            processor = VLMProcessor(image_processor=image_processor, tokenizer=tokenizer)
        else:
            processor = VLMProcessor.from_names(
                model_cfg.visual_encoder.hf_name,
                model_cfg.language_model.hf_name,
                trust_remote_code=True,
                use_fast=True,
                model_max_length=model_cfg.language_model.max_seq_length,
                padding_side=model_cfg.language_model.padding_side,
            )
        attach_audio_feature_extractor(processor, audio_config)
        add_special_tokens(processor.tokenizer, model_cfg.language_model)
        VLMForCausalLM, VLMConfig = get_dynamic_vlm(model_cfg.language_model.hf_name)
        config = VLMConfig(
            vision_config=vision_config,
            connector_config=connector_config,
            audio_config=audio_config,
            lazy_load=True,
            **language_config,
        )
        # Visual-aux structural fields (spec 2026-06-06) must be on the config
        # BEFORE model construction — the causal __init__ builds the head from
        # them. Plain python types; they serialize into checkpoint config.json
        # (conversation_version pattern) so reloads rebuild the head.
        config.visual_aux_objective = str(model_cfg.visual_aux.objective)
        config.visual_aux_head_depth = int(model_cfg.visual_aux.head_depth)
        config.visual_aux_head_hidden = int(model_cfg.visual_aux.head_hidden)
        # Visual-expert structural fields (spec 2026-06-14) must be on the config
        # BEFORE construction — the inner __init__ attaches the per-layer experts
        # from them. Plain types; they serialize into checkpoint config.json so
        # reloads rebuild the structure (visual_aux pattern). `visual_expert` is
        # the master switch; ffn/norm/attention toggle the three EVEv2-style
        # experts independently (ffn defaults True for back-compat with configs
        # that only set `enabled`).
        config.visual_expert = bool(model_cfg.visual_expert.enabled)
        config.visual_expert_ffn = bool(model_cfg.visual_expert.ffn)
        config.visual_expert_norm = bool(model_cfg.visual_expert.norm)
        config.visual_expert_attention = bool(model_cfg.visual_expert.attention)
        config.visual_expert_layers = (
            [int(x) for x in model_cfg.visual_expert.layers]
            if model_cfg.visual_expert.layers is not None
            else None
        )
        config.visual_expert_init_from_text = bool(model_cfg.visual_expert.init_from_text)
        # Visual-prefix structural fields (spec 2026-06-14): on the config BEFORE
        # construction so the inner __init__/init_other_components build the stack
        # from them; serialized into checkpoint config.json so reloads rebuild it.
        config.visual_prefix = bool(model_cfg.visual_prefix.enabled)
        config.visual_prefix_depth = int(model_cfg.visual_prefix.depth)
        config.visual_prefix_heads = int(model_cfg.visual_prefix.heads)
        config.visual_prefix_intermediate = int(model_cfg.visual_prefix.intermediate)
        # Generation pathway structural fields (spec 2026-06-20): on the config
        # BEFORE construction so the causal __init__ builds the x-prediction head
        # + in-context timestep embedder from them; serialized into checkpoint
        # config.json so reloads rebuild them (visual_aux pattern).
        config.generation = bool(model_cfg.generation.enabled)
        config.generation_resolution = int(model_cfg.generation.resolution)
        config.generation_patch_size = int(model_cfg.generation.patch_size)
        config.generation_noise_scale = float(model_cfg.generation.noise_scale)
        config.generation_t_mu = float(model_cfg.generation.t_mu)
        config.generation_t_sigma = float(model_cfg.generation.t_sigma)
        config.generation_sample_steps = int(model_cfg.generation.sample_steps)
        config.generation_cfg_scale = float(model_cfg.generation.cfg_scale)
        config.generation_label_drop = float(model_cfg.generation.label_drop)
        config.generation_independent_embed = bool(model_cfg.generation.independent_embed)
        config.generation_embed_patch_size = int(model_cfg.generation.embed_patch_size)
        config.generation_embed_posemb_size = int(model_cfg.generation.embed_posemb_size)
        config.generation_embed_bottleneck_dim = int(model_cfg.generation.embed_bottleneck_dim)
        config.generation_perceptual_enabled = bool(model_cfg.generation.perceptual_enabled)
        config.generation_perceptual_lpips_weight = float(
            model_cfg.generation.perceptual_lpips_weight
        )
        config.generation_perceptual_dino_weight = float(
            model_cfg.generation.perceptual_dino_weight
        )
        config.generation_perceptual_lpips_net = str(model_cfg.generation.perceptual_lpips_net)
        config.generation_perceptual_dino_model = str(model_cfg.generation.perceptual_dino_model)
        config.generation_perceptual_resize = int(model_cfg.generation.perceptual_resize)
        config.generation_perceptual_t_gate = float(model_cfg.generation.perceptual_t_gate)
        config.generation_perceptual_warmup_steps = int(
            model_cfg.generation.perceptual_warmup_steps
        )
        config.generation_rope_2d = bool(model_cfg.generation.rope_2d)
        # Visual-distill structural fields (spec 2026-06-21) must be on the
        # config BEFORE construction — the causal __init__ builds the projection
        # head from them. Plain types; they serialize into checkpoint config.json
        # (visual_aux pattern) so reloads rebuild the head. The frozen teacher is
        # attached AFTER construction (it is not a checkpoint-persisted module).
        config.visual_distill = bool(model_cfg.visual_distill.enabled)
        config.visual_distill_method = str(model_cfg.visual_distill.method)
        config.visual_distill_teacher_kind = str(model_cfg.visual_distill.teacher_kind)
        config.visual_distill_teacher_name = str(model_cfg.visual_distill.teacher_name)
        config.visual_distill_layers = (
            [int(x) for x in model_cfg.visual_distill.layers]
            if model_cfg.visual_distill.layers is not None
            else None
        )
        config.visual_distill_head_hidden = int(model_cfg.visual_distill.head_hidden)
        config.visual_distill_loss = str(model_cfg.visual_distill.loss)
        config.visual_distill_teacher_out_size = int(model_cfg.visual_distill.teacher_out_size)
        # Anti-collapse dials (see AGENTS.md "Anti-collapse distill port (ST-2)"): flattened onto the
        # config so the head reads them via getattr at build (visual_aux pattern).
        # All default OFF -> bit-identical to the plain per-patch cosine distill.
        config.visual_distill_debias_target = bool(model_cfg.visual_distill.debias_target)
        config.visual_distill_debias_momentum = float(model_cfg.visual_distill.debias_momentum)
        config.visual_distill_debias_std = bool(model_cfg.visual_distill.debias_std)
        config.visual_distill_rkd_dist_weight = float(model_cfg.visual_distill.rkd_dist_weight)
        config.visual_distill_rkd_angle_weight = float(model_cfg.visual_distill.rkd_angle_weight)
        config.visual_distill_rkd_angle_triplets = int(model_cfg.visual_distill.rkd_angle_triplets)
        config.visual_distill_vicreg_var_weight = float(model_cfg.visual_distill.vicreg_var_weight)
        config.visual_distill_vicreg_cov_weight = float(model_cfg.visual_distill.vicreg_cov_weight)
        config.visual_distill_vicreg_gamma = float(model_cfg.visual_distill.vicreg_gamma)
        config.visual_distill_ac_warmup_steps = int(model_cfg.visual_distill.ac_warmup_steps)
        config.visual_distill_mgd_weight = float(model_cfg.visual_distill.mgd_weight)
        config.visual_distill_mgd_beta = float(model_cfg.visual_distill.mgd_beta)
        config.visual_distill_sigreg_weight = float(model_cfg.visual_distill.sigreg_weight)
        config.visual_distill_sigreg_dirs = int(model_cfg.visual_distill.sigreg_dirs)
        config.visual_distill_sigreg_knots = int(model_cfg.visual_distill.sigreg_knots)
        config.visual_distill_sigreg_warmup_steps = int(
            model_cfg.visual_distill.sigreg_warmup_steps
        )
        # Learnable-query structural fields (BREEN port, spec 2026-06-24): on the
        # config BEFORE construction so the causal __init__ builds the query
        # Parameter from them; serialized into checkpoint config.json (visual_aux
        # pattern) so reloads rebuild it with the right shape. enabled=False ->
        # no Parameter (bit-identical baseline).
        config.learnable_query = bool(model_cfg.learnable_query.enabled)
        config.learnable_query_num_fine = int(model_cfg.learnable_query.num_fine)
        config.learnable_query_num_coarse = int(model_cfg.learnable_query.num_coarse)
        config.learnable_query_placement = str(model_cfg.learnable_query.placement)
        # Per-expert sigmoid gate flag (BREEN). On the config before construction
        # so install_visual_experts builds the gate Linears (serialized).
        config.visual_expert_gate = bool(model_cfg.visual_expert.gate)
        # Teacher feature dim must be known BEFORE construction (the head's
        # output width) — read it from the teacher's config (tiny JSON; the full
        # weights load once later in attach_distill_teacher). Serialized so
        # reloads rebuild the head with the right width without the teacher.
        if model_cfg.visual_distill.enabled:
            if str(model_cfg.visual_distill.teacher_kind) in ("clip", "siglip"):
                tcfg = AutoConfig.from_pretrained(model_cfg.visual_distill.teacher_name)
                tcfg = getattr(tcfg, "vision_config", tcfg)
                config.visual_distill_teacher_dim = int(tcfg.hidden_size)
            else:  # vae
                from diffusers import AutoencoderKL  # pyright: ignore[reportMissingImports]

                config.visual_distill_teacher_dim = int(
                    AutoencoderKL.load_config(model_cfg.visual_distill.teacher_name)[
                        "latent_channels"
                    ]
                )
        else:
            config.visual_distill_teacher_dim = 0
        model = VLMForCausalLM.from_pretrained(
            model_cfg.language_model.hf_name,
            config=config,
            trust_remote_code=True,
            dtype=torch.bfloat16
            if trainer_cfg.bf16
            else torch.float16
            if trainer_cfg.fp16
            else torch.float32,
            attn_implementation=trainer_cfg.attn_implementation,
        )
        if model.config.vocab_size < len(processor.tokenizer):
            model.model.resize_token_embeddings(len(processor.tokenizer))
        model.model.init_other_components()
        # Fresh-build only: copy each text FFN's weights into its visual expert
        # AFTER the HF backbone weights are loaded (Mono-InternVL init-from-LLM).
        # Reloads skip this — the checkpoint already carries trained experts.
        if getattr(config, "visual_expert", False) and config.visual_expert_init_from_text:
            from .models.modeling_vlm import init_visual_experts_from_text

            init_visual_experts_from_text(model.model)
        # Fresh-build only: near-identity init for the per-expert sigmoid gates
        # (BREEN). Reloads carry trained gates. No-op when gates are off.
        if getattr(config, "visual_expert", False) and getattr(config, "visual_expert_gate", False):
            from .models.modeling_vlm import init_visual_expert_gates

            init_visual_expert_gates(model.model)
        # Fresh-build only: materialize + randn-init the learnable query Parameter
        # (BREEN). from_pretrained leaves this missing key uninitialized (NaN);
        # reloads carry the trained queries. No-op when learnable_query is off.
        if getattr(config, "learnable_query", False):
            from .models.modeling_vlm import init_learnable_query

            init_learnable_query(model)
        # Fresh-build only: re-init the visual-distill head's anti-collapse EMA
        # buffers (debias_mean/var/inited, _ac_step). from_pretrained leaves these
        # buffers as `to_empty` garbage (missing keys; _init_weights never touches
        # buffers) — a garbage-truthy `debias_inited` makes the EMA subtract an
        # uninitialized (Inf) mean -> NaN cosine -> every microbatch skipped.
        # Reloads carry the trained buffers and skip this. No-op when off.
        if getattr(config, "visual_distill", False):
            from .models.modeling_vlm import init_visual_distill_buffers

            init_visual_distill_buffers(model)
        # Fresh-build only: transplant a pretrained ViT conv patch-embed into the
        # connector's "warm tokenizer" stem (spec 2026-06-22, encoder-free
        # catch-up). Reloads skip this — the checkpoint carries the trained stem.
        init_patch_stem_from_encoder(model, model_cfg)
        # Fresh-build only: zero the generation x-prediction head (DiT zero
        # final layer) AFTER weights load — reloads carry the trained head.
        if getattr(config, "generation", False):
            model.init_generation_modules()
        # E1 causal control (spec 2026-06-18): destroy the pretrained text prior
        # by re-initializing ONLY the LM backbone (embeddings, decoder layers,
        # final norm, untied lm_head) to the config initializer — connector and
        # vision params keep their fresh init. Tests whether, with no cheap prior
        # to ride, the native model is forced to condition on the image.
        if bool(getattr(model_cfg.language_model, "random_init", False)):
            inner = model.model
            lm_mods = [inner.embed_tokens, inner.norm, *list(inner.layers)]

            # transformers 5.x init.normal_/ones_/zeros_ SKIP any tensor flagged
            # _is_hf_initialized=True (which every from_pretrained param carries),
            # so a bare .apply(_init_weights) is a silent no-op. Clear the flag on
            # the targeted LM tensors first so the re-init actually overwrites them.
            def _reinit(mod: Any) -> None:
                for t in (*mod.parameters(recurse=True), *mod.buffers(recurse=True)):
                    t._is_hf_initialized = False
                mod.apply(model._init_weights)

            for mod in lm_mods:
                _reinit(mod)
            if not bool(getattr(model.config, "tie_word_embeddings", False)):
                _reinit(model.lm_head)
            log.warning(
                "random_init=True: LM backbone re-initialized — pretrained text "
                "prior DESTROYED (connector/vision preserved). E1 causal control."
            )
        model.config.lazy_load = False

    # Visual distillation is native (encoder-free / raw-patch) ONLY: the loss
    # reconstructs each image from its raw patches and indexes per-image
    # image_position_ids (compute_distill_loss -> distill_positions[k]). A
    # classic encoder-backed model provides images + image_sizes but NO
    # image_position_ids, so distill_positions is None and the loss crashes the
    # moment an image block appears. Fail loud at load (both build paths) — the
    # head is structurally present whenever visual_distill is enabled.
    if bool(getattr(model.config, "visual_distill", False)) and (
        getattr(model.model, "vision_model", None) is not None
    ):
        raise ValueError(
            "model.visual_distill.enabled requires the encoder-free / raw-patch "
            "path (visual_encoder.hf_name=null), but this checkpoint is "
            "encoder-backed (has a vision_model) — classic encoders don't carry "
            "the per-patch image_position_ids the distill loss indexes. Use a "
            "native (encoder-free) model or disable visual_distill"
        )

    # Frozen distill teacher (spec 2026-06-21): attach on both build paths,
    # AFTER the head exists. Off the module tree -> not in .parameters() below.
    attach_distill_teacher(model)

    log.info(model.config)

    total_params = sum(p.numel() for p in model.parameters())
    from .train.set_trainable import format_param_count

    log.info(f"Model loaded: {model_cfg.name} ({format_param_count(total_params)} parameters)")
    log.info("Model loaded successfully")
    return model, processor


def vlm(cfg: AppConfig) -> None:
    set_seed(cfg.trainer.seed)
    if cfg.is_training:
        log.info("Training mode")
        # Resolve precision: user-provided overrides win; otherwise auto-detect.
        resolved_bf16, resolved_tf32 = resolve_precision(cfg.trainer.bf16, cfg.trainer.tf32)
        cfg.trainer.bf16 = resolved_bf16
        cfg.trainer.tf32 = resolved_tf32
        training_args = get_training_args(cfg.trainer)
        if cfg.dataset.type == "energon":
            # Fail fast (seconds, not minutes): these checks would otherwise
            # only fire in train(), after the slow model load.
            validate_energon_args(training_args)
        model, processor = load_model(cfg.model, cfg.trainer)
        model.to(training_args.device)
        data_args = get_data_args(cfg.dataset, cfg.model)
        # Record the classic-path image policy in the checkpoint config (like
        # conversation_version in train.py) so inference reproduces training
        # preprocessing without guessing. Encoder-free checkpoints ignore it.
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        # Self-describing checkpoints: inference must rebuild prompts with the
        # SAME image layout the model was trained on (plan 2026-06-10).
        model.config.image_position = str(data_args.image_position)
        # BREEN learnable-query placement is the same kind of self-describing,
        # branch-agnostic field as image_position: an S2 SFT can override it
        # (e.g. after_image -> after_text) while loading from an S1 checkpoint
        # whose config.json still says after_image. The fresh-build branch sets
        # it once from model_cfg, but a reload would keep the stale checkpoint
        # value and inference (eval.py reads config.learnable_query_placement)
        # would silently disagree with how S2 actually trained. Refresh it here
        # from the SAME source training uses (data_args.query_placement) so
        # train == saved config == inference on both build paths.
        model.config.learnable_query_placement = str(data_args.query_placement)
        # Cross-modal 4D mask dials (early-fusion access arms, plan 2026-06-10):
        # place the model-config dials onto model.config (applies on both fresh
        # and from_pretrained paths) BEFORE train() validates them against the
        # real layer count and re-copies the normalized result. Flat names mirror
        # visual_aux_objective so they serialize into checkpoint config.json and
        # inference self-describes the mask mode/window.
        model.config.cross_modal_mask_mode = str(cfg.model.cross_modal_mask.mode)
        model.config.cross_modal_mask_window = [int(x) for x in cfg.model.cross_modal_mask.window]
        model.config.cross_modal_mask_bidirectional = bool(cfg.model.cross_modal_mask.bidirectional)
        # Chat-delimiter token ids the conversation preprocessor unmasks into the
        # labels (see _cross_modal_prefix_skip_ids). Only emit them when the
        # cross-modal arm is active. Serialized (flat name) so a resume re-reads
        # the same skip set.
        skip_ids: list[int] | None = None
        if str(cfg.model.cross_modal_mask.mode) != "none":
            skip_ids = _cross_modal_prefix_skip_ids(str(cfg.trainer.version), processor.tokenizer)
        model.config.cross_modal_prefix_skip_ids = skip_ids
        # Image-grounding margin loss dials (spec 2026-06-18). Pure training
        # loss, no module to build -> set here on model.config like the
        # cross_modal dials (branch-agnostic, before train()); flat names so
        # they serialize into checkpoint config.json (visual_aux pattern) and a
        # resume re-reads the same loss. weight=0.0 -> bit-identical baseline.
        model.config.grounding_weight = (
            float(cfg.model.grounding.weight) if bool(cfg.model.grounding.enabled) else 0.0
        )
        model.config.grounding_margin = float(cfg.model.grounding.margin)
        model.config.grounding_corruption = str(cfg.model.grounding.corruption)
        # Visual-distill loss weight (spec 2026-06-21): trainer dial set on
        # model.config like grounding_weight (branch-agnostic, before train());
        # flat name serializes into checkpoint config.json. 0.0 when disabled ->
        # the loss is never built (bit-identical baseline).
        model.config.visual_distill_weight = (
            float(cfg.trainer.visual_distill_weight)
            if bool(cfg.model.visual_distill.enabled)
            else 0.0
        )
        log.info("Creating data module")
        if cfg.dataset.type == "energon":
            # lazy import: needs megatron-energon + multistorageclient, which
            # local-json training environments may not have installed
            from .data.energon_dataset import build_energon_train_loader

            energon_loader = build_energon_train_loader(
                cfg.dataset,
                processor,
                data_args,
                batch_size=cfg.trainer.per_device_train_batch_size,
            )
            data_module = dict(train_dataset=None, eval_dataset=None, data_collator=None)
            train(
                model,
                training_args,
                data_module,
                processor,
                energon_loader=energon_loader,
                energon_num_workers=cfg.dataset.num_workers,
            )
        else:
            data_module = make_supervised_data_module(processor=processor, data_args=data_args)
            train(model, training_args, data_module, processor)


def validate_config(cfg: AppConfig) -> None:
    OmegaConf.to_container(cfg, throw_on_missing=True)
    # Cross-section config invariants that are otherwise silently mis-trained:
    #   #27 batch_token_budget without length_buckets,
    #   #14 generation data/model patch-size mismatch,
    #   #5  plain (2-turn caption) template composed with instruct data.
    # Fails fast here in main() (seconds), before the slow model load.
    validate_dataset_config(cfg.dataset, cfg.model, cfg.trainer)


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="config")  # pyright: ignore
def main(cfg: AppConfig) -> None:
    validate_config(cfg)

    if cfg.trainer.run_name == "small-vlm":
        cfg.trainer.run_name = f"{cfg.model.name}-{cfg.trainer.name}-{cfg.dataset.name}"
        log.info(f"Setting run_name to: {cfg.trainer.run_name}")

    vlm(cfg)


register_configs()


def main_cli():
    i = 0
    while i < len(sys.argv):
        if sys.argv[i].startswith("--local_rank="):
            sys.argv.pop(i)
        else:
            i += 1
    main()


def push_to_hub():
    from .utils import push_vlm_to_hub

    push_vlm_to_hub()


if __name__ == "__main__":
    main_cli()
