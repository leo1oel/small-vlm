import logging
import sys
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer, set_seed

from vlm.config.config_schema import LanguageModelConfig

from .config import AppConfig, ModelConfig, TrainerConfig, register_configs
from .data import get_data_args, make_supervised_data_module
from .models import VLMProcessor, get_dynamic_vlm
from .models.image_processing_raw import RawImageProcessor
from .train import get_training_args, train, validate_energon_args
from .utils.precision import resolve_precision

log: logging.Logger = logging.getLogger(name=__name__)
CONFIG_PATH: Path = Path(__file__).resolve().parent / "config"


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
        model.config.lazy_load = False

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
