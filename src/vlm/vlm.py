import logging
import os
import random
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizer

from vlm.config.config_schema import LanguageModelConfig

from .config import AppConfig, ModelConfig, TrainerConfig, register_configs
from .data import get_data_args, make_supervised_data_module
from .models import VLMProcessor, get_dynamic_vlm
from .train import get_training_args, train

log: logging.Logger = logging.getLogger(name=__name__)
CONFIG_PATH: Path = Path(__file__).resolve().parent / "config"


def seed_everything(seed: int) -> None:
    log.info(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_model(cfg: ModelConfig) -> None:
    components = {
        "model": {"name": cfg.name, "path": CONFIG_PATH / "model" / f"{cfg.name}.yaml"},
        "visual_encoder": {
            "name": cfg.visual_encoder.name,
            "path": CONFIG_PATH / "model" / "visual_encoder" / f"{cfg.visual_encoder.name}.yaml",
        },
        "language_model": {
            "name": cfg.language_model.name,
            "path": CONFIG_PATH / "model" / "language_model" / f"{cfg.language_model.name}.yaml",
        },
        "connector": {
            "name": cfg.connector.name,
            "path": CONFIG_PATH / "model" / "connector" / f"{cfg.connector.name}.yaml",
        },
    }

    log.info(
        f"Loading model: [bold red][link=file://{components['model']['path']}]{components['model']['name']}[/link][/bold red]"
    )
    log.info(
        f"Visual encoder: [bold cyan][link=file://{components['visual_encoder']['path']}]{components['visual_encoder']['name']}[/link][/bold cyan]"
    )
    log.info(
        f"Language model: [bold blue][link=file://{components['language_model']['path']}]{components['language_model']['name']}[/link][/bold blue]"
    )
    log.info(
        f"Connector: [bold yellow][link=file://{components['connector']['path']}]{components['connector']['name']}[/link][/bold yellow]"
    )


def add_special_tokens(tokenizer: PreTrainedTokenizer, config: LanguageModelConfig) -> None:
    """Adds special tokens to the tokenizer if they don't exist."""
    # Create a mapping of tokens to their attribute names

    token_config: dict = {
        "image_token": getattr(config, "image_token", "<image>"),
        "image_patch_token": getattr(config, "image_patch_token", "<im_patch>"),
        "image_start_token": getattr(config, "image_start_token", "<im_start>"),
        "image_end_token": getattr(config, "image_end_token", "<im_end>"),
        "system_token": getattr(config, "system_token", None),
        "user_token": getattr(config, "user_token", None),
        "assistant_token": getattr(config, "assistant_token", "<|assistant|>"),
    }
    token_mapping = {
        token_config["system_token"]: "system_token_id",
        token_config["user_token"]: "user_token_id",
        token_config["assistant_token"]: "assistant_token_id",
    }
    if config.use_image_patch_token:
        token_mapping[token_config["image_patch_token"]] = "image_patch_token_id"
    if config.use_start_end_tokens:
        token_mapping[token_config["image_start_token"]] = "image_start_token_id"
        token_mapping[token_config["image_end_token"]] = "image_end_token_id"

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
    if tokens_to_add:
        log.info(f"Adding tokens: {tokens_to_add}")
        tokenizer.add_tokens(tokens_to_add, special_tokens=True)

        # Now set the IDs for newly added tokens
        for token in tokens_to_add:
            token_id = tokenizer.convert_tokens_to_ids(token)


def load_model(model_cfg: ModelConfig, trainer_cfg: TrainerConfig):
    log.info("Loading model")

    # Load model
    if trainer_cfg.from_pretrained:
        log.info("Loading processor from pretrained: {trainer_cfg.from_pretrained}")
        processor = VLMProcessor.from_pretrained(
            trainer_cfg.from_pretrained,
        )
        log.info(f"Loading model from pretrained: {trainer_cfg.from_pretrained}")
        VLMForCasualLM, VLMConfig = get_dynamic_vlm(model_cfg.language_model.hf_name)
        model: VLMForCasualLM = VLMForCasualLM.from_pretrained(
            trainer_cfg.from_pretrained,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16
            if trainer_cfg.bf16
            else torch.float16
            if trainer_cfg.fp16
            else torch.float32,
            attn_implementation=trainer_cfg.attn_implementation,
        )
    else:
        vision_config = OmegaConf.to_container(model_cfg.visual_encoder)
        connector_config = OmegaConf.to_container(model_cfg.connector)
        processor = VLMProcessor.from_names(
            model_cfg.visual_encoder.hf_name,
            model_cfg.language_model.hf_name,
            trust_remote_code=True,
            use_fast=True,
            model_max_length=model_cfg.language_model.max_seq_length,
            padding_side=model_cfg.language_model.padding_side,
        )
        add_special_tokens(processor.tokenizer, model_cfg.language_model)
        VLMForCauslLM, VLMConfig = get_dynamic_vlm(model_cfg.language_model.hf_name)
        config = VLMConfig(
            vision_config=vision_config,
            connector_config=connector_config,
            lazy_load=True,
            **OmegaConf.to_container(model_cfg.language_model),
        )
        model = VLMForCauslLM.from_pretrained(
            model_cfg.language_model.hf_name,
            config=config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
            if trainer_cfg.bf16
            else torch.float16
            if trainer_cfg.fp16
            else torch.float32,
            attn_implementation=trainer_cfg.attn_implementation,
        )
        model.model.resize_token_embeddings(len(processor.tokenizer))
        model.model.init_other_components()
        model.config.lazy_load = False

    log.info(model.config)
    log.info("Model loaded successfully")
    return model, processor


def vlm(cfg: AppConfig) -> None:
    seed_everything(cfg.trainer.seed)
    if cfg.mode.is_training:
        log.info("Training mode")
        training_args = get_training_args(cfg.trainer)
        model, processor = load_model(cfg.model, cfg.trainer)
        model.to(training_args.device)
        data_args = get_data_args(cfg.dataset, cfg.model)
        log.info("Creating data module")
        data_module = make_supervised_data_module(processor=processor, data_args=data_args)
        train(model, training_args, data_module, processor)


def validate_config(cfg: AppConfig) -> None:
    OmegaConf.to_container(cfg, throw_on_missing=True)


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="config")  # pyright: ignore
def main(cfg: AppConfig) -> None:
    validate_config(cfg)
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


if __name__ == "__main__":
    main_cli()
