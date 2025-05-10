import logging
import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoConfig

from .config import AppConfig, ModelConfig, TrainerConfig, register_configs
from .data import get_data_args
from .models import VLM, VLMConfig
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
        "llm": {
            "name": cfg.llm.name,
            "path": CONFIG_PATH / "model" / "llm" / f"{cfg.llm.name}.yaml",
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
        f"LLM: [bold blue][link=file://{components['llm']['path']}]{components['llm']['name']}[/link][/bold blue]"
    )
    log.info(
        f"Connector: [bold yellow][link=file://{components['connector']['path']}]{components['connector']['name']}[/link][/bold yellow]"
    )


def load_model(model_cfg: ModelConfig, trainer_cfg: TrainerConfig, device: torch.device) -> VLM:
    if trainer_cfg.from_pretrained:
        log.info(f"Loading model from pretrained: {trainer_cfg.from_pretrained}")
        model: VLM = VLM.from_pretrained(
            trainer_cfg.from_pretrained,
            torch_device=device,
            attn_implementation=trainer_cfg.attn_implementation,
        )
        return model
    else:
        print_model(model_cfg)
        config: VLMConfig = VLMConfig(
            OmegaConf.to_container(model_cfg, resolve=True),
            "float16" if trainer_cfg.fp16 else ("bfloat16" if trainer_cfg.bf16 else "float32"),
            AutoConfig.from_pretrained(model_cfg.llm.hf_name).hidden_size,
        )
        model = VLM(config, device, trainer_cfg.attn_implementation)
        return model


def vlm(cfg: AppConfig) -> None:
    seed_everything(cfg.trainer.seed)
    if cfg.mode.is_training:
        log.info("Training mode")
        training_args = get_training_args(cfg.trainer)
        model: VLM = load_model(cfg.model, cfg.trainer, training_args.device)
        model.set_trainable_params(cfg.trainer.unfreeze)
        data_args = get_data_args(cfg.dataset, cfg.model, model.visual_encoder.preprocessor)
        train(model, training_args, data_args)


def validate_config(cfg: AppConfig) -> None:
    OmegaConf.to_container(cfg, throw_on_missing=True)


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="config")  # pyright: ignore
def main(cfg: AppConfig) -> None:
    validate_config(cfg)
    vlm(cfg)


register_configs()

if __name__ == "__main__":
    main()
