import logging
from pathlib import Path

import hydra
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf

from .config import AppConfig, ModelConfig, TrainerConfig, register_configs
from .data.data_arguments import get_data_args
from .models import VLM
from .train.train import train
from .train.training_arguments import get_training_args

log: logging.Logger = logging.getLogger(name=__name__)
CONFIG_PATH: Path = Path(__file__).resolve().parent / "config"
seed_everything(42, workers=True)


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


def load_model(model_cfg: ModelConfig, trainer_cfg: TrainerConfig) -> VLM:
    print_model(model_cfg)
    model: VLM = VLM(model_cfg, trainer_cfg)
    return model


def vlm(cfg: AppConfig) -> None:
    if cfg.mode.is_training:
        log.info("Training mode")
        training_args = get_training_args(cfg.trainer)
        log.info(f"Training arguments: {training_args}")
        data_args = get_data_args(cfg.data)
        log.info(f"Data arguments: {data_args}")
        model: VLM = load_model(cfg.model, cfg.trainer)
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
