import logging
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

from .config import AppConfig, ModelConfig, TrainerConfig, register_configs
from .data import (
    get_inference_dataloader,
    get_test_dataloader,
    get_train_dataloader,
    get_val_dataloader,
)
from .inference import inference
from .models import VLM
from .train.trainer import train

log: logging.Logger = logging.getLogger(name=__name__)
config_path: Path = Path(__file__).resolve().parent / "config"
seed_everything(42, workers=True)


def print_model(cfg: ModelConfig) -> None:
    model_name: str = cfg.name
    model_config_path: Path = (config_path / "model" / f"{model_name}.yaml").resolve()
    model_url: str = f"file://{model_config_path}"

    visual_encoder_name: str = cfg.visual_encoder.name
    visual_encoder_path: Path = (
        config_path / "model" / "visual_encoder" / f"{visual_encoder_name}.yaml"
    ).resolve()
    visual_url: str = f"file://{visual_encoder_path}"

    llm_name: str = cfg.llm.name
    llm_path: Path = (config_path / "model" / "llm" / f"{llm_name}.yaml").resolve()
    llm_url: str = f"file://{llm_path}"

    connector_name: str = cfg.connector.name
    connector_path: Path = (
        config_path / "model" / "connector" / f"{connector_name}.yaml"
    ).resolve()
    connector_url: str = f"file://{connector_path}"

    log.info(f"Loading model: [bold red][link={model_url}]{model_name}[/link][/bold red]")
    log.info(
        f"Visual encoder: [bold cyan][link={visual_url}]{visual_encoder_name}[/link][/bold cyan]"
    )
    log.info(f"LLM: [bold blue][link={llm_url}]{llm_name}[/link][/bold blue]")
    log.info(f"Connector: [bold yellow][link={connector_url}]{connector_name}[/link][/bold yellow]")


def load_model(model_cfg: ModelConfig, trainer_cfg: TrainerConfig) -> VLM:
    print_model(model_cfg)
    model: VLM = VLM(model_cfg, trainer_cfg)
    return model


def vlm(cfg: AppConfig) -> None:
    if cfg.mode.is_training:
        model: VLM = load_model(cfg.model, cfg.trainer)
        log.info("[bold red]Training mode[/bold red]")
        train_dataloader: DataLoader[dict[str, torch.Tensor]] | None = get_train_dataloader(
            cfg.dataset, model
        )
        if train_dataloader is not None:
            log.info(
                f"[bold green]Training data load successfully:[/bold green] {len(train_dataloader)}"
            )
        else:
            log.error("[bold red]Training data load failed[/bold red]")
            raise ValueError("Training data load failed")

        val_dataloader: DataLoader[dict[str, torch.Tensor]] | None = get_val_dataloader(
            cfg.dataset, model
        )
        if val_dataloader is not None:
            log.info(
                f"[bold green]Validation data load successfully:[/bold green] {len(val_dataloader)}"
            )
        else:
            log.warning("[bold yellow]Validation data load failed[/bold yellow]")

        test_dataloader: DataLoader[dict[str, torch.Tensor]] | None = get_test_dataloader(
            cfg.dataset, model
        )
        if test_dataloader is not None:
            log.info(
                f"[bold green]Test data load successfully:[/bold green] {len(test_dataloader)}"
            )
        else:
            log.warning("[bold yellow]Test data load failed[/bold yellow]")

        train(cfg.trainer, model, train_dataloader, val_dataloader, test_dataloader)
    else:
        log.info("[bold red]Inference mode[/bold red]")
        inference_dataloader: DataLoader[dict[str, torch.Tensor]] | None = get_inference_dataloader(
            cfg.inference
        )
        inference(cfg.inference, inference_dataloader)


def validate_config(cfg: AppConfig) -> None:
    OmegaConf.to_container(cfg, throw_on_missing=True)


@hydra.main(version_base=None, config_path=str(config_path), config_name="config")  # pyright: ignore
def main(cfg: AppConfig) -> None:
    validate_config(cfg)
    vlm(cfg)


register_configs()

if __name__ == "__main__":
    main()
