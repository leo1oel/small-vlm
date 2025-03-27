import logging
from pathlib import Path

import hydra
import torch
from lightning.pytorch.utilities.model_summary.model_summary import ModelSummary
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
    log.info("[bold green]Model summary for an example input:[/bold green]")
    log.info(ModelSummary(model))  # pyright: ignore
    return model


def vlm(cfg: AppConfig) -> None:
    model: VLM = load_model(cfg.model, cfg.trainer)
    if cfg.mode.is_training:
        train_dataloader: DataLoader[dict[str, torch.Tensor]] = get_train_dataloader(
            cfg.dataset, model
        )
        log.info(
            f"[bold green]Training data load successfully:[/bold green] {len(train_dataloader)}"
        )
        val_dataloader: DataLoader[dict[str, torch.Tensor]] = get_val_dataloader(cfg.dataset, model)
        log.info(
            f"[bold green]Validation data load successfully:[/bold green] {len(val_dataloader)}"
        )
        test_dataloader: DataLoader[dict[str, torch.Tensor]] = get_test_dataloader(
            cfg.dataset, model
        )
        log.info(f"[bold green]Test data load successfully:[/bold green] {len(test_dataloader)}")
        train(cfg.trainer, model, train_dataloader, val_dataloader, test_dataloader)
    else:
        inference_dataloader: DataLoader[dict[str, torch.Tensor]] = get_inference_dataloader(
            cfg.dataset, model
        )
        log.info(
            f"[bold green]Inference data load successfully:[/bold green] {len(inference_dataloader)}"
        )
        inference(cfg.trainer, inference_dataloader)


@hydra.main(version_base=None, config_path=str(config_path), config_name="config")  # pyright: ignore
def main(cfg: AppConfig) -> None:
    vlm(cfg)


register_configs()

if __name__ == "__main__":
    main()
