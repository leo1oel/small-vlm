import logging
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from .config import AppConfig, ModelConfig, TrainerConfig, register_configs
from .data import DataModule
from .inference import inference
from .models import VLM
from .train.trainer import train

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


def load_model(
    model_cfg: ModelConfig, trainer_cfg: TrainerConfig, lazy_loading: bool = False
) -> VLM:
    print_model(model_cfg)
    model: VLM = VLM(model_cfg, trainer_cfg, lazy_loading)
    return model


def vlm(cfg: AppConfig) -> None:
    if cfg.mode.is_training:
        log.info("Training mode")
        # only load necessary components for dataset processing
        model: VLM = load_model(cfg.model, cfg.trainer, lazy_loading=True)
        data_module = DataModule(
            cfg.dataset,
            cfg.trainer.num_training_samples,
            model,
            cfg.trainer.batch_size,
            cfg.trainer.chat_template,
        )
        # Get dataloaders
        train_dataloader = data_module.train_dataloader
        val_dataloader = data_module.val_dataloader
        test_dataloader = data_module.test_dataloader

        # Check if training data is available
        if train_dataloader is None:
            log.error("Training data load failed")
            raise ValueError("Training data load failed")

        log.info(f"Training data loaded successfully: {len(train_dataloader)} batches")
        if cfg.trainer.num_training_samples is None:
            cfg.trainer.num_training_samples = data_module.num_samples["train"]

        # Log validation and test data status
        if val_dataloader:
            log.info(f"Validation data loaded successfully: {len(val_dataloader)} batches")
        else:
            log.warning("Validation data load failed")

        if test_dataloader:
            log.info(f"Test data loaded successfully: {len(test_dataloader)} batches")
        else:
            log.warning("Test data load failed")

        # initialize all components
        model.initialize_components()
        train(cfg.trainer, model, train_dataloader, val_dataloader, test_dataloader)
    else:
        log.info("Inference mode")
        inference(cfg.inference, cfg.dataset)


def validate_config(cfg: AppConfig) -> None:
    OmegaConf.to_container(cfg, throw_on_missing=True)


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="config")  # pyright: ignore
def main(cfg: AppConfig) -> None:
    validate_config(cfg)
    vlm(cfg)


register_configs()

if __name__ == "__main__":
    main()
