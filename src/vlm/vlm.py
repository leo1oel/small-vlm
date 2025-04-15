import logging
from pathlib import Path

import hydra
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf

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
        # Load model components needed for DataModule first
        # Lazy loading might still be useful if model initialization is heavy
        model: VLM = load_model(cfg.model, cfg.trainer, lazy_loading=True)

        # Initialize DataModule
        data_module = DataModule(
            cfg.dataset,
            cfg.trainer.num_training_samples,  # Pass initial value (can be None)
            model,
            cfg.trainer.batch_size,
            cfg.trainer.chat_template,
            # Make sure processed_data_dir is configured if needed, or uses default
            # processed_data_dir=cfg.data.processed_dir # Example if you add it to config
        )

        data_module.prepare_data()
        data_module.setup(stage="fit")

        # Check if training data was loaded successfully during setup
        if "train" not in data_module.num_samples or data_module.num_samples["train"] == 0:
            log.error("Training data failed to load or is empty after setup.")
            raise ValueError("Training data load failed or is empty.")

        # Update num_training_samples if it was initially None
        if cfg.trainer.num_training_samples is None:
            if "train" in data_module.num_samples:
                cfg.trainer.num_training_samples = data_module.num_samples["train"]
                log.info(
                    f"Updated cfg.trainer.num_training_samples to {cfg.trainer.num_training_samples}"
                )
            else:
                # This case should ideally be caught by the check above, but added for safety
                log.error("Cannot update num_training_samples because train split was not loaded.")
                raise ValueError("Failed to determine number of training samples.")

        # Log validation data status (optional, as Trainer will handle loading)
        if "val" in data_module.num_samples:
            log.info(
                f"Validation data loaded successfully during setup: {data_module.num_samples['val']} samples"
            )
        else:
            log.warning("Validation data not found or failed to load during setup.")

        # initialize all components
        model.initialize_components()
        train(cfg.trainer, model, data_module)
    else:
        log.info("Inference mode")
        model = load_model(cfg.model, cfg.trainer, lazy_loading=False)
        inference(cfg.inference, model, cfg.dataset)


def validate_config(cfg: AppConfig) -> None:
    OmegaConf.to_container(cfg, throw_on_missing=True)


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="config")  # pyright: ignore
def main(cfg: AppConfig) -> None:
    validate_config(cfg)
    vlm(cfg)


register_configs()

if __name__ == "__main__":
    main()
