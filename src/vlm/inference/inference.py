import logging
from pathlib import Path
from typing import Any

import pytorch_lightning as pl

from ..config.config_schema import DatasetConfig, InferenceConfig
from ..data.data_module import DataModule
from ..models.model import VLM

log: logging.Logger = logging.getLogger(name=__name__)


def inference(config: InferenceConfig, data_config: DatasetConfig) -> None:  # pyright: ignore
    log.info(f"[bold green]Loading model from checkpoint:[/bold green] {config.checkpoint_path}")
    model: VLM = VLM.load_from_checkpoint(Path(config.checkpoint_path), map_location="cuda")
    trainer: pl.Trainer = pl.Trainer()
    data_module: DataModule = DataModule(
        data_config,
        config.num_inference_samples,
        model,
        1,
        config.chat_template,
    )
    results: list[Any] = trainer.predict(  # pyright: ignore
        model=model, dataloaders=data_module.train_dataloader
    )
    for result in results:
        print(result)
