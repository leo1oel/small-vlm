import logging
from typing import Any

import pytorch_lightning as pl

from ..config.config_schema import DatasetConfig, InferenceConfig
from ..data.data_module import DataModule
from ..models.model import VLM

log: logging.Logger = logging.getLogger(name=__name__)


def inference(config: InferenceConfig, data_config: DatasetConfig) -> None:  # pyright: ignore
    log.info(f"[bold green]Loading model from checkpoint:[/bold green] {config.checkpoint_path}")
    model = VLM.load_from_checkpoint(config.checkpoint_path)
    trainer: pl.Trainer = pl.Trainer(devices=1)
    data_module: DataModule = DataModule(
        data_config,
        config.num_inference_samples,
        model,
        1,
        config.chat_template,
        do_generation=True,
    )
    results: list[Any] = trainer.predict(  # pyright: ignore
        model=model, dataloaders=data_module.predict_dataloader
    )
    print(results)
