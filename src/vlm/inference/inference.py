import logging
from typing import Any, override

import pytorch_lightning as pl

from ..config.config_schema import DatasetConfig, InferenceConfig
from ..data.data_module import DataModule
from ..models.model import VLM

log: logging.Logger = logging.getLogger(name=__name__)


class PredictDataModule(pl.LightningDataModule):
    def __init__(self, data_module: DataModule):
        super().__init__()
        self.data_module: DataModule = data_module

    @override
    def predict_dataloader(self):
        return self.data_module.predict_dataloader


def inference(config: InferenceConfig, model: VLM, data_config: DatasetConfig) -> None:  # pyright: ignore
    log.info(f"[bold green]Loading model from checkpoint:[/bold green] {config.checkpoint_path}")
    trainer: pl.Trainer = pl.Trainer()
    data_module: DataModule = DataModule(
        data_config,
        config.num_inference_samples,
        model,
        1,
        config.chat_template,
    )
    model.initialize_components()

    # predict_data = PredictDataModule(data_module)
    results: list[Any] = trainer.predict(  # pyright: ignore
        model=model, dataloaders=data_module.predict_dataloader, ckpt_path=config.checkpoint_path
    )
    print(results)
