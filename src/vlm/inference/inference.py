import logging
from typing import Any

import lightning as L
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint  # pyright: ignore

from ..config.config_schema import DatasetConfig, InferenceConfig
from ..data.data_module import DataModule
from ..models.model import VLM

log: logging.Logger = logging.getLogger(name=__name__)


def inference(config: InferenceConfig, model: VLM, data_config: DatasetConfig) -> None:  # pyright: ignore
    log.info(f"[bold green]Loading model from checkpoint:[/bold green] {config.checkpoint_path}")
    model = load_state_dict_from_zero_checkpoint(model, config.checkpoint_path)
    trainer: L.Trainer = L.Trainer()
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
