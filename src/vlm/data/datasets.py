from logging import getLogger

import torch
from datasets import load_dataset  # pyright: ignore
from torch.utils.data import Dataset

from ..config.config_schema import DatasetConfig
from ..models.model import VLM

log = getLogger(__name__)


def get_dataset(cfg: DatasetConfig, model: VLM, split: str) -> Dataset[dict[str, torch.Tensor]]:
    dataset_type: str = cfg.type
    if dataset_type == "huggingface":
        log.info(f"[bold green]Start loading huggingface dataset:[/bold green] {cfg.name}")
        return load_dataset(cfg.name, split=split, trust_remote_code=True).map(
            model.transform, batched=True
        )  # pyright: ignore
    else:
        raise ValueError(f"Dataset type {dataset_type} not supported")
