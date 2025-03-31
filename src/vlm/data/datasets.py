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
        dataset = load_dataset(cfg.hf_name, split=split, trust_remote_code=True)  # pyright: ignore
        dataset = dataset.select(range(10))  # pyright: ignore
        dataset: Dataset[dict[str, torch.Tensor]] = dataset.map(  # pyright: ignore
            model.transform,
            num_proc=int(cfg.num_proc) if cfg.num_proc else None,  # pyright: ignore
        )  # pyright: ignore
        return dataset  # pyright: ignore
    else:
        raise ValueError(f"Dataset type {dataset_type} not supported")
