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
        return load_dataset(cfg.hf_name, split=split, trust_remote_code=True).map(
            model.transform,
            num_proc=int(cfg.num_proc) if cfg.num_proc and cfg.num_proc.isdigit() else None,  # pyright: ignore
        )  # pyright: ignore
    else:
        raise ValueError(f"Dataset type {dataset_type} not supported")
