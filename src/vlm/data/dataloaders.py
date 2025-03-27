from collections.abc import Callable

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from ..config.config_schema import DatasetConfig
from ..models.model import VLM
from .datasets import get_dataset


def get_collate_fn(
    tokenizer: AutoTokenizer,
) -> Callable[[list[tuple[torch.Tensor, torch.Tensor]]], tuple[torch.Tensor, torch.Tensor]]:  # pyright: ignore
    def collate_fn(
        batch: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:  # pyright: ignore
        max_length: int = max(len(item[0]) for item in batch)  # pyright: ignore

        pad_input_ids: list[int] = []
        pad_labels: list[int] = []

        for item in batch:  # pyright: ignore
            pad_length: int = max_length - len(item[0])  # pyright: ignore
            padded_input_ids = torch.cat(
                [
                    item[0],
                    torch.full(
                        (pad_length,),
                        tokenizer.pad_token_id,  # pyright: ignore
                    ),
                ]
            )
            padded_labels = torch.cat(
                [
                    item[1],
                    torch.full((pad_length,), -100),
                ]
            )

            pad_input_ids.append(padded_input_ids)  # pyright: ignore
            pad_labels.append(padded_labels)  # pyright: ignore

        input_ids: torch.Tensor = torch.stack(pad_input_ids)  # pyright: ignore
        labels: torch.Tensor = torch.stack(pad_labels)  # pyright: ignore

        return input_ids, labels

    return collate_fn


def get_train_dataloader(cfg: DatasetConfig, model: VLM) -> DataLoader[dict[str, torch.Tensor]]:  # pyright: ignore
    dataset: Dataset[dict[str, torch.Tensor]] = get_dataset(cfg, model, "train")
    collate_fn: Callable[
        [list[tuple[torch.Tensor, torch.Tensor]]], tuple[torch.Tensor, torch.Tensor]
    ] = get_collate_fn(model.language_model.tokenizer)  # pyright: ignore
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)


def get_val_dataloader(cfg: DatasetConfig, model: VLM) -> DataLoader[dict[str, torch.Tensor]]:  # pyright: ignore
    dataset: Dataset[dict[str, torch.Tensor]] = get_dataset(cfg, model, "val")
    collate_fn: Callable[
        [list[tuple[torch.Tensor, torch.Tensor]]], tuple[torch.Tensor, torch.Tensor]
    ] = get_collate_fn(model.language_model.tokenizer)  # pyright: ignore
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)


def get_inference_dataloader(cfg: DatasetConfig, model: VLM) -> DataLoader[dict[str, torch.Tensor]]:  # pyright: ignore
    dataset: Dataset[dict[str, torch.Tensor]] = get_dataset(cfg, model, "test")
    collate_fn: Callable[
        [list[tuple[torch.Tensor, torch.Tensor]]], tuple[torch.Tensor, torch.Tensor]
    ] = get_collate_fn(model.language_model.tokenizer)  # pyright: ignore
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
