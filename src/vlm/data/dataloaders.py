import os
from collections.abc import Callable
from logging import getLogger

import torch
from datasets import load_dataset  # pyright: ignore
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from ..config.config_schema import DatasetConfig, InferenceConfig
from ..models.model import VLM
from .datasets import get_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

log = getLogger(__name__)


def get_collate_fn(
    tokenizer: AutoTokenizer,
) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
    def collate_fn(
        batch: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        max_text_length: int = max(len(item["text"]) for item in batch)
        max_label_length: int = max(len(item["label"]) for item in batch)
        image: list[torch.Tensor] = []
        pad_input_ids: list[torch.Tensor] = []
        pad_labels: list[torch.Tensor] = []

        for item in batch:
            text_pad_length: int = max_text_length - len(item["text"])
            padded_input_ids: torch.Tensor = torch.cat(
                [
                    torch.tensor(item["text"]),
                    torch.full(
                        (text_pad_length,),
                        tokenizer.pad_token_id,  # pyright: ignore
                    ),
                ]
            )
            label_pad_length: int = max_label_length - len(item["label"])
            padded_labels: torch.Tensor = torch.cat(
                [
                    torch.tensor(item["label"]),
                    torch.full((label_pad_length,), -100),
                ]
            )

            pad_input_ids.append(padded_input_ids)  # pyright: ignore
            pad_labels.append(padded_labels)  # pyright: ignore
            image.append(torch.tensor(item["image"]))

        input_ids: torch.Tensor = torch.stack(pad_input_ids)  # pyright: ignore
        labels: torch.Tensor = torch.stack(pad_labels)  # pyright: ignore
        images: torch.Tensor = torch.stack(image)  # pyright: ignore
        return {
            "images": images,
            "texts": input_ids,
            "labels": labels,
        }

    return collate_fn


def get_train_dataloader(
    cfg: DatasetConfig, model: VLM
) -> DataLoader[dict[str, torch.Tensor]] | None:  # pyright: ignore
    try:
        dataset: Dataset[dict[str, torch.Tensor]] = get_dataset(cfg, model, "train")
        collate_fn: Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]] = (
            get_collate_fn(model.language_model.tokenizer)
        )  # pyright: ignore
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
    except ValueError:
        return None


def get_val_dataloader(
    cfg: DatasetConfig, model: VLM
) -> DataLoader[dict[str, torch.Tensor]] | None:  # pyright: ignore
    try:
        dataset: Dataset[dict[str, torch.Tensor]] = get_dataset(cfg, model, "val")
        collate_fn: Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]] = (
            get_collate_fn(model.language_model.tokenizer)
        )  # pyright: ignore
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
    except ValueError:
        return None


def get_test_dataloader(
    cfg: DatasetConfig, model: VLM
) -> DataLoader[dict[str, torch.Tensor]] | None:  # pyright: ignore
    try:
        dataset: Dataset[dict[str, torch.Tensor]] = get_dataset(cfg, model, "test")
        collate_fn: Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]] = (
            get_collate_fn(model.language_model.tokenizer)
        )  # pyright: ignore
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
    except ValueError:
        return None


def get_inference_dataloader(cfg: InferenceConfig) -> DataLoader[dict[str, torch.Tensor]] | None:  # pyright: ignore
    try:
        dataset: Dataset[dict[str, torch.Tensor]] = load_dataset(
            cfg.hf_name,  # pyright: ignore
            split=cfg.split,
            trust_remote_code=True,
        )
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
    except (ValueError, TypeError):
        return None
