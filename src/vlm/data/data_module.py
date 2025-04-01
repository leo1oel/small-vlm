import json
import logging
import os
from collections.abc import Callable
from typing import Any, Literal, cast

import torch
from datasets import Dataset, DatasetDict, load_dataset  # pyright: ignore
from PIL import Image
from torch.utils.data import DataLoader
from transformers import BaseImageProcessor, PreTrainedTokenizer

from ..config.config_schema import DatasetConfig
from ..models.model import VLM
from ..utils.chat_template import get_chat_template

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
log = logging.getLogger(__name__)


class DataModule:
    def __init__(
        self,
        dataset_config: DatasetConfig,
        model: VLM,
        batch_size: int,
        chat_template: str,
    ):
        self.dataset_config: DatasetConfig = dataset_config
        self.model: VLM = model
        self.batch_size: int = batch_size
        self.chat_template: str = get_chat_template(chat_template)
        self._raw_dataset: DatasetDict | None = None
        self._processed_datasets: dict[str, Dataset] = {}

        # model components
        self.tokenizer: PreTrainedTokenizer = self.model.language_model.tokenizer
        self.image_preprocessor: BaseImageProcessor = self.model.visual_encoder.preprocessor
        self.image_token_id: int = cast(int, self.model.language_model.token_config.image_token_id)
        self.image_token_size: int = self.model.visual_encoder.token_size

        self.transform: Callable[
            [dict[str, Image.Image | str], bool], dict[str, torch.Tensor | list[torch.Tensor]]
        ] = self._build_transform()

        self._load_raw_dataset()

        self.num_samples: dict[str, int] = {}

    def _load_raw_dataset(self):
        try:
            dataset_type = self.dataset_config.type
            if dataset_type == "huggingface":
                log.info(f"Loading HuggingFace dataset: {self.dataset_config.name}")
                self._raw_dataset = cast(
                    DatasetDict, load_dataset(self.dataset_config.hf_name, trust_remote_code=True)
                )
            else:
                log.warning(f"Dataset type {dataset_type} not supported")
                raise ValueError(f"Dataset type {dataset_type} not supported")
        except Exception as e:
            log.warning(f"Failed to load raw dataset: {str(e)}")
            self._raw_dataset = None

    def get_dataset(self, split: Literal["train", "val", "test"]) -> Dataset | None:
        if split in self._processed_datasets:
            log.info(f"Using cached processed dataset for split: {split}")
            return self._processed_datasets[split]

        if self._raw_dataset is None:
            log.error("Raw dataset is not loaded")
            return None

        if split not in self._raw_dataset:
            log.error(f"Split {split} not found in dataset")
            return None

        try:
            log.info(f"Processing dataset for split: {split}")
            num_proc = getattr(self.dataset_config, "num_proc", None)
            log.info(f"Using {num_proc} processes for mapping")

            processed_split = self._raw_dataset[split].map(
                self.transform,
                num_proc=num_proc,
                load_from_cache_file=True,
            )

            self._processed_datasets[split] = processed_split
            self.num_samples[split] = len(processed_split)
            return processed_split

        except Exception as e:
            log.error(f"Failed to process dataset for split {split}: {str(e)}")
            return None

    def _build_transform(
        self,
    ) -> Callable[
        [dict[str, Image.Image | str], bool], dict[str, torch.Tensor | list[torch.Tensor]]
    ]:
        def transform(
            item: dict[str, Image.Image | str], do_generation: bool = False
        ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
            image_tensor = self._process_image(item)

            text_str = self._extract_text(item)

            text = json.loads(text_str.replace("\n", "\\n"))
            text_and_label = self._text_transform(text, do_generation)

            return {
                "image": cast(torch.Tensor, image_tensor),
                "text": text_and_label[0],
                "label": text_and_label[1],
            }

        return transform

    def _process_image(self, item: dict[str, Any]) -> torch.Tensor | None:
        if "image" not in item:
            error_msg = f"Cannot find image in item {item}"
            log.error(error_msg)
            raise ValueError(error_msg)

        image = item["image"]
        if isinstance(image, torch.Tensor):
            return image
        elif isinstance(image, Image.Image):
            original_image = image.convert("RGB")
            input_image = self.image_preprocessor(original_image, return_tensors="pt")
            return input_image["pixel_values"].squeeze(0)

    def _extract_text(self, item: dict[str, Any]) -> str:
        if "text" in item:
            return item["text"]
        elif "conversations" in item:
            return item["conversations"]
        else:
            error_msg = f"Cannot find text in item {item}"
            log.error(error_msg)
            raise ValueError(error_msg)

    def _text_transform(
        self, text: list[dict[str, str]], do_generation: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conversation = self._prepare_conversation(text)

        input_ids = self._apply_chat_template(conversation, do_generation)

        labels = self._prepare_labels(input_ids, conversation)

        expanded_labels = self._handle_image_tokens(input_ids, labels)
        return (input_ids, torch.tensor(expanded_labels))

    def _prepare_conversation(self, text: list[dict[str, str]]) -> list[dict[str, str]]:
        return [
            {"role": "user" if item["from"] == "human" else "assistant", "content": item["value"]}
            for item in text
        ]

    def _apply_chat_template(
        self, conversation: list[dict[str, str]], do_generation: bool
    ) -> torch.Tensor:
        self.tokenizer.chat_template = self.chat_template
        input_ids = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=do_generation,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )[0]

        return cast(torch.Tensor, input_ids)

    # def preparelabels(self, input_ids: torch.Tensor) -> torch.Tensor:
    #     labels = torch.full_like(input_ids, -100)
    #     labels[:-1] = input_ids[1:].clone()
    #     assistant_ranges = self._find_assistant_ranges(input_ids)
    #     for i in range(len(labels)):
    #         is_in_assistant_range = any(start <= i <= end for start, end in assistant_ranges)
    #         if not is_in_assistant_range:
    #             labels[i] = -100
    #     return labels

    def _prepare_labels(
        self, input_ids: torch.Tensor, conversation: list[dict[str, str]]
    ) -> torch.Tensor:
        labels = torch.full_like(input_ids, -100)
        labels[:-1] = input_ids[1:].clone()

        assistant_msgs = [msg for msg in conversation if msg["role"] == "assistant"]
        if not assistant_msgs:
            return torch.full_like(input_ids, -100)

        full_text: str = cast(
            str,
            self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            ),
        )

        for msg in assistant_msgs:
            assistant_text = msg["content"]
            start_char = full_text.find(assistant_text)
            if start_char != -1:
                prefix = full_text[:start_char]
                start_token = len(self.tokenizer.encode(prefix, add_special_tokens=False))
                end_token = (
                    start_token
                    + len(self.tokenizer.encode(assistant_text, add_special_tokens=False))
                    - 1
                )

                for _ in range(start_token, min(end_token + 1, len(labels))):
                    pass

                for i in range(len(labels)):
                    if not (start_token <= i <= end_token):
                        labels[i] = -100

        return labels

    # def _find_assistant_ranges(self, input_ids: torch.Tensor) -> list[tuple[int, int]]:
    #     assistant_ranges: list[tuple[int, int]] = []
    #     in_assistant = False
    #     start_idx = None

    #     for i, token_id in enumerate(input_ids):
    #         token = self.tokenizer.decode([token_id])

    #         if "<|assistant|>" in token:
    #             in_assistant = True
    #             start_idx = i
    #         elif "<|end|>" in token and in_assistant:
    #             if start_idx is not None:
    #                 assistant_ranges.append((start_idx, i - 1))
    #             in_assistant = False
    #             start_idx = None

    #     return assistant_ranges

    def _handle_image_tokens(self, input_ids: torch.Tensor, labels: torch.Tensor) -> list[int]:
        expanded_labels: list[int] = []

        for i, token_id in enumerate(input_ids):
            expanded_labels.append(cast(int, labels[i].item()))

            if token_id == self.image_token_id:
                expanded_labels.extend([-100] * (self.image_token_size - 1))

        return expanded_labels

    def collate_fn(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Custom collate function for batching data together."""

        # Get maximum lengths for padding
        max_text_length = max(len(item["text"]) for item in batch)
        max_label_length = max(len(item["label"]) for item in batch)

        # Prepare containers
        images: list[torch.Tensor] = []
        input_ids: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []

        # Process each item in the batch
        for item in batch:
            # Pad input sequences
            text_pad_length = max_text_length - len(item["text"])
            padded_input_ids = torch.cat(
                [
                    torch.tensor(item["text"]),
                    torch.full((text_pad_length,), cast(int, self.tokenizer.pad_token_id)),
                ]
            )

            # Pad label sequences (using -100 as padding)
            label_pad_length = max_label_length - len(item["label"])
            padded_labels = torch.cat(
                [torch.tensor(item["label"]), torch.full((label_pad_length,), -100)]
            )

            # Add to lists
            input_ids.append(padded_input_ids)
            labels.append(padded_labels)
            images.append(torch.tensor(item["image"]))

        # Stack all tensors
        return {
            "images": torch.stack(images),
            "texts": torch.stack(input_ids),
            "labels": torch.stack(labels),
        }

    def get_dataloader(self, split: Literal["train", "val", "test"]) -> DataLoader[Dataset] | None:
        dataset = self.get_dataset(split)
        if not dataset:
            return None

        return DataLoader(
            dataset,  # pyright: ignore
            batch_size=self.batch_size,
            shuffle=(split == "train"),  # Only shuffle training data
            collate_fn=self.collate_fn,
            num_workers=self.dataset_config.num_workers,
            pin_memory=self.dataset_config.pin_memory,
            persistent_workers=self.dataset_config.persistent_workers,
        )

    @property
    def train_dataloader(self) -> DataLoader[Dataset] | None:
        return self.get_dataloader("train")

    @property
    def val_dataloader(self) -> DataLoader[Dataset] | None:
        return self.get_dataloader("val")

    @property
    def test_dataloader(self) -> DataLoader[Dataset] | None:
        return self.get_dataloader("test")
