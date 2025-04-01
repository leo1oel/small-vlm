import os
import logging
from typing import Literal, cast, Callable, Any
import json
from PIL import Image
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict, Dataset  # pyright: ignore
from transformers import PreTrainedTokenizer, BaseImageProcessor

from ..config.config_schema import DatasetConfig, InferenceConfig
from ..models.model import VLM

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
log = logging.getLogger(__name__)


class DataModule:
    def __init__(
        self,
        dataset_config: DatasetConfig,
        model: VLM,
    ):
        self.dataset_config: DatasetConfig = dataset_config
        self.model: VLM = model

        # model components
        self.tokenizer: PreTrainedTokenizer = self.model.language_model.tokenizer
        self.image_preprocessor: BaseImageProcessor = self.model.visual_encoder.preprocessor
        self.image_token_id: int = cast(int, self.model.language_model.token_config.image_token_id)
        self.image_token_size: int = self.model.visual_encoder.token_size

        self.transform: Callable[[dict[str, Image.Image | str], bool], dict[str, torch.Tensor | list[torch.Tensor]]] = self._build_transform()

    def get_dataset(self, split: Literal["train", "val", "test"]) -> Dataset | None:
        try:
            dataset_type = self.dataset_config.type
            if dataset_type == "huggingface":
                log.info(f"Loading HuggingFace dataset: {self.dataset_config.name} ({split})")
                # Load the dataset
                dataset: DatasetDict = cast(DatasetDict, load_dataset(
                    self.dataset_config.hf_name,
                    trust_remote_code=True
                ))

                map_dataset: DatasetDict = dataset.map(
                    self.transform,
                    num_proc=getattr(self.dataset_config, "num_proc", None)
                )

                return map_dataset[split]
            else:
                log.error(f"Dataset type {dataset_type} not supported")
                raise ValueError(f"Dataset type {dataset_type} not supported")
        except Exception as e:
            log.error(f"Failed to load {split} dataset: {str(e)}")
            return None

    def _build_transform(
        self,
    ) -> Callable[[dict[str, Image.Image | str], bool], dict[str, torch.Tensor | list[torch.Tensor]]]:
        def transform(
            item: dict[str, Image.Image | str], do_generation: bool = False
        ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
            image_tensor = self._process_image(item)

            text_str = self._extract_text(item)

            text = json.loads(text_str.replace("\n", "\\n"))
            text_and_label = self._text_transform(
                text, do_generation
            )

            return {
                "image": cast(torch.Tensor, image_tensor),
                "text": text_and_label[0],
                "label": text_and_label[1]
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
            input_image = self.image_preprocessor(
                original_image, return_tensors="pt"
            )
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
        self,
        text: list[dict[str, str]],
        do_generation: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conversation = self._prepare_conversation(text)

        input_ids = self._apply_chat_template(conversation, do_generation)

        labels = self._prepare_labels(input_ids)

        expanded_labels = self._handle_image_tokens(input_ids, labels)
        return (input_ids, torch.tensor(expanded_labels))

    def _prepare_conversation(self, text: list[dict[str, str]]) -> list[dict[str, str]]:
        return [
            {"role": "user" if item["from"] == "human" else "assistant", "content": item["value"]}
            for item in text
        ]

    def _apply_chat_template(self, conversation: list[dict[str, str]], do_generation: bool) -> torch.Tensor:
        input_ids = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=do_generation,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )[0]

        return cast(torch.Tensor, input_ids)

    def _prepare_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        labels = torch.full_like(input_ids, -100)
        labels[:-1] = input_ids[1:].clone()

        assistant_ranges = self._find_assistant_ranges(input_ids)

        for i in range(len(labels)):
            is_in_assistant_range = any(start <= i <= end for start, end in assistant_ranges)
            if not is_in_assistant_range:
                labels[i] = -100

        return labels

    def _find_assistant_ranges(self, input_ids: torch.Tensor) -> list[tuple[int, int]]:
        assistant_ranges: list[tuple[int, int]] = []
        in_assistant = False
        start_idx = None

        for i, token_id in enumerate(input_ids):
            token = self.tokenizer.decode([token_id])

            if "<|assistant|>" in token:
                in_assistant = True
                start_idx = i
            elif "<|end|>" in token and in_assistant:
                if start_idx is not None:
                    assistant_ranges.append((start_idx, i - 1))
                in_assistant = False
                start_idx = None

        return assistant_ranges

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
            padded_input_ids = torch.cat([
                torch.tensor(item["text"]),
                torch.full((text_pad_length,), cast(int, self.tokenizer.pad_token_id))
            ])

            # Pad label sequences (using -100 as padding)
            label_pad_length = max_label_length - len(item["label"])
            padded_labels = torch.cat([
                torch.tensor(item["label"]),
                torch.full((label_pad_length,), -100)
            ])

            # Add to lists
            input_ids.append(padded_input_ids)
            labels.append(padded_labels)
            images.append(torch.tensor(item["image"]))

        # Stack all tensors
        return {
            "images": torch.stack(images),
            "texts": torch.stack(input_ids),
            "labels": torch.stack(labels)
        }

    def get_dataloader(self, split: Literal["train", "val", "test"]) -> DataLoader[Dataset] | None:
        dataset = self.get_dataset(split)
        if not dataset:
            return None

        return DataLoader(
            dataset,  # pyright: ignore
            batch_size=self.dataset_config.batch_size,
            shuffle=(split == "train"),  # Only shuffle training data
            collate_fn=self.collate_fn,
            num_workers=self.dataset_config.num_workers,
            pin_memory=self.dataset_config.pin_memory,
            persistent_workers=self.dataset_config.persistent_workers
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


class InferenceDataModule:
    """Data module specifically for inference."""

    def __init__(self, config: InferenceConfig):
        self.config: InferenceConfig = config

    # def get_dataloader(self) -> DataLoader[Dataset] | None:
    #     try:
    #         dataset: Dataset = load_dataset(
    #             self.config.hf_name,
    #             split=self.config.split,
    #             trust_remote_code=True
    #         )

    #         return DataLoader(
    #             dataset,
    #             batch_size=self.config.batch_size,
    #             shuffle=False,
    #             num_workers=self.config.num_workers,
    #             pin_memory=True,
    #             persistent_workers=True
    #         )
    #     except Exception as e:
    #         logger.error(f"Failed to load inference dataset: {str(e)}")
    #         return None