import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast, override

import lightning as L
import torch
from datasets import Dataset, DatasetDict, load_dataset  # pyright: ignore
from datasets import Image as HFImage
from PIL import Image
from torch.utils.data import DataLoader
from transformers.image_processing_utils import BaseImageProcessor
from transformers.tokenization_utils import PreTrainedTokenizer

from ..config.config_schema import DatasetConfig
from ..models import VLM
from ..utils.chat_template import get_chat_template

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
log = logging.getLogger(__name__)


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        num_training_samples: int | None,
        model: VLM,
        batch_size: int,
        chat_template: str,
        do_generation: bool = False,
        processed_data_dir: str = "../../../processed_data",
    ):
        super().__init__()
        self.model: VLM = model
        self.dataset_config: DatasetConfig = dataset_config
        self.batch_size: int = batch_size
        self._raw_dataset: DatasetDict | None = None
        self._processed_datasets: dict[str, Dataset] = {}
        self.num_training_samples: int | None = num_training_samples
        self.num_samples: dict[str, int] = {}
        self.do_generation: bool = do_generation
        self.processed_data_dir: Path = Path(processed_data_dir)

        # model components
        self.tokenizer: PreTrainedTokenizer = model.language_model.tokenizer
        self.image_preprocessor: BaseImageProcessor = model.visual_encoder.preprocessor
        self.image_token_id: int = cast(int, model.language_model.token_config.image_token_id)
        self.image_token_size: int = model.visual_encoder.token_size
        self.eos_token: str = cast(str, self.tokenizer.eos_token)
        self.eos_token_id: int = cast(int, self.tokenizer.eos_token_id)
        self.system_token: str = model.language_model.token_config.system_token
        self.user_token: str = model.language_model.token_config.user_token
        self.assistant_token: str = model.language_model.token_config.assistant_token

        self.chat_template_name: str = chat_template
        self.chat_template: str = get_chat_template(
            chat_template, self.eos_token, self.system_token, self.user_token, self.assistant_token
        )

        self.transform: Callable[
            [dict[str, Image.Image | str]], dict[str, torch.Tensor | Image.Image]
        ] = self._build_transform()

    def _load_raw_dataset(self):
        try:
            dataset_type = self.dataset_config.type
            if dataset_type == "huggingface":
                log.info(f"Loading HuggingFace dataset: {self.dataset_config.name}")
                self._raw_dataset = cast(
                    DatasetDict,
                    load_dataset(cast(str, self.dataset_config.hf_name), trust_remote_code=True),
                )
                if self.num_training_samples is not None:
                    self._raw_dataset["train"] = self._raw_dataset["train"].select(
                        range(min(self.num_training_samples, len(self._raw_dataset["train"])))
                    )
            elif dataset_type == "json":
                log.info(f"Loading JSON dataset: {self.dataset_config.name}")
                self._raw_dataset = cast(
                    DatasetDict,
                    load_dataset(
                        "json",
                        data_files=self.dataset_config.json_path,
                    ).cast_column("image", HFImage()),
                )
                if self.num_training_samples is not None:
                    log.info(f"Selecting {self.num_training_samples} training samples")
                    self._raw_dataset["train"] = self._raw_dataset["train"].select(
                        range(min(self.num_training_samples, len(self._raw_dataset["train"])))
                    )
            else:
                log.warning(f"Dataset type {dataset_type} not supported")
                raise ValueError(f"Dataset type {dataset_type} not supported")
        except Exception as e:
            log.warning(f"Failed to load raw dataset: {str(e)}", exc_info=True)
            self._raw_dataset = None

    @override
    def prepare_data(self) -> None:
        """
        Downloads/loads the raw dataset and processes (maps) it.
        This method is called only once per node (on rank 0).
        """
        if self._raw_dataset is None:
            log.info("Loading raw dataset in prepare_data...")
            self._load_raw_dataset()

        if self._raw_dataset is None:
            log.error("Raw dataset failed to load. Cannot prepare data.")
            return

        base_processed_data_path = self.processed_data_dir / self.dataset_config.name
        absolute_processed_data_path = base_processed_data_path.resolve()
        base_processed_data_path.mkdir(parents=True, exist_ok=True)
        log.info(f"Ensured processed data directory exists: {absolute_processed_data_path}")

        for split in self._raw_dataset.keys():  # pyright: ignore
            processed_split_path = base_processed_data_path / cast(str, split)
            absolute_split_path = processed_split_path.resolve()
            if processed_split_path.exists():
                log.info(
                    f"Processed data for split '{split}' already exists at {absolute_split_path}. Skipping map."
                )
                continue

            log.info(f"Processing raw dataset for split: {split} in prepare_data...")
            try:
                num_proc = getattr(self.dataset_config, "num_proc", None)
                log.info(f"Using {num_proc} processes for mapping split '{split}'")

                processed_split = self._raw_dataset[split].map(
                    self.transform,
                    num_proc=num_proc,
                    load_from_cache_file=True,
                    desc=f"Processing {split} split",
                )
                log.info(f"Saving processed split '{split}' to {absolute_split_path}")
                processed_split.save_to_disk(processed_split_path)

            except Exception as e:
                log.warning(
                    f"Failed to process dataset for split {split} in prepare_data: {str(e)}"
                )
                if split == "train":
                    raise e

    @override
    def setup(self, stage: str | None = None) -> None:
        """
        Loads the processed dataset from disk.
        This method is called on every GPU process.
        """
        if not self._processed_datasets:
            base_processed_data_path = self.processed_data_dir / self.dataset_config.name
            absolute_processed_data_path = base_processed_data_path.resolve()
            log.info(
                f"Loading processed datasets from {absolute_processed_data_path} in setup (stage: {stage})..."
            )
            splits_to_load = []
            if stage == "fit" or stage is None:
                splits_to_load.extend(["train", "val"])
            if stage == "validate" or stage is None:
                splits_to_load.append("val")
            if stage == "test" or stage is None:
                splits_to_load.append("test")
            if stage == "predict" or stage is None:
                predict_split_path = base_processed_data_path / "predict"
                train_split_path = base_processed_data_path / "train"
                if predict_split_path.exists():
                    splits_to_load.append("predict")
                elif train_split_path.exists():  # Use Path.exists()
                    splits_to_load.append("train")  # Fallback to train if predict doesn't exist

            splits_to_load: list[Any] = list(set(splits_to_load))

            for split in splits_to_load:
                processed_split_path = base_processed_data_path / split
                absolute_split_path = processed_split_path.resolve()
                if not processed_split_path.exists():
                    log.warning(
                        f"Processed data for split '{split}' not found at {absolute_split_path}. Skipping."
                    )
                    continue
                try:
                    log.info(
                        f"Loading processed dataset for split: {split} from {absolute_split_path}"
                    )
                    # load_from_disk accepts Path objects or strings
                    processed_split = Dataset.load_from_disk(processed_split_path)
                    self._processed_datasets[split] = processed_split
                    self.num_samples[split] = len(processed_split)
                    log.info(f"Loaded {self.num_samples[split]} samples for split '{split}'.")
                except Exception as e:
                    log.warning(
                        f"Failed to load processed dataset for split {split} from {absolute_split_path}: {str(e)}"
                    )
                    if split == "train" and stage == "fit":
                        raise e

    def get_dataset(self, split: Literal["train", "val", "test", "predict"]) -> Dataset | None:
        # Construct base path using Path
        base_processed_data_path = self.processed_data_dir / self.dataset_config.name

        if split == "predict":
            actual_split = "predict" if "predict" in self._processed_datasets else "train"
            log.info(f"Predict requested, using '{actual_split}' split.")
            ds = self._processed_datasets.get(actual_split)
        else:
            ds = self._processed_datasets.get(split)

        if ds is None:
            log.warning(f"Dataset for split '{split}' not found or loaded.")
            log.warning(f"Attempting direct load for split '{split}' (may indicate setup issue)")
            processed_split_path = base_processed_data_path / split
            absolute_split_path = processed_split_path.resolve()
            if processed_split_path.exists():
                try:
                    ds = Dataset.load_from_disk(processed_split_path)
                    self._processed_datasets[split] = ds
                    self.num_samples[split] = len(ds)
                except Exception as e:
                    log.warning(f"Direct load failed for split '{split}': {e}")
                    return None
            else:
                log.warning(f"Processed data path not found for direct load: {absolute_split_path}")
                return None

        return ds

    def _build_transform(
        self,
    ) -> Callable[[dict[str, Image.Image | str]], dict[str, torch.Tensor | Image.Image]]:
        return self._transform

    def _transform(self, item: dict[str, Any]) -> dict[str, Any]:
        original_text = self._extract_text(item)
        if isinstance(original_text, str):
            text = json.loads(original_text.replace("\n", "\\n"))
        else:
            text = original_text
        text_and_label = self._text_transform(text)

        item["text"] = text_and_label[0]
        item["label"] = text_and_label[1]
        return item

    def _extract_text(self, item: dict[str, Any]) -> str | list[dict[str, str]]:
        if "text" in item:
            return item["text"]
        elif "conversations" in item:
            return item["conversations"]
        else:
            error_msg = f"Cannot find text in item {item}"
            log.error(error_msg)
            raise ValueError(error_msg)

    def _text_transform(self, text: list[dict[str, str]]) -> tuple[torch.Tensor, torch.Tensor]:
        conversation = self._prepare_conversation(text)

        input_ids = self._apply_chat_template(conversation)

        labels = self._prepare_labels(input_ids)

        expanded_labels = self._handle_image_tokens(input_ids, labels)
        return (input_ids, torch.tensor(expanded_labels))

    def _prepare_conversation(self, text: list[dict[str, str]]) -> list[dict[str, str]]:
        # Check if the data is already in the correct format
        if all(("role" in item and "content" in item) for item in text):
            return text
        # Otherwise, convert from the expected format with "from" and "value" fields
        return [
            {"role": "user" if item["from"] == "human" else "assistant", "content": item["value"]}
            for item in text
        ]

    def _apply_chat_template(self, conversation: list[dict[str, str]]) -> torch.Tensor:
        self.tokenizer.chat_template = self.chat_template
        input_ids = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=self.do_generation,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )[0]

        return cast(torch.Tensor, input_ids)

    def _prepare_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.chat_template_name == "plain":
            labels = torch.full_like(input_ids, -100)
            if len(input_ids) > 1:
                labels[1:-1] = input_ids[2:].clone()
            return labels

        labels = torch.full_like(input_ids, -100)
        # Shift labels to the right by one
        labels[:-1] = input_ids[1:].clone()
        # Find the ranges of the assistant messages
        assistant_ranges = self._find_assistant_ranges(input_ids)
        # Set the labels to -100 for the tokens that are not in the assistant ranges
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

            if self.assistant_token in token:
                in_assistant = True
                start_idx = i
            elif self.eos_token in token and in_assistant:
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

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
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
            with torch.no_grad():
                img_tensor = self.image_preprocessor(
                    item["image"].convert("RGB"), return_tensors="pt", device="cpu"
                )["pixel_values"].squeeze(0)
            images.append(img_tensor)

        # Stack all tensors
        return {
            "images": torch.stack(images),
            "texts": torch.stack(input_ids),
            "labels": torch.stack(labels),
        }

    def get_dataloader(
        self, split: Literal["train", "val", "test", "predict"]
    ) -> DataLoader[Dataset] | list[DataLoader[Dataset]]:
        dataset = self.get_dataset(split)
        if not dataset:
            return []

        return DataLoader(
            dataset,  # pyright: ignore
            batch_size=self.batch_size,
            shuffle=(split == "train"),  # Only shuffle training data
            collate_fn=self.collate_fn,
            num_workers=self.dataset_config.num_workers,
            pin_memory=self.dataset_config.pin_memory,
            persistent_workers=self.dataset_config.persistent_workers,
        )

    @override
    def train_dataloader(self) -> DataLoader[Dataset] | list[DataLoader[Dataset]]:
        return self.get_dataloader("train")

    @override
    def val_dataloader(self) -> DataLoader[Dataset] | list[DataLoader[Dataset]]:
        return self.get_dataloader("val")

    @override
    def test_dataloader(self) -> DataLoader[Dataset] | list[DataLoader[Dataset]]:
        return self.get_dataloader("test")

    @override
    def predict_dataloader(self) -> DataLoader[Dataset] | list[DataLoader[Dataset]]:
        return self.get_dataloader("predict")
