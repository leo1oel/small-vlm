import copy
import json
import logging
import os
import random
from collections.abc import Sequence

import torch
import transformers
import webdataset as wds
from huggingface_hub import HfFileSystem, get_token, hf_hub_url
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from transformers.image_processing_utils import BaseImageProcessor

from ..models.processing_vlm import VLMProcessor
from .data_arguments import DataArguments
from .dataset import preprocess

ImageFile.LOAD_TRUNCATED_IMAGES = True

log: logging.Logger = logging.getLogger(name=__name__)


def create_clip_webdataset_iterator(urls: str, tokenizer, image_preprocess, multi_gpu: bool = True):
    """Create webdataset iterator for CLIP data with multi-GPU support"""
    field_names = {
        "image": "image.jpg",
        "text": "synthetic_caption.txt",
    }

    # Build pipeline components
    pipeline_components = []

    # Handle URL format for SimpleShardList vs WebDataset
    if urls.startswith("pipe:"):
        # For pipe URLs, we need to use WebDataset directly
        if multi_gpu:
            print(
                "⚠️ WARNING: Using pipe URLs with multi-GPU training. Consider using direct tar URLs for better performance."
            )

        dataset = (
            wds.WebDataset(urls, shardshuffle=True)
            .decode(
                wds.autodecode.ImageHandler("pil"),
                handler=wds.warn_and_continue,
            )
            .rename(
                **field_names,
                handler=wds.warn_and_continue,
            )
            .map_dict(
                image=lambda img: image_preprocess(img) if img is not None else None,
                text=lambda text: tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                )
                if text is not None
                else None,
                handler=wds.warn_and_continue,
            )
        )
    else:
        # For direct URLs, build explicit pipeline for multi-GPU support
        # First, expand URLs if they contain braceexpand patterns
        import braceexpand

        if "::" in urls:
            url_list = urls.split("::")
        else:
            url_list = [urls]

        # Expand any brace patterns like {0000..0575}
        expanded_urls = []
        for url in url_list:
            expanded = list(braceexpand.braceexpand(url))
            expanded_urls.extend(expanded)
            log.info(f"Expanded URL pattern '{url}' to {len(expanded)} URLs")

        log.info(f"Total URLs after expansion: {len(expanded_urls)}")
        if len(expanded_urls) <= 5:  # Show URLs if not too many
            for i, url in enumerate(expanded_urls):
                log.info(f"  URL {i}: {url}")
        else:
            log.info(f"  First URL: {expanded_urls[0]}")
            log.info(f"  Last URL: {expanded_urls[-1]}")

        pipeline_components = [
            wds.SimpleShardList(expanded_urls),
            wds.shuffle(1000),  # Shard shuffle
        ]

        if multi_gpu:
            pipeline_components.extend(
                [
                    wds.split_by_node,  # Critical for multi-node training
                    wds.split_by_worker,  # Critical for multi-worker training
                ]
            )

        pipeline_components.extend(
            [
                wds.tarfile_to_samples(handler=wds.warn_and_continue),
                wds.shuffle(5000),  # Sample shuffle
                wds.decode(
                    wds.autodecode.ImageHandler("pil"),
                    handler=wds.warn_and_continue,
                ),
                wds.rename(
                    **field_names,
                    handler=wds.warn_and_continue,
                ),
                wds.map_dict(
                    image=lambda img: image_preprocess(img) if img is not None else None,
                    text=lambda text: tokenizer(
                        text,
                        padding="max_length",
                        truncation=True,
                        max_length=77,
                        return_tensors="pt",
                    )
                    if text is not None
                    else None,
                    handler=wds.warn_and_continue,
                ),
            ]
        )

        dataset = wds.DataPipeline(*pipeline_components)

    return dataset


def create_clip_webdataset_from_hf(
    dataset_path: str, split: str = "train", estimated_length: int = None
):
    """Create webdataset URLs from HuggingFace dataset path"""
    splits = {
        "train": "**/*-train-*.tar",
        "validation": "**/*-validation-*.tar",
        "test": "**/*-test-*.tar",
    }

    fs = HfFileSystem()
    pattern = splits.get(split, splits["train"])
    files = [fs.resolve_path(path) for path in fs.glob(f"hf://datasets/{dataset_path}/" + pattern)]
    urls = [hf_hub_url(file.repo_id, file.path_in_repo, repo_type="dataset") for file in files]

    token = get_token()
    if token:
        urls_str = f"pipe: curl -s -L -H 'Authorization:Bearer {token}' " + "::".join(
            [f"'{url}'" for url in urls]
        )
    else:
        urls_str = "::".join(urls)

    return urls_str, estimated_length or 100000  # Return URL and estimated length


class ImagePreprocess:
    """Image preprocessing class that handles image transformations."""

    def __init__(self, image_processor: BaseImageProcessor, data_args: DataArguments):
        self.image_processor = image_processor
        self.data_args = data_args

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Process PIL image into tensor format."""
        image_aspect_ratio = getattr(self.data_args, "image_aspect_ratio", "square")

        if image_aspect_ratio == "pad":
            image = self._expand2square(
                image, tuple(int(x * 255) for x in self.image_processor.image_mean)
            )

        processed = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return processed

    def _expand2square(
        self, pil_img: Image.Image, background_color: tuple[int, int, int]
    ) -> Image.Image:
        """Expand image to square by padding."""
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result


class UnifiedDataset(Dataset):
    def __init__(
        self,
        vlm_data_path: str,
        clip_data_path: str,
        processor: VLMProcessor,
        clip_tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        vlm_batch_size: int,
        clip_batch_size: int,
        training_strategy: str = "min_length",
    ):
        super().__init__()

        self.tokenizer: transformers.PreTrainedTokenizer = processor.tokenizer
        self.image_processor: BaseImageProcessor = processor.image_processor
        self.clip_tokenizer = clip_tokenizer
        self.data_args = data_args
        self.training_strategy = training_strategy
        self.vlm_batch_size = vlm_batch_size
        self.clip_batch_size = clip_batch_size

        # Store data paths for lazy loading
        self.vlm_data_path = vlm_data_path
        self.clip_data_path = clip_data_path
        self.vlm_data = None
        self.clip_data = None

        # Initialize preprocessors
        self.image_preprocess = ImagePreprocess(self.image_processor, data_args)

        # Initialize CLIP data iterator for webdataset
        self.clip_webdataset_iterator = None
        self.clip_iterator_exhausted = False

        # Create shuffled indices for VLM data to ensure each sample is used once
        self.vlm_shuffled_indices = None

        # Get dataset lengths
        self.vlm_length = self._get_dataset_length(vlm_data_path)

        if clip_batch_size > 0:
            if data_args.clip_data_type == "webdataset":
                if data_args.clip_dataset_size is None:
                    raise ValueError(
                        "clip_dataset_size must be specified when using webdataset for CLIP data"
                    )
                self.clip_length = data_args.clip_dataset_size
                # Initialize webdataset iterator
                self._init_clip_webdataset()
            else:
                # Traditional JSON loading
                self.clip_length = self._get_dataset_length(clip_data_path)
        else:
            self.clip_length = 0

        # Calculate total effective samples without creating huge indices
        if self.vlm_batch_size == 0 and self.clip_batch_size == 0:
            raise ValueError("Both vlm_batch_size and clip_batch_size cannot be 0")

        if self.vlm_batch_size == 0:
            self.total_samples = self.clip_length
        elif self.clip_batch_size == 0:
            self.total_samples = self.vlm_length
        else:
            # Calculate how many super batches we can make
            super_batch_size = self.vlm_batch_size + self.clip_batch_size
            num_super_batches = min(
                self.vlm_length // self.vlm_batch_size, self.clip_length // self.clip_batch_size
            )
            self.total_samples = num_super_batches * super_batch_size

        log.info("=== UnifiedDataset Initialized ===")
        log.info(f"VLM data: {self.vlm_length} samples")
        log.info(f"CLIP data: {self.clip_length} samples")
        log.info(f"VLM batch size: {vlm_batch_size}")
        log.info(f"CLIP batch size: {clip_batch_size}")

        if vlm_batch_size == 0:
            log.info("Mode: CLIP-only training")
        elif clip_batch_size == 0:
            log.info("Mode: VLM-only training")
        else:
            log.info(f"Super batch size: {vlm_batch_size + clip_batch_size}")
            num_super_batches = min(
                self.vlm_length // vlm_batch_size, self.clip_length // clip_batch_size
            )
            log.info(f"Number of super batches: {num_super_batches}")

        log.info(f"Total effective samples: {self.total_samples}")
        log.info("==" * 20)

    def _init_clip_webdataset(self):
        """Initialize CLIP webdataset iterator"""
        if self.data_args.clip_data_type == "webdataset":
            if self.data_args.clip_webdataset_urls:
                urls = self.data_args.clip_webdataset_urls
            elif self.clip_data_path and self.clip_data_path.startswith("hf://"):
                # Create URLs from HuggingFace path
                dataset_path = self.clip_data_path.replace("hf://datasets/", "")
                urls, _ = create_clip_webdataset_from_hf(dataset_path)
            else:
                raise ValueError(
                    "Either clip_webdataset_urls or HF dataset path must be provided for webdataset"
                )

            # Create the webdataset iterator with multi-GPU support
            self.clip_webdataset_iterator = create_clip_webdataset_iterator(
                urls, self.clip_tokenizer, self.image_preprocess, multi_gpu=True
            )
            self.clip_webdataset_iter = iter(self.clip_webdataset_iterator)
            log.info(f"Initialized CLIP webdataset with URLs: {urls}")

    def _get_dataset_length(self, data_path: str) -> int:
        """Get dataset length without loading all data"""
        with open(data_path) as f:
            # Skip to end and count lines, or use a more efficient method
            data = json.load(f)
            return len(data)

    def _lazy_load_vlm_data(self):
        """Lazily load VLM data when needed"""
        if self.vlm_data is None:
            log.info("Lazy loading VLM data...")
            with open(self.vlm_data_path) as f:
                self.vlm_data = json.load(f)

            # Create shuffled indices for VLM data (ensures each sample used once per epoch)
            if self.vlm_shuffled_indices is None:
                self.vlm_shuffled_indices = list(range(len(self.vlm_data)))
                random.shuffle(self.vlm_shuffled_indices)
                log.info(
                    f"Created shuffled indices for {len(self.vlm_shuffled_indices)} VLM samples"
                )

    def _lazy_load_clip_data(self):
        """Lazily load CLIP data when needed"""
        if self.clip_data is None:
            log.info("Lazy loading CLIP data...")
            with open(self.clip_data_path) as f:
                self.clip_data = json.load(f)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        """Get dataset sample with automatic retry for invalid samples"""
        return self._get_item_with_retry(i)

    def _get_item_with_retry(self, i: int, max_retries: int = 100) -> dict[str, torch.Tensor]:
        """Get dataset item with retry mechanism to skip invalid samples"""
        original_i = i
        attempts = 0

        while attempts < max_retries:
            sample = self._get_item_internal(i)
            if sample is not None:
                if attempts > 0:
                    log.info(f"Found valid sample at index {i} after {attempts} attempts")
                return sample

            # Skip to next sample if current one is invalid
            attempts += 1
            i = (i + 1) % self.total_samples

            # Log periodically to avoid spam
            if attempts % 10 == 1:
                log.warning(
                    f"Skipped {attempts} invalid samples, still searching (started from index {original_i})"
                )

        # If we reach here, there might be a serious data issue
        raise RuntimeError(
            f"Failed to find valid sample after {max_retries} attempts starting from index {original_i}. "
            f"This suggests a serious issue with your dataset - most samples appear to be invalid."
        )

    def _get_item_internal(self, i: int) -> dict[str, torch.Tensor] | None:
        """Internal method to get dataset item (can return None for invalid samples)"""
        # Calculate task type and data index dynamically
        if self.vlm_batch_size == 0:
            # Only CLIP data
            return self._get_clip_sample(i % self.clip_length)
        elif self.clip_batch_size == 0:
            # Only VLM data
            return self._get_vlm_sample(i % self.vlm_length)
        else:
            # Mixed batches
            super_batch_size = self.vlm_batch_size + self.clip_batch_size
            super_batch_idx = i // super_batch_size
            position_in_super_batch = i % super_batch_size

            if position_in_super_batch < self.vlm_batch_size:
                # VLM sample - use shuffled indices to ensure each sample used once
                vlm_seq_idx = super_batch_idx * self.vlm_batch_size + position_in_super_batch
                vlm_idx = vlm_seq_idx % self.vlm_length  # Handle wraparound for multiple epochs
                return self._get_vlm_sample(vlm_idx)
            else:
                # CLIP sample
                clip_position = position_in_super_batch - self.vlm_batch_size
                clip_idx = super_batch_idx * self.clip_batch_size + clip_position
                return self._get_clip_sample(clip_idx % self.clip_length)

    def _get_vlm_sample(self, idx: int) -> dict[str, torch.Tensor]:
        """Get VLM sample following LazySupervisedDataset pattern"""
        try:
            self._lazy_load_vlm_data()
            # Use shuffled index to ensure random order but no repetition within epoch
            actual_idx = self.vlm_shuffled_indices[idx]
            sample = self.vlm_data[actual_idx]
            data_dict = preprocess(
                [copy.deepcopy(sample["conversations"])],
                self.tokenizer,
                self.data_args,
                has_image=True,
            )
        except Exception as e:
            log.error(f"Error in _get_vlm_sample(idx={idx}): {e}")
            return None

        # Check if preprocess returned valid data
        if not data_dict or "input_ids" not in data_dict or "labels" not in data_dict:
            log.warning(f"Preprocess returned invalid data for VLM sample {idx}")
            return None

        # Extract single tensors from the lists (since we only have one conversation)
        if isinstance(data_dict["input_ids"], list):
            data_dict["input_ids"] = data_dict["input_ids"][0]
        if isinstance(data_dict["labels"], list):
            data_dict["labels"] = data_dict["labels"][0]

        # Ensure tensors are properly shaped (squeeze any extra dimensions)
        if isinstance(data_dict["input_ids"], torch.Tensor) and data_dict["input_ids"].dim() > 1:
            data_dict["input_ids"] = data_dict["input_ids"].squeeze()
        if isinstance(data_dict["labels"], torch.Tensor) and data_dict["labels"].dim() > 1:
            data_dict["labels"] = data_dict["labels"].squeeze()

        data_dict["task_mode"] = "vlm"

        # Add image if present
        if "image" in sample:
            image_file = sample["image"]
            image_folder = self.data_args.image_folder
            try:
                image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
                processed_image = self.image_preprocess(image)
                image_size = image.size  # (width, height)
                # Format as expected by DataCollatorForSupervisedDataset: list of (image_tensor, image_size, modality) tuples
                data_dict["image"] = [(processed_image, image_size, "image")]
            except (FileNotFoundError, OSError) as e:
                # Use both logging and print for maximum visibility in HuggingFace Trainer
                error_msg = (
                    f"VLM image file not found: {os.path.join(image_folder, image_file)} "
                    f"(sample_id: {sample.get('id', 'unknown')}) - SKIPPING this sample. Error: {e}"
                )
                log.warning(error_msg)
                print(f"⚠️ DATASET WARNING: {error_msg}")
                # Return None to indicate this sample should be skipped
                return None
        elif self.data_args.is_multimodal:
            # Create dummy image for multimodal models
            crop_size = getattr(
                self.image_processor,
                "crop_size",
                getattr(self.image_processor, "size", {"height": 224, "width": 224}),
            )
            dummy_image = torch.zeros(3, crop_size["height"], crop_size["width"])
            data_dict["image"] = [(dummy_image, (crop_size["width"], crop_size["height"]), "image")]

        return data_dict

    def _get_clip_sample(self, idx: int) -> dict:
        """Get CLIP sample following CLIPDataset pattern"""
        if self.data_args.clip_data_type == "webdataset":
            return self._get_clip_sample_from_webdataset()
        else:
            return self._get_clip_sample_from_json(idx)

    def _get_clip_sample_from_webdataset(self) -> dict:
        """Get CLIP sample from webdataset iterator"""
        try:
            if self.clip_iterator_exhausted:
                # Reset iterator when exhausted
                self.clip_webdataset_iter = iter(self.clip_webdataset_iterator)
                self.clip_iterator_exhausted = False

            sample = next(self.clip_webdataset_iter)

            return {
                "task_mode": "clip",
                "image": sample["image"],  # Already preprocessed
                "clip_input_ids": sample["text"]["input_ids"].squeeze(0),
                "clip_attention_mask": sample["text"]["attention_mask"].squeeze(0),
            }

        except StopIteration:
            # Iterator exhausted, will reset on next call
            self.clip_iterator_exhausted = True
            # Return a dummy sample to maintain training flow
            crop_size = getattr(
                self.image_processor,
                "crop_size",
                getattr(self.image_processor, "size", {"height": 224, "width": 224}),
            )
            dummy_image = torch.zeros(3, crop_size["height"], crop_size["width"])
            dummy_text = self.clip_tokenizer(
                "dummy", padding="max_length", truncation=True, max_length=77, return_tensors="pt"
            )
            return {
                "task_mode": "clip",
                "image": dummy_image,
                "clip_input_ids": dummy_text["input_ids"].squeeze(0),
                "clip_attention_mask": dummy_text["attention_mask"].squeeze(0),
            }
        except Exception as e:
            log.error(f"Error in _get_clip_sample_from_webdataset: {e}")
            return None

    def _get_clip_sample_from_json(self, idx: int) -> dict:
        """Get CLIP sample from JSON data"""
        try:
            self._lazy_load_clip_data()
            sample = self.clip_data[idx]
        except Exception as e:
            log.error(f"Error in _get_clip_sample_from_json(idx={idx}): {e}")
            return None

        # Extract caption from conversation format
        caption = self._extract_caption(sample)
        text_inputs = self.clip_tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )

        # Load image
        if "image" not in sample:
            raise ValueError("CLIP sample must contain image")

        image_file = sample["image"]
        clip_image_folder = getattr(
            self.data_args, "clip_image_folder", self.data_args.image_folder
        )
        try:
            image = Image.open(os.path.join(clip_image_folder, image_file)).convert("RGB")
            processed_image = self.image_preprocess(image)
        except (FileNotFoundError, OSError) as e:
            # Use both logging and print for maximum visibility in HuggingFace Trainer
            error_msg = (
                f"CLIP image file not found: {os.path.join(clip_image_folder, image_file)} "
                f"(sample_id: {sample.get('id', 'unknown')}) - SKIPPING this sample. Error: {e}"
            )
            log.warning(error_msg)
            print(f"⚠️ DATASET WARNING: {error_msg}")
            # Return None to indicate this sample should be skipped
            return None

        return {
            "task_mode": "clip",
            "image": processed_image,  # Note: singular 'image' for consistency with VLM
            "clip_input_ids": text_inputs.input_ids.squeeze(0),  # Remove batch dimension
            "clip_attention_mask": text_inputs.attention_mask.squeeze(0),  # Remove batch dimension
        }

    def _extract_caption(self, sample):
        """Extract caption from conversation format"""
        conversations = sample["conversations"]
        gpt_response = None

        for conv in conversations:
            if conv["from"] == "gpt":
                gpt_response = conv["value"]
                break

        if gpt_response is None:
            raise ValueError("No GPT response found in CLIP sample")

        # Extract caption by removing template prefixes and suffixes
        caption = gpt_response

        # Remove possible template prefixes
        prefixes_to_remove = [
            "This image shows ",
            "I can see ",
            "In this picture, ",
            "This image contains ",
            "The image shows ",
        ]

        for prefix in prefixes_to_remove:
            if caption.startswith(prefix):
                caption = caption[len(prefix) :]
                break

        # Remove trailing period
        if caption.endswith("."):
            caption = caption[:-1]

        return caption.strip()


class UnifiedDataCollator:
    """
    Data collator that handles mixed batches of VLM and CLIP samples.

    Separates samples by task_mode and applies appropriate collation for each task type.
    """

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer
        from .dataset import DataCollatorForSupervisedDataset

        self.vlm_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    def __call__(self, instances: Sequence[dict]):
        # All instances should be valid since __getitem__ now handles invalid samples
        if not instances:
            log.warning("DataCollator: Empty batch received")
            return {}

        # Separate instances by task type
        vlm_instances = [inst for inst in instances if inst["task_mode"] == "vlm"]
        clip_instances = [inst for inst in instances if inst["task_mode"] == "clip"]

        batch = {"task_modes": []}

        # Process VLM instances
        if vlm_instances:
            vlm_batch = self.vlm_collator(vlm_instances)
            for key, value in vlm_batch.items():
                batch[key] = value
            batch["task_modes"].extend(["vlm"] * len(vlm_instances))

        # Process CLIP instances
        if clip_instances:
            clip_images = torch.stack([inst["image"] for inst in clip_instances])
            clip_input_ids = torch.stack([inst["clip_input_ids"] for inst in clip_instances])
            clip_attention_mask = torch.stack(
                [inst["clip_attention_mask"] for inst in clip_instances]
            )

            batch["clip_images"] = clip_images
            batch["clip_input_ids"] = clip_input_ids
            batch["clip_attention_mask"] = clip_attention_mask
            batch["task_modes"].extend(["clip"] * len(clip_instances))

        return batch


def make_unified_data_module(
    processor: transformers.PreTrainedTokenizer,
    clip_tokenizer: transformers.PreTrainedTokenizer,
    data_args: DataArguments,
    training_strategy: str = "min_length",
) -> dict:
    train_dataset = UnifiedDataset(
        vlm_data_path=data_args.data_path,
        clip_data_path=data_args.clip_data_path,
        processor=processor,
        clip_tokenizer=clip_tokenizer,
        data_args=data_args,
        training_strategy=training_strategy,
        vlm_batch_size=data_args.vlm_batch_size,
        clip_batch_size=data_args.clip_batch_size,
    )

    data_collator = UnifiedDataCollator(tokenizer=processor.tokenizer)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
