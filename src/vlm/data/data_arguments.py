from dataclasses import dataclass, field

from ..config import DatasetConfig, ModelConfig


@dataclass
class DataArguments:
    data_path: str | None = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = True
    is_multimodal: bool = True
    early_mix_text: bool = False
    image_folder: str | None = field(default=None)
    use_start_end_tokens: bool = False
    use_image_patch_token: bool = False
    image_token: str = "<image>"
    image_start_token: str = "<im_start>"
    image_end_token: str = "<im_end>"
    image_patch_token: str = "<im_patch>"
    ignore_index: int = -100
    image_token_index: int = -200
    image_aspect_ratio: str = "square"
    clip_data_path: str | None = field(default=None)
    clip_image_folder: str | None = field(default=None)
    clip_webdataset_urls: str | None = field(
        default=None, metadata={"help": "WebDataset URLs for CLIP data"}
    )
    clip_data_type: str = field(
        default="json", metadata={"help": "CLIP data type: 'json' or 'webdataset'"}
    )
    clip_dataset_size: int | None = field(
        default=None,
        metadata={"help": "Number of samples in CLIP dataset (required for webdataset)"},
    )
    vlm_batch_size: int | None = field(default=None)
    clip_batch_size: int | None = field(default=None)


def get_data_args(data_config: DatasetConfig, trainer_config: ModelConfig) -> DataArguments:
    return DataArguments(
        data_path=data_config.path,
        lazy_preprocess=data_config.lazy_preprocess,
        is_multimodal=data_config.is_multimodal,
        image_folder=data_config.image_folder,
        image_token=trainer_config.language_model.image_token,
        early_mix_text=data_config.early_mix_text,
        use_start_end_tokens=trainer_config.language_model.use_start_end_tokens,
        use_image_patch_token=trainer_config.language_model.use_image_patch_token,
        image_start_token=trainer_config.language_model.image_start_token,
        image_end_token=trainer_config.language_model.image_end_token,
        image_patch_token=trainer_config.language_model.image_patch_token,
        ignore_index=trainer_config.language_model.ignore_index,
        image_token_index=trainer_config.language_model.image_token_index,
        image_aspect_ratio=data_config.image_aspect_ratio,
        clip_data_path=data_config.clip_data_path,
        clip_image_folder=data_config.clip_image_folder,
        clip_webdataset_urls=data_config.clip_webdataset_urls,
        clip_data_type=data_config.clip_data_type,
        clip_dataset_size=data_config.clip_dataset_size,
        vlm_batch_size=data_config.vlm_batch_size,
        clip_batch_size=data_config.clip_batch_size,
    )
