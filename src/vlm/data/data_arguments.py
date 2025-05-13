from dataclasses import dataclass, field

from ..config import DatasetConfig, ModelConfig


@dataclass
class DataArguments:
    data_path: str | None = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = True
    is_multimodal: bool = True
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


def get_data_args(data_config: DatasetConfig, trainer_config: ModelConfig) -> DataArguments:
    return DataArguments(
        data_path=data_config.path,
        lazy_preprocess=data_config.lazy_preprocess,
        is_multimodal=data_config.is_multimodal,
        image_folder=data_config.image_folder,
        use_start_end_tokens=trainer_config.language_model.use_start_end_tokens,
        use_image_patch_token=trainer_config.language_model.use_image_patch_token,
        image_token=trainer_config.language_model.image_token,
        image_start_token=trainer_config.language_model.image_start_token,
        image_end_token=trainer_config.language_model.image_end_token,
        image_patch_token=trainer_config.language_model.image_patch_token,
        ignore_index=trainer_config.language_model.ignore_index,
        image_token_index=trainer_config.language_model.image_token_index,
        image_aspect_ratio=data_config.image_aspect_ratio,
    )
