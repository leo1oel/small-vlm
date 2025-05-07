from dataclasses import dataclass, field

from transformers import BaseImageProcessor

from ..config.config_schema import DatasetConfig, ModelConfig


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
    image_preprocessor: BaseImageProcessor | None = field(
        default=None, metadata={"help": "Image preprocessor for the visual encoder."}
    )


def get_data_args(
    data_config: DatasetConfig, trainer_config: ModelConfig, image_processor: BaseImageProcessor
) -> DataArguments:
    return DataArguments(
        data_path=data_config.path,
        lazy_preprocess=data_config.lazy_preprocess,
        is_multimodal=data_config.is_multimodal,
        image_folder=data_config.image_folder,
        use_start_end_tokens=trainer_config.llm.use_start_end_tokens,
        use_image_patch_token=trainer_config.llm.use_image_patch_token,
        image_token=trainer_config.llm.image_token,
        image_start_token=trainer_config.llm.image_start_token,
        image_end_token=trainer_config.llm.image_end_token,
        image_patch_token=trainer_config.llm.image_patch_token,
        ignore_index=trainer_config.llm.ignore_index,
        image_token_index=trainer_config.llm.image_token_index,
        image_preprocessor=image_processor,
    )
