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
    image_preprocessor: BaseImageProcessor | None = field(
        default=None, metadata={"help": "Image preprocessor for the visual encoder."}
    )
    image_aspect_ratio: str = "square"


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
        image_preprocessor=image_processor,
        image_aspect_ratio=data_config.image_aspect_ratio,
    )
