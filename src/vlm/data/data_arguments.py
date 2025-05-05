from dataclasses import dataclass, field

from ..config.config_schema import DatasetConfig


@dataclass
class DataArguments:
    data_path: str | None = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = True
    is_multimodal: bool = True
    image_folder: str | None = field(default=None)
    image_aspect_ratio: str = "square"


def get_data_args(config: DatasetConfig) -> DataArguments:
    return DataArguments(
        data_path=config.path,
        lazy_preprocess=config.lazy_preprocess,
        is_multimodal=config.is_multimodal,
        image_folder=config.image_folder,
        image_aspect_ratio=config.image_aspect_ratio,
    )
