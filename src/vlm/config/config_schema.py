from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


@dataclass
class VisualEncoderConfig:
    name: str
    hf_name: str
    type: str
    feature_dim: int
    img_size: int
    patch_size: int
    output_layer: int


@dataclass
class LLMConfig:
    name: str
    hf_name: str
    type: str


@dataclass
class ConnectorConfig:
    name: str
    type: str
    visual_dim: int
    text_dim: int


@dataclass
class ModelConfig:
    name: str
    visual_encoder: VisualEncoderConfig
    llm: LLMConfig
    connector: ConnectorConfig


@dataclass
class DatasetConfig:
    name: str


@dataclass
class TrainerConfig:
    name: str
    ignore_index: int = -100


@dataclass
class ModeConfig:
    is_training: bool


@dataclass
class AppConfig:
    mode: ModeConfig
    model: ModelConfig
    dataset: DatasetConfig
    trainer: TrainerConfig


def register_configs() -> None:
    cs: ConfigStore = ConfigStore.instance()
    cs.store(name="cfg", node=AppConfig)
