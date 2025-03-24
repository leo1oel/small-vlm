from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


@dataclass
class VisualEncoderConfig:
    name: str
    hf_name: str


@dataclass
class LLMConfig:
    name: str
    hf_name: str


@dataclass
class ConnectorConfig:
    name: str


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
