from dataclasses import dataclass


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
    cs = ConfigStore.instance()
    # 注册主配置
    cs.store(name="config", node=AppConfig)