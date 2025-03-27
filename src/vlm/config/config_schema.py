from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


@dataclass
class VisualEncoderConfig:
    name: str
    hf_name: str
    type: str
    hidden_size: int
    img_size: int
    patch_size: int
    output_layer: int


@dataclass
class LLMConfig:
    name: str
    hf_name: str
    type: str
    hidden_size: int
    vocab_size: int
    max_seq_length: int
    image_token: str
    pad_token: str


@dataclass
class ConnectorConfig:
    name: str
    type: str


@dataclass
class ModelConfig:
    name: str
    visual_encoder: VisualEncoderConfig
    llm: LLMConfig
    connector: ConnectorConfig


@dataclass
class DatasetConfig:
    name: str
    hf_name: str
    type: str
    batch_size: int


@dataclass
class UnfreezeConfig:
    train_visual_encoder: bool
    train_language_model: bool
    train_connector: bool


@dataclass
class LearningRateConfig:
    visual_encoder_learning_rate: float
    language_model_learning_rate: float
    connector_learning_rate: float
    default_lr: float


@dataclass
class WeightDecayConfig:
    visual_encoder_weight_decay: float
    language_model_weight_decay: float
    connector_weight_decay: float


@dataclass
class SchedulerConfig:
    warmup_ratio: float
    warmup_start_factor: float
    min_lr_ratio: float


@dataclass
class OptimizerConfig:
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float


@dataclass
class TrainerConfig:
    name: str
    unfreeze: UnfreezeConfig
    learning_rate: LearningRateConfig
    weight_decay: WeightDecayConfig
    scheduler: SchedulerConfig
    optimizer: OptimizerConfig
    num_training_samples: int
    batch_size: int
    ignore_index: int = -100
    default_root_dir: str = "./checkpoints"
    debug: bool = False
    experiment_name: str = "vlm_training"
    max_epochs: int = 30
    save_top_k: int = 3
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"
    early_stopping: bool = False
    patience: int = 5
    log_every_n_steps: int = 50
    val_check_interval: float = 0.5
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    precision: str = "16-mixed"
    accelerator: str = "gpu"
    devices: int | list[int] = 1
    strategy: str | None = None
    resume_from_checkpoint: bool = True
    checkpoint_path: str | None = None
    wandb_project_name: str = "vlm-training"
    log_model_to_wandb: bool = False


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
