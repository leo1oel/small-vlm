from pytorch_lightning.utilities.types import OptimizerLRScheduler

from ..config.config_schema import TrainerConfig


def get_optimizer(trainer_config: TrainerConfig) -> OptimizerLRScheduler:  # pyright: ignore
    pass
