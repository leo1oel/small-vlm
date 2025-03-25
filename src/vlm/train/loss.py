import torch

from ..config.config_schema import TrainerConfig


def get_loss(
    trainer_config: TrainerConfig, outputs: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    pass
