import logging

import torch

from ..config.config_schema import TrainerConfig

log = logging.getLogger(__name__)


def get_loss(
    trainer_config: TrainerConfig, outputs: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    ignore_index: int = (
        trainer_config.ignore_index if hasattr(trainer_config, "ignore_index") else -100
    )
    if outputs.dim() > 2:
        # [batch_size, seq_len, vocab_size] -> [batch_size*seq_len, vocab_size]
        outputs = outputs.reshape(-1, outputs.size(-1))

    if labels.dim() > 1:
        # [batch_size, seq_len] -> [batch_size*seq_len]
        labels = labels.reshape(-1)
    loss: torch.Tensor = loss_fct(outputs, labels, ignore_index)
    return loss


def loss_fct(outputs: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    return torch.nn.CrossEntropyLoss(ignore_index=ignore_index)(outputs, labels)  # pyright: ignore
