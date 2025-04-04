import logging
from typing import Any, cast

import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch.nn.parameter import Parameter
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from ..config import TrainerConfig

log = logging.getLogger(__name__)


def get_optimizer(
    trainer_config: TrainerConfig,
    param_groups: dict[str, dict[str, list[Parameter]]],
) -> OptimizerLRScheduler:
    optimizer_grouped_parameters = _build_optimizer_params(trainer_config, param_groups)

    optimizer = _create_optimizer(trainer_config, optimizer_grouped_parameters)

    scheduler = _create_scheduler(trainer_config, optimizer)

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "name": "learning_rate",
        },
    }


def _build_optimizer_params(
    config: TrainerConfig,
    param_groups: dict[str, dict[str, list[Parameter]]],
) -> list[dict[str, Any]]:
    optimizer_params: list[dict[str, Any]] = []

    optimizer_params.extend(
        _get_module_param_groups(
            module_name="visual_encoder",
            param_groups=param_groups,
            weight_decay=config.weight_decay.visual_encoder_weight_decay,
            learning_rate=config.learning_rate.visual_encoder_learning_rate,
        )
    )

    optimizer_params.extend(
        _get_module_param_groups(
            module_name="language_model",
            param_groups=param_groups,
            weight_decay=config.weight_decay.language_model_weight_decay,
            learning_rate=config.learning_rate.language_model_learning_rate,
        )
    )

    optimizer_params.extend(
        _get_module_param_groups(
            module_name="connector",
            param_groups=param_groups,
            weight_decay=config.weight_decay.connector_weight_decay,
            learning_rate=config.learning_rate.connector_learning_rate,
        )
    )

    return optimizer_params


def _get_module_param_groups(
    module_name: str,
    param_groups: dict[str, dict[str, list[Parameter]]],
    weight_decay: float,
    learning_rate: float,
) -> list[dict[str, Any]]:
    return [
        {
            "params": param_groups[module_name]["decay"],
            "weight_decay": weight_decay,
            "lr": learning_rate,
        },
        {
            "params": param_groups[module_name]["no_decay"],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
    ]


def _create_optimizer(config: TrainerConfig, optimizer_params: list[dict[str, Any]]) -> AdamW:
    return AdamW(
        optimizer_params,
        lr=config.learning_rate.default_lr,
        betas=(config.optimizer.adam_beta1, config.optimizer.adam_beta2),
        eps=config.optimizer.adam_epsilon,
    )


def _create_scheduler(config: TrainerConfig, optimizer: AdamW) -> SequentialLR:
    total_steps = _calculate_total_steps(config)

    warmup_steps = int(total_steps * config.scheduler.warmup_ratio)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=config.scheduler.warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=config.scheduler.min_lr_ratio * config.learning_rate.default_lr,
    )

    return SequentialLR(
        optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps]
    )


def _calculate_total_steps(config: TrainerConfig) -> int:
    num_devices: int = 1
    devices: int | str = config.devices
    if devices == "auto" or devices == -1:
        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
    elif isinstance(devices, int):
        num_devices = devices
    else:
        log.error(f"Invalid devices: {devices}")
        raise ValueError(f"Invalid devices: {devices}")

    samples_per_step: int = config.batch_size * config.accumulate_grad_batches * num_devices

    total_samples = cast(int, config.num_training_samples) * config.max_epochs

    total_steps = total_samples // samples_per_step

    if total_samples % samples_per_step != 0:
        total_steps += 1

    log.info(f"Total steps: {total_steps} for {total_samples} total samples")

    return total_steps
