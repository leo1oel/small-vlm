from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch.nn.parameter import Parameter
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from ..config import TrainerConfig


def get_optimizer(
    trainer_config: TrainerConfig, param_groups: dict[str, dict[str, list[Parameter]]]
) -> OptimizerLRScheduler:
    optimizer_grouped_parameters = []

    if "visual_encoder" in param_groups and trainer_config.unfreeze.train_visual_encoder:
        optimizer_grouped_parameters.append(
            {
                "params": param_groups["visual_encoder"]["decay"],
                "weight_decay": trainer_config.weight_decay.visual_encoder_weight_decay,
                "lr": trainer_config.learning_rate.visual_encoder_learning_rate,
            }
        )

        optimizer_grouped_parameters.append(
            {
                "params": param_groups["visual_encoder"]["no_decay"],
                "weight_decay": 0.0,
                "lr": trainer_config.learning_rate.visual_encoder_learning_rate,
            }
        )

    if "language_model" in param_groups and trainer_config.unfreeze.train_language_model:
        optimizer_grouped_parameters.append(
            {
                "params": param_groups["language_model"]["decay"],
                "weight_decay": trainer_config.weight_decay.language_model_weight_decay,
                "lr": trainer_config.learning_rate.language_model_learning_rate,
            }
        )

        optimizer_grouped_parameters.append(
            {
                "params": param_groups["language_model"]["no_decay"],
                "weight_decay": 0.0,
                "lr": trainer_config.learning_rate.language_model_learning_rate,
            }
        )

    if "connector" in param_groups and trainer_config.unfreeze.train_connector:
        optimizer_grouped_parameters.append(
            {
                "params": param_groups["connector"]["decay"],
                "weight_decay": trainer_config.weight_decay.connector_weight_decay,
                "lr": trainer_config.learning_rate.connector_learning_rate,
            }
        )

        optimizer_grouped_parameters.append(
            {
                "params": param_groups["connector"]["no_decay"],
                "weight_decay": 0.0,
                "lr": trainer_config.learning_rate.connector_learning_rate,
            }
        )

    optimizer = AdamW(
        optimizer_grouped_parameters,  # pyright: ignore
        lr=trainer_config.learning_rate.default_lr,
        betas=(trainer_config.optimizer.adam_beta1, trainer_config.optimizer.adam_beta2),
        eps=trainer_config.optimizer.adam_epsilon,
    )

    total_steps: int = (
        trainer_config.num_training_samples
        * trainer_config.max_epochs
        // (trainer_config.batch_size * trainer_config.accumulate_grad_batches)
    )

    if (trainer_config.num_training_samples * trainer_config.max_epochs) % (
        trainer_config.batch_size * trainer_config.accumulate_grad_batches
    ) != 0:
        total_steps += 1

    warmup_steps: int = int(total_steps * trainer_config.scheduler.warmup_ratio)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=trainer_config.scheduler.warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=trainer_config.scheduler.min_lr_ratio * trainer_config.learning_rate.default_lr,
    )

    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps]
    )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor": "val_loss",
            "name": "learning_rate",
        },
    }
