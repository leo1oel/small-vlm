from logging import getLogger
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from ..config import TrainerConfig
from ..models import VLM

torch.set_float32_matmul_precision("high")

log = getLogger(__name__)


def train(
    config: TrainerConfig,
    model: VLM,
    train_dataloader: DataLoader[dict[str, torch.Tensor]],
    val_dataloader: DataLoader[dict[str, torch.Tensor]] | None = None,
    test_dataloader: DataLoader[dict[str, torch.Tensor]] | None = None,
) -> str:
    # Setup Wandb logger
    wandb_logger: WandbLogger = WandbLogger(
        name=config.experiment_name,
        project=config.wandb_project_name,
        log_model="all" if config.log_model_to_wandb else False,
    )
    wandb_logger.watch(model)

    callbacks = []

    # Checkpoint callback
    has_val_dataloader = val_dataloader is not None

    monitor_metric = config.monitor_metric
    if not has_val_dataloader and "val_" in monitor_metric:
        monitor_metric = monitor_metric.replace("val_", "train_")
        log.warning(
            f"[bold yellow]No validation dataloader provided. Falling back to monitor {monitor_metric}[/bold yellow]"
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(config.default_root_dir) / "checkpoints",
        monitor=monitor_metric,
        mode=config.monitor_mode,
        save_last=True,
        every_n_epochs=config.save_every_n_epochs,
        every_n_train_steps=config.save_every_n_train_steps,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(
        logging_interval="step", log_momentum=True, log_weight_decay=True
    )
    callbacks.append(lr_monitor)

    # Early stopping (optional)
    if config.early_stopping:
        early_stopping = EarlyStopping(
            monitor=config.monitor_metric,
            patience=config.patience,
            mode=config.monitor_mode,
            verbose=True,
        )
        callbacks.append(early_stopping)

    # Configure trainer
    trainer_kwargs = {  # pyright: ignore
        "default_root_dir": config.default_root_dir,
        "callbacks": callbacks,
        "logger": wandb_logger,
        "max_epochs": config.max_epochs,
        "log_every_n_steps": config.log_every_n_steps,
        "val_check_interval": config.val_check_interval,
        "gradient_clip_val": config.gradient_clip_val,
        "accumulate_grad_batches": config.accumulate_grad_batches,
        "accelerator": config.accelerator,
        "devices": config.devices,
        "strategy": config.strategy,
        "precision": config.precision,
        "deterministic": True,
    }

    if config.debug:
        trainer_kwargs.update(
            {
                "fast_dev_run": True,
                "profiler": "advanced",
                "overfit_batches": 0.01,
                "detect_anomaly": True,
            }
        )

    trainer = pl.Trainer(**trainer_kwargs)  # pyright: ignore

    trainable_config: dict[str, bool] = {
        "visual_encoder": config.unfreeze.train_visual_encoder,
        "language_model": config.unfreeze.train_language_model,
        "connector": config.unfreeze.train_connector,
    }
    model.set_trainable_params(trainable_config)

    ckpt_path = None
    if config.resume_from_checkpoint:
        if (Path(config.default_root_dir) / "checkpoints" / "last.ckpt").exists():
            ckpt_path = Path(config.default_root_dir) / "checkpoints" / "last.ckpt"
        elif hasattr(config, "checkpoint_path") and config.checkpoint_path:
            ckpt_path = config.checkpoint_path

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path,
    )

    if test_dataloader is not None:
        trainer.test(model=model, dataloaders=test_dataloader)

    wandb_logger.experiment.finish()

    log.info(
        f"[bold green]Best model checkpoint:[/bold green] {checkpoint_callback.best_model_path}"
    )
    best_score = checkpoint_callback.best_model_score
    if best_score is not None:
        log.info(f"[bold green]Best validation score:[/bold green] {best_score:.4f}")
    else:
        log.info("[bold red]Best validation score is not available.[/bold red]")

    return checkpoint_callback.best_model_path  # pyright: ignore
