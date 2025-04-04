from logging import getLogger
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import FSDPStrategy
from torch import nn
from torch.utils.data import DataLoader

from ..config import TrainerConfig
from ..models import VLM

torch.set_float32_matmul_precision("high")

log = getLogger(__name__)


def train(
    config: TrainerConfig,
    model: VLM,
    train_dataloader: DataLoader[Any],
    val_dataloader: DataLoader[Any] | None = None,
    test_dataloader: DataLoader[Any] | None = None,
) -> str:
    _setup_trainable_params(model, config)

    wandb_logger = _setup_wandb_logger(config, model)

    callbacks = _setup_callbacks(config, val_dataloader is not None)

    trainer = _setup_trainer(config, callbacks, wandb_logger)

    ckpt_path = _find_checkpoint_path(config)

    trainer.fit(
        model=model,
        ckpt_path=ckpt_path,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    if test_dataloader is not None:
        trainer.test(model=model, dataloaders=test_dataloader)

    wandb_logger.experiment.finish()

    best_model_path = _log_best_model_info(callbacks[0])

    return best_model_path


def _setup_wandb_logger(config: TrainerConfig, model: VLM) -> WandbLogger:
    logger = WandbLogger(
        name=config.experiment_name,
        project=config.wandb_project_name,
        log_model="all" if config.log_model_to_wandb else False,
    )

    # log model gradients and graph
    logger.watch(model)
    return logger


def _setup_callbacks(config: TrainerConfig, has_val_dataloader: bool) -> list[Any]:
    callbacks: list[Any] = []

    monitor_metric = config.monitor_metric
    if not has_val_dataloader and "val_" in monitor_metric:
        monitor_metric = monitor_metric.replace("val_", "train_")
        log.warning(f"No validation dataloader provided. Falling back to monitor {monitor_metric}")

    checkpoint_dir = Path(config.default_root_dir) / "checkpoints"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor=monitor_metric,
        mode=config.monitor_mode,
        save_last=True,
        every_n_epochs=config.save_every_n_epochs,
        every_n_train_steps=config.save_every_n_train_steps,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    lr_monitor = LearningRateMonitor(
        logging_interval="step", log_momentum=True, log_weight_decay=True
    )
    callbacks.append(lr_monitor)

    callbacks.append(RichProgressBar())
    callbacks.append(RichModelSummary())

    if config.early_stopping:
        early_stopping = EarlyStopping(
            monitor=config.monitor_metric,
            patience=config.patience,
            mode=config.monitor_mode,
            verbose=True,
        )
        callbacks.append(early_stopping)

    return callbacks


def _setup_trainer(config: TrainerConfig, callbacks: list[Any], logger: WandbLogger) -> pl.Trainer:
    trainer_kwargs: dict[str, Any] = {
        "default_root_dir": config.default_root_dir,
        "callbacks": callbacks,
        "logger": logger,
        "max_epochs": config.max_epochs,
        "log_every_n_steps": config.log_every_n_steps,
        "val_check_interval": config.val_check_interval,
        "gradient_clip_val": config.gradient_clip_val,
        "accumulate_grad_batches": config.accumulate_grad_batches,
        "accelerator": config.accelerator,
        "devices": config.devices,
        "precision": config.precision,
        "deterministic": True,
    }

    if config.strategy == "fsdp":
        trainer_kwargs["strategy"] = FSDPStrategy(
            # Enable activation checkpointing on these layers
            activation_checkpointing_policy={
                nn.TransformerEncoderLayer,
                nn.TransformerDecoderLayer,
                nn.MultiheadAttention,
            },
        )
    else:
        trainer_kwargs["strategy"] = config.strategy

    if config.debug:
        debug_kwargs = {
            "fast_dev_run": True,
            "profiler": "advanced",
            "overfit_batches": 0.01,
            "detect_anomaly": True,
        }
        trainer_kwargs.update(debug_kwargs)

    return pl.Trainer(**trainer_kwargs)  # pyright: ignore


def _setup_trainable_params(model: VLM, config: TrainerConfig) -> None:
    trainable_config = {
        "visual_encoder": config.unfreeze.train_visual_encoder,
        "language_model": config.unfreeze.train_language_model,
        "connector": config.unfreeze.train_connector,
    }
    model.set_trainable_params(trainable_config)


def _find_checkpoint_path(config: TrainerConfig) -> str | None:
    if not config.resume_from_checkpoint:
        return None

    if hasattr(config, "checkpoint_path") and config.checkpoint_path:
        return config.checkpoint_path

    last_ckpt_path = Path(config.default_root_dir) / "checkpoints" / "last.ckpt"
    if last_ckpt_path.exists():
        return str(last_ckpt_path)

    return None


def _log_best_model_info(checkpoint_callback: ModelCheckpoint) -> str:
    best_model_path: Any = checkpoint_callback.best_model_path
    log.info(f"Best model checkpoint: {best_model_path}")

    best_score = checkpoint_callback.best_model_score
    if best_score is not None:
        log.info(f"Best model score: {best_score:.4f}")
    else:
        log.info("Best model score is not available.")

    return best_model_path
