import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from ..config.config_schema import TrainerConfig
from ..models.model import VLM


def train(
    config: TrainerConfig,
    model: VLM,
    train_dataloader: DataLoader[dict[str, torch.Tensor]],
    val_dataloader: DataLoader[dict[str, torch.Tensor]],
) -> None:  # pyright: ignore
    debug: bool = config.debug
    if debug:
        trainer: pl.Trainer = pl.Trainer(
            default_root_dir=config.default_root_dir,
            fast_dev_run=True,
            limit_train_batches=0.1,
            limit_val_batches=0.01,
            num_sanity_val_steps=2,
            profiler="simple",
        )
    else:
        trainer = pl.Trainer(default_root_dir=config.default_root_dir)
        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=config.default_root_dir + "/last.ckpt",
        )
