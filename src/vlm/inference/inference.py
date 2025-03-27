import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from ..config.config_schema import TrainerConfig
from ..models.model import VLM

wandb_logger = WandbLogger(project="small-vlm", log_model="all")


def inference(
    cfg: TrainerConfig, inference_dataloader: DataLoader[dict[str, torch.Tensor]]
) -> None:  # pyright: ignore
    model: VLM = VLM.load_from_checkpoint(cfg.default_root_dir + "/last.ckpt")
    model.eval()
    trainer: Trainer = Trainer(logger=wandb_logger)
    trainer.predict(model=model, dataloaders=inference_dataloader)  # pyright: ignore
