import logging
from pathlib import Path
from typing import override

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from vlm.models.model import VLM

from ..config.config_schema import InferenceConfig

log: logging.Logger = logging.getLogger(name=__name__)


class SimpleDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(self, data: list[dict[str, torch.Tensor | str]]) -> None:
        self.data: list[dict[str, torch.Tensor | str]] = data

    def __len__(self) -> int:
        return len(self.data)

    @override
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        return self.data[idx]


def inference(
    config: InferenceConfig, inference_dataloader: DataLoader[dict[str, torch.Tensor]] | None = None
) -> None:  # pyright: ignore
    log.info(f"[bold green]Loading model from checkpoint:[/bold green] {config.checkpoint_path}")
    model: VLM = VLM.load_from_checkpoint(Path(config.checkpoint_path), map_location="cuda")
    trainer: pl.Trainer = pl.Trainer(devices=1)
    if inference_dataloader is not None:
        log.info(f"[bold green]Predicting on dataloader:[/bold green] {len(inference_dataloader)}")
        trainer.predict(model=model, dataloaders=inference_dataloader)  # pyright: ignore
    else:
        log.info("[bold green]Predicting on given data[/bold green]")
        image_paths: list[str] = config.image_path if config.image_path is not None else []
        texts: list[str] = config.text if config.text is not None else []
        if len(texts) == 0 and len(image_paths) == 0:
            log.error("[bold red]image_paths or texts must be provided[/bold red]")
            raise ValueError("image_paths or texts must be provided")
        data: list[dict[str, torch.Tensor | str]] = []
        for idx, image_path in enumerate(image_paths):
            image: Image.Image = Image.open(image_path).convert("RGB")
            image_tensor: torch.Tensor = model.visual_encoder.preprocessor(  # pyright: ignore
                image, return_tensors="pt"
            )["pixel_values"].squeeze(0)
            data.append({"image": image_tensor, "text": texts[idx]})
        dataset: SimpleDataset = SimpleDataset(data)
        dataloader: DataLoader[dict[str, torch.Tensor | str]] = DataLoader(
            dataset, batch_size=config.batch_size, num_workers=config.num_workers
        )  # pyright: ignore
        predictions: list[str] = trainer.predict(model=model, dataloaders=dataloader)  # pyright: ignore
        for idx, prediction in enumerate(predictions):
            log.info(f"[bold green]Prediction {idx}:[/bold green] {prediction}")
