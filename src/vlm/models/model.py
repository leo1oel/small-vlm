import logging
from typing import override

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from ..config.config_schema import (
    ConnectorConfig,
    LLMConfig,
    ModelConfig,
    TrainerConfig,
    VisualEncoderConfig,
)
from ..train import get_loss, get_optimizer
from .connectors import Connector
from .language_models import LanguageModel
from .visual_encoders import VisualEncoder

log: logging.Logger = logging.getLogger(name=__name__)


class VLM(pl.LightningModule):
    def __init__(self, model_config: ModelConfig, trainer_config: TrainerConfig) -> None:
        super().__init__()
        self.model_config: ModelConfig = model_config
        self.trainer_config: TrainerConfig = trainer_config
        self.visual_encoder: VisualEncoder = self._build_visual_encoder()
        self.language_model: LanguageModel = self._build_language_model()
        self.connector: Connector = self._build_connector()

        self.save_hyperparameters(model_config, trainer_config)

    def _build_visual_encoder(self) -> VisualEncoder:
        encoder_config: VisualEncoderConfig = self.model_config.visual_encoder
        if encoder_config.type == "vit":
            from .visual_encoders import ViTEncoder

            return ViTEncoder(encoder_config)
        else:
            log.error(f"[bold red]Unknown visual encoder type: {encoder_config.type}[/bold red]")
            raise ValueError(f"Unknown visual encoder type: {encoder_config.type}")

    def _build_language_model(self) -> LanguageModel:
        llm_config: LLMConfig = self.model_config.llm
        if llm_config.type == "hf_llm":
            from .language_models import HFLLMLanguageModel

            return HFLLMLanguageModel(llm_config)
        else:
            log.error(f"[bold red]Unknown language model type: {llm_config.type}[/bold red]")
            raise ValueError(f"Unknown language model type: {llm_config.type}")

    def _build_connector(self) -> Connector:
        connector_config: ConnectorConfig = self.model_config.connector
        if connector_config.type == "linear":
            from .connectors import LinearConnector

            return LinearConnector(connector_config)
        else:
            log.error(f"[bold red]Unknown connector type: {connector_config.type}[/bold red]")
            raise ValueError(f"Unknown connector type: {connector_config.type}")

    @override
    def forward(self, images: torch.Tensor, text: None | torch.Tensor = None) -> torch.Tensor:
        vision_features: torch.Tensor = self.visual_encoder(images)

        multimodal_features: torch.Tensor = self.connector(vision_features, text)
        outputs: torch.Tensor = self.language_model(multimodal_features)

        return outputs

    @override
    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images: torch.Tensor = batch["images"]
        text: None | torch.Tensor = batch.get("text")
        labels: torch.Tensor = batch["labels"]

        outputs: torch.Tensor = self(images, text)
        loss: torch.Tensor = self._calculate_loss(outputs, labels)

        self.log("train_loss", loss, prog_bar=True)  # pyright: ignore[reportUnknownMemberType]
        return loss

    @override
    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images: torch.Tensor = batch["images"]
        text: None | torch.Tensor = batch.get("text")
        labels: torch.Tensor = batch["labels"]

        outputs: torch.Tensor = self(images, text)
        loss: torch.Tensor = self._calculate_loss(outputs, labels)

        self.log("val_loss", loss, prog_bar=True)  # pyright: ignore[reportUnknownMemberType]
        return loss

    @override
    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images: torch.Tensor = batch["images"]
        text: None | torch.Tensor = batch.get("text")
        labels: torch.Tensor = batch["labels"]

        outputs: torch.Tensor = self(images, text)
        loss: torch.Tensor = self._calculate_loss(outputs, labels)
        return loss

    def _calculate_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return get_loss(self.trainer_config, outputs, labels)

    @override
    def configure_optimizers(self) -> OptimizerLRScheduler:
        return get_optimizer(self.trainer_config)
