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
        self.example_input_array: tuple[torch.Tensor | list[torch.Tensor], torch.Tensor] = (
            [torch.randn(1, 3, 224, 224), torch.randn(3, 3, 224, 224)],  # 图像输入
            self.language_model.tokenizer(
                ["test <|image|>.", "test <|image|> multiple <|image|> images <|image|>."],
                padding=True,
                return_tensors="pt",
            ).input_ids,  # pyright: ignore
        )

        self.save_hyperparameters(model_config, trainer_config)

    def _build_visual_encoder(self) -> VisualEncoder:
        encoder_config: VisualEncoderConfig = self.model_config.visual_encoder
        if encoder_config.type == "vit":
            from .visual_encoders import ViTEncoder

            log.info("[bold green]Building ViTEncoder[/bold green]")
            return ViTEncoder(encoder_config)
        else:
            log.error(f"[bold red]Unknown visual encoder type: {encoder_config.type}[/bold red]")
            raise ValueError(f"Unknown visual encoder type: {encoder_config.type}")

    def _build_language_model(self) -> LanguageModel:
        llm_config: LLMConfig = self.model_config.llm
        if llm_config.type == "hf_llm":
            from .language_models import HFLLMLanguageModel

            log.info("[bold green]Building HFLLMLanguageModel[/bold green]")
            return HFLLMLanguageModel(llm_config)
        else:
            log.error(f"[bold red]Unknown language model type: {llm_config.type}[/bold red]")
            raise ValueError(f"Unknown language model type: {llm_config.type}")

    def _build_connector(self) -> Connector:
        connector_config: ConnectorConfig = self.model_config.connector
        if connector_config.type == "linear":
            from .connectors import LinearConnector

            log.info("[bold green]Building LinearConnector[/bold green]")
            return LinearConnector(connector_config)
        else:
            log.error(f"[bold red]Unknown connector type: {connector_config.type}[/bold red]")
            raise ValueError(f"Unknown connector type: {connector_config.type}")

    @override
    def forward(
        self, images: torch.Tensor | list[torch.Tensor], texts: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(images, list):
            image_counts: list[int] = [tensor.shape[0] for tensor in images]
            all_images: torch.Tensor = torch.cat(images, dim=0)  # [total_images, C, H, W]
            all_features: torch.Tensor = self.visual_encoder(all_images)
            vision_features: tuple[torch.Tensor, ...] = torch.split(
                all_features, image_counts, dim=0
            )
        else:
            vision_features = (self.visual_encoder(images),)

        llm: LanguageModel = self.language_model
        connector_output: tuple[torch.Tensor, torch.Tensor] = self.connector(
            vision_features, texts, llm.embeddings, llm.image_token_id
        )
        multimodal_features: torch.Tensor = connector_output[0]
        attention_mask: torch.Tensor = connector_output[1]
        log.debug(f"[bold yellow]multimodal_features: {multimodal_features.shape}[/bold yellow]")
        log.debug(f"[bold yellow]attention_mask: {attention_mask.shape}[/bold yellow]")
        outputs: torch.Tensor = llm(input_embeds=multimodal_features, attention_mask=attention_mask)
        log.debug(f"[bold yellow]outputs: {outputs.shape}[/bold yellow]")
        return outputs

    @override
    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images: torch.Tensor | list[torch.Tensor] = batch["images"]
        text: torch.Tensor = batch["text"]
        labels: torch.Tensor = batch["labels"]

        outputs: torch.Tensor = self(images, text)
        loss: torch.Tensor = self._calculate_loss(outputs, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    @override
    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images: torch.Tensor | list[torch.Tensor] = batch["images"]
        text: torch.Tensor = batch["text"]
        labels: torch.Tensor = batch["labels"]

        outputs: torch.Tensor = self(images, text)
        loss: torch.Tensor = self._calculate_loss(outputs, labels)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    @override
    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images: torch.Tensor | list[torch.Tensor] = batch["images"]
        text: torch.Tensor = batch["text"]
        labels: torch.Tensor = batch["labels"]

        outputs: torch.Tensor = self(images, text)
        loss: torch.Tensor = self._calculate_loss(outputs, labels)

        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _calculate_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return get_loss(self.trainer_config, outputs, labels)

    @override
    def configure_optimizers(self) -> OptimizerLRScheduler:
        return get_optimizer(self.trainer_config)
