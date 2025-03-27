import json
import logging
from collections.abc import Callable
from typing import override

import pytorch_lightning as pl
import torch
from PIL import Image
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch.nn.parameter import Parameter

from ..config import (
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
        self.image_hidden_size: int = self.visual_encoder.hidden_size  # pyright: ignore
        self.text_hidden_size: int = self.language_model.hidden_size  # pyright: ignore
        self.connector: Connector = self._build_connector()
        self.transform: Callable[
            [dict[str, Image.Image | list[dict[str, str]]]],
            dict[str, torch.Tensor | list[torch.Tensor]],
        ] = self._build_transform()
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
            return LinearConnector(connector_config, self.image_hidden_size, self.text_hidden_size)
        elif connector_config.type == "mlp":
            from .connectors import MLPConnector

            log.info("[bold green]Building MLPConnector[/bold green]")
            return MLPConnector(connector_config, self.image_hidden_size, self.text_hidden_size)
        else:
            log.error(f"[bold red]Unknown connector type: {connector_config.type}[/bold red]")
            raise ValueError(f"Unknown connector type: {connector_config.type}")

    def _build_transform(
        self,
    ) -> Callable[
        [dict[str, Image.Image | list[dict[str, str]]]],
        dict[str, torch.Tensor | list[torch.Tensor]],
    ]:
        def transform(
            item: dict[str, Image.Image | list[dict[str, str]]],
        ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
            if "image" in item:
                original_image: Image.Image = item["image"].convert("RGB")  # pyright: ignore
            else:
                log.error(f"Cannot find image in item {item}")
                raise ValueError(f"Cannot find image in item {item}")
            if "text" in item:
                text_str: str = item["text"]  # pyright: ignore
            elif "conversations" in item:
                text_str: str = item["conversations"]  # pyright: ignore
            else:
                log.error(f"Cannot find text in item {item}")
                raise ValueError(f"Cannot find text in item {item}")
            input_image = self.visual_encoder.preprocessor(original_image, return_tensors="pt")  # pyright: ignore
            image_tensor: torch.Tensor = input_image["pixel_values"]  # pyright: ignore
            text: list[dict[str, str]] = json.loads(text_str.replace("\n", "\\n"))
            text_and_label: tuple[torch.Tensor, torch.Tensor] = self.language_model.transform(
                text, self.visual_encoder.token_size
            )
            return {"image": image_tensor, "text": text_and_label[0], "label": text_and_label[1]}  # pyright: ignore

        return transform

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
            vision_features, texts, llm.embeddings, llm.image_token_id, llm.pad_token_id
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
        no_decay: list[str] = ["bias", "LayerNorm.weight", "layer_norm.weight", "ln_"]
        param_groups: dict[str, dict[str, list[Parameter]]] = {}

        if self.trainer_config.unfreeze.train_visual_encoder:
            decay_params: list[Parameter] = [
                p
                for n, p in self.visual_encoder.named_parameters()
                if p.requires_grad and not any(no_decay_name in n for no_decay_name in no_decay)
            ]

            no_decay_params: list[Parameter] = [
                p
                for n, p in self.visual_encoder.named_parameters()
                if p.requires_grad and any(no_decay_name in n for no_decay_name in no_decay)
            ]

            if decay_params or no_decay_params:
                param_groups["visual_encoder"] = {
                    "decay": decay_params,
                    "no_decay": no_decay_params,
                }

        if self.trainer_config.unfreeze.train_language_model:
            decay_params = [
                p
                for n, p in self.language_model.named_parameters()
                if p.requires_grad and not any(no_decay_name in n for no_decay_name in no_decay)
            ]

            no_decay_params = [
                p
                for n, p in self.language_model.named_parameters()
                if p.requires_grad and any(no_decay_name in n for no_decay_name in no_decay)
            ]

            if decay_params or no_decay_params:
                param_groups["language_model"] = {
                    "decay": decay_params,
                    "no_decay": no_decay_params,
                }

        if self.trainer_config.unfreeze.train_connector:
            decay_params = [
                p
                for n, p in self.connector.named_parameters()
                if p.requires_grad and not any(no_decay_name in n for no_decay_name in no_decay)
            ]

            no_decay_params = [
                p
                for n, p in self.connector.named_parameters()
                if p.requires_grad and any(no_decay_name in n for no_decay_name in no_decay)
            ]

            if decay_params or no_decay_params:
                param_groups["connector"] = {"decay": decay_params, "no_decay": no_decay_params}

        return get_optimizer(self.trainer_config, param_groups)

    def freeze_visual_encoder(self, freeze: bool = True) -> None:
        for param in self.visual_encoder.parameters():
            param.requires_grad = not freeze

    def freeze_language_model(self, freeze: bool = True, except_layer_norm: bool = True) -> None:
        for name, param in self.language_model.named_parameters():
            if except_layer_norm and (
                "layernorm" in name.lower()
                or "layer_norm" in name.lower()
                or "ln_" in name.lower()
                or "embedding" in name.lower()
            ):
                param.requires_grad = True
            else:
                param.requires_grad = not freeze

    def freeze_connector(self, freeze: bool = True) -> None:
        for param in self.connector.parameters():
            param.requires_grad = not freeze

    def set_trainable_params(self, config: dict[str, bool]) -> None:
        if "visual_encoder" in config:
            self.freeze_visual_encoder(not config["visual_encoder"])

        if "language_model" in config:
            self.freeze_language_model(not config["language_model"])

        if "connector" in config:
            self.freeze_connector(not config["connector"])

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        log.debug(
            f"[bold yellow]Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%} of total)[/bold yellow]"
        )

        for module_name, module in [
            ("visual_encoder", self.visual_encoder),
            ("language_model", self.language_model),
            ("connector", self.connector),
        ]:
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total = sum(p.numel() for p in module.parameters())
            if total > 0:
                log.debug(
                    f"[bold yellow]  - {module_name}: {trainable:,} ({trainable / total:.2%} of {total:,})[/bold yellow]"
                )
