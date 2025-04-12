import logging
from typing import Any, override

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
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
    def __init__(
        self,
        model_config: ModelConfig,
        trainer_config: TrainerConfig,
        lazy_loading: bool = False,
    ) -> None:
        super().__init__()
        # process config
        self.model_config: ModelConfig = self._process_config(model_config)
        self.trainer_config: TrainerConfig = self._process_config(trainer_config)

        # initialize components
        self._initialize_components()
        if not lazy_loading:
            self.initialize_components()

    def _process_config(self, config: Any) -> Any:
        if isinstance(config, dict):
            return OmegaConf.create(config)  # pyright: ignore
        return config

    # initialize all components
    def initialize_components(self) -> None:
        self.visual_encoder.initialize_components()
        self.language_model.initialize_components()
        self._connector = self._build_connector()
        # setup example input
        self._setup_example_input(self.visual_encoder.img_size)

        self._save_hyperparameters()

    # only initialize components needed for dataset processing
    def _initialize_components(self) -> None:
        self._visual_encoder: VisualEncoder = self._build_visual_encoder()
        self._language_model: LanguageModel = self._build_language_model()
        self._connector: Connector

    def _setup_example_input(self, img_size: int) -> None:
        self.example_input_array: tuple[torch.Tensor | list[torch.Tensor], torch.Tensor] = (
            [torch.randn(1, 3, img_size, img_size), torch.randn(2, 3, img_size, img_size)],
            self.language_model.tokenizer(
                ["test <|image|>.", "test <|image|> multiple images <|image|>."],
                padding=True,
                return_tensors="pt",
            ).input_ids,
        )

    def _save_hyperparameters(self) -> None:
        self.save_hyperparameters(
            {
                "model_config": OmegaConf.to_container(self.model_config, resolve=True),
                "trainer_config": OmegaConf.to_container(self.trainer_config, resolve=True),
            }
        )

    @property
    def visual_encoder(self) -> VisualEncoder:
        return self._visual_encoder

    @property
    def language_model(self) -> LanguageModel:
        return self._language_model

    @property
    def connector(self) -> Connector:
        return self._connector

    def _build_visual_encoder(self) -> VisualEncoder:
        encoder_config: VisualEncoderConfig = self.model_config.visual_encoder
        if encoder_config.type == "hf_visual_encoder":
            from .visual_encoders import HFVisualEncoder

            return HFVisualEncoder(encoder_config)
        else:
            error_msg = f"Unknown visual encoder type: {encoder_config.type}"
            log.error(error_msg)
            raise ValueError(error_msg)

    def _build_language_model(self) -> LanguageModel:
        llm_config: LLMConfig = self.model_config.llm
        if llm_config.type == "hf_llm":
            from .language_models import HFLLMLanguageModel

            return HFLLMLanguageModel(llm_config)
        else:
            error_msg = f"Unknown language model type: {llm_config.type}"
            log.error(error_msg)
            raise ValueError(error_msg)

    def _build_connector(self) -> Connector:
        connector_config: ConnectorConfig = self.model_config.connector
        if connector_config.type == "linear":
            from .connectors import LinearConnector

            return LinearConnector(
                connector_config, self.visual_encoder.hidden_size, self.language_model.hidden_size
            )
        elif connector_config.type == "mlp":
            from .connectors import MLPConnector

            return MLPConnector(
                connector_config, self.visual_encoder.hidden_size, self.language_model.hidden_size
            )
        else:
            error_msg = f"Unknown connector type: {connector_config.type}"
            log.error(error_msg)
            raise ValueError(error_msg)

    @override
    def forward(
        self,
        images: torch.Tensor | list[torch.Tensor] | None = None,
        texts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if images is None and texts is None:
            raise ValueError("Either images or texts must be provided")

        if images is not None:
            vision_features = self._process_vision_features(images)
        else:
            vision_features = None

        connector_output = self._connect_features(vision_features, texts)
        multimodal_features, attention_mask = connector_output

        log.debug(f"multimodal_features: {multimodal_features.shape}")
        log.debug(f"attention_mask: {attention_mask.shape}")

        # torch.save(multimodal_features, "multimodal_features.pt")
        # torch.save(attention_mask, "attention_mask.pt")
        outputs = self.language_model(
            inputs_embeds=multimodal_features, attention_mask=attention_mask
        )
        log.debug(f"outputs: {outputs.shape}")

        return outputs

    def _process_vision_features(
        self, images: torch.Tensor | list[torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        if isinstance(images, list):
            image_counts = [tensor.shape[0] for tensor in images]
            all_images = torch.cat(images, dim=0)  # [total_images, C, H, W]
            all_features = self.visual_encoder(all_images)
            return torch.split(all_features, image_counts, dim=0)
        else:
            return (self.visual_encoder(images),)

    def _connect_features(
        self, vision_features: tuple[torch.Tensor, ...] | None, texts: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        llm = self.language_model
        return self.connector(
            vision_features,
            texts,
            llm.embeddings,
            llm.image_token_id,
            llm.pad_token_id,
            self.model_config.connector.mask_format,
        )

    @override
    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train_loss")

    @override
    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val_loss")

    @override
    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test_loss")

    def _shared_step(self, batch: dict[str, torch.Tensor], log_name: str) -> torch.Tensor:
        images = batch["images"]
        texts = batch["texts"]
        labels = batch["labels"]

        log.debug(f"texts: {texts.shape}")
        log.debug(f"labels: {labels.shape}")
        # Save tensors to disk
        # torch.save(texts, "texts.pt")
        # torch.save(labels, "labels.pt")

        outputs = self(images, texts)
        # torch.save(outputs, "outputs.pt")
        loss = self._calculate_loss(outputs, labels)

        self.log(
            log_name,
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    @override
    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> str:
        images = batch["images"]
        texts = batch["texts"]
        vision_features = self._process_vision_features(images)

        connector_output = self._connect_features(vision_features, texts)
        multimodal_features, attention_mask = connector_output
        predictions = self._generate_predictions(multimodal_features, attention_mask)
        return predictions

    def _generate_predictions(
        self, multimodal_features: torch.Tensor, attention_mask: torch.Tensor
    ) -> str:
        output = self.language_model.language_model.generate(
            inputs_embeds=multimodal_features,
            attention_mask=attention_mask,
            num_beams=3,
            max_new_tokens=5,
            do_sample=False,
        )

        return self.language_model.tokenizer.decode(output[0], skip_special_tokens=True)

    def _calculate_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return get_loss(self.trainer_config, outputs, labels)

    @override
    def configure_optimizers(self) -> OptimizerLRScheduler:
        log.info("configure_optimizers")
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "ln_"]
        param_groups: dict[str, dict[str, list[Parameter]]] = {}

        visual_encoder_params = self._collect_param_groups(self.visual_encoder, no_decay)
        param_groups["visual_encoder"] = visual_encoder_params

        language_model_params = self._collect_param_groups(self.language_model, no_decay)
        param_groups["language_model"] = language_model_params

        connector_params = self._collect_param_groups(self.connector, no_decay)
        param_groups["connector"] = connector_params

        return get_optimizer(self.trainer_config, param_groups)

    def _collect_param_groups(
        self, module: torch.nn.Module, no_decay: list[str]
    ) -> dict[str, list[Parameter]]:
        decay_params = [
            p
            for n, p in module.named_parameters()
            if p.requires_grad and not any(no_decay_name in n for no_decay_name in no_decay)
        ]

        no_decay_params = [
            p
            for n, p in module.named_parameters()
            if p.requires_grad and any(no_decay_name in n for no_decay_name in no_decay)
        ]

        if decay_params or no_decay_params:
            return {"decay": decay_params, "no_decay": no_decay_params}
        return {"decay": [], "no_decay": []}

    def freeze_visual_encoder(self, freeze: bool = True) -> None:
        for param in self.visual_encoder.parameters():
            param.requires_grad = not freeze

    def freeze_language_model(self, freeze: bool = True, except_layer_norm: bool = True) -> None:
        for name, param in self.language_model.named_parameters():
            if except_layer_norm and (
                "layernorm" in name.lower() or "layer_norm" in name.lower() or "ln_" in name.lower()
                # or "embedding" in name.lower()
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

        self._log_trainable_params()

    def _log_trainable_params(self) -> None:
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        log.info(
            f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%} of total)"
        )
        for module_name, module in [
            ("visual_encoder", self.visual_encoder),
            ("language_model", self.language_model),
            ("connector", self.connector),
        ]:
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total = sum(p.numel() for p in module.parameters())
            if total > 0:
                log.info(f"  - {module_name}: {trainable:,} ({trainable / total:.2%} of {total:,})")
