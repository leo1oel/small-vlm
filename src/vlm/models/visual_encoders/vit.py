import logging
from typing import override

import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModel

from ...config.config_schema import VisualEncoderConfig
from .base import VisualEncoder

log: logging.Logger = logging.getLogger(name=__name__)


class ViTEncoder(VisualEncoder):
    def __init__(self, config: VisualEncoderConfig) -> None:
        super().__init__(config)

    @override
    def _build_preprocessor(self) -> AutoImageProcessor:
        self.preprocessor: AutoImageProcessor = AutoImageProcessor.from_pretrained(
            self.hf_name, trust_remote_code=True, use_fast=True
        )
        return self.preprocessor  # pyright: ignore

    @override
    def _build_visual_encoder(self) -> AutoModel:
        self.visual_encoder: AutoModel = AutoModel.from_pretrained(
            self.hf_name, trust_remote_code=True
        )
        if getattr(self.visual_encoder, "vision_model", None) is not None:  # pyright: ignore
            self.visual_encoder = self.visual_encoder.vision_model  # pyright: ignore

        return self.visual_encoder  # pyright: ignore

    @override
    def _build_hf_config(self) -> AutoConfig:
        self.hf_config: AutoConfig = AutoConfig.from_pretrained(
            self.hf_name, trust_remote_code=True
        )
        return self.hf_config  # pyright: ignore

    @override
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.visual_encoder(images, output_hidden_states=True)  # pyright: ignore
        hidden_states: torch.Tensor = outputs.hidden_states[self.output_layer]  # pyright: ignore
        if not self.config.use_cls_token:
            return hidden_states[:, 1:].contiguous()  # pyright: ignore
        return hidden_states  # pyright: ignore
