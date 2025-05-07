import logging
from typing import cast, override

from torch import Tensor, device, dtype
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    BaseImageProcessor,
    PretrainedConfig,
    PreTrainedModel,
)

from ...config.config_schema import VisualEncoderConfig
from .base import VisualEncoder

log: logging.Logger = logging.getLogger(name=__name__)


class HFVisualEncoder(VisualEncoder):
    def __init__(
        self, config: VisualEncoderConfig, torch_dtype: dtype, torch_device: device
    ) -> None:
        super().__init__(config, torch_dtype, torch_device)

    @override
    def _build_preprocessor(self) -> BaseImageProcessor:
        return cast(
            BaseImageProcessor,
            AutoImageProcessor.from_pretrained(
                self.hf_name,
                trust_remote_code=True,
                use_fast=True,
            ),
        )

    @override
    def _build_visual_encoder(self) -> PreTrainedModel:
        visual_encoder: PreTrainedModel = cast(
            PreTrainedModel,
            AutoModel.from_pretrained(
                self.hf_name,
                trust_remote_code=True,
            ),
        )
        if getattr(visual_encoder, "vision_model", None):
            visual_encoder = visual_encoder.vision_model  # pyright: ignore
        visual_encoder.to(
            dtype=self.torch_dtype,
            device=self.torch_device,
        )
        return visual_encoder

    @override
    def _build_hf_config(self) -> PretrainedConfig:
        return cast(
            PretrainedConfig, AutoConfig.from_pretrained(self.hf_name, trust_remote_code=True)
        )

    @override
    def forward(self, images: Tensor | list[Tensor]) -> Tensor | list[Tensor]:
        if type(images) is list:
            image_features: list[Tensor] | Tensor = []
            for image in images:
                outputs = self.visual_encoder(
                    image.to(
                        dtype=self.torch_dtype,
                        device=self.torch_device,
                    ).unsqueeze(0),
                    output_hidden_states=True,
                )
                hidden_states: Tensor = outputs.hidden_states[self.output_layer].to(image.dtype)
                if not self.config.use_cls_token:
                    image_features.append(hidden_states[:, 1:])
                else:
                    image_features.append(hidden_states)
        else:
            outputs = self.visual_encoder(
                images.to(
                    dtype=self.torch_dtype,
                    device=self.torch_device,
                ),
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[self.output_layer].to(images.dtype)
            if not self.config.use_cls_token:
                image_features = hidden_states[:, 1:]
            else:
                image_features = hidden_states
        return image_features
