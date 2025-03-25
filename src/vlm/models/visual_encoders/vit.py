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
            self.hf_name, trust_remote_code=True
        )  # pyright: ignore[reportUnknownMemberType]
        return self.preprocessor  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    @override
    def _build_visual_encoder(self) -> AutoModel:
        self.visual_encoder: AutoModel = AutoModel.from_pretrained(
            self.hf_name, trust_remote_code=True
        )  # pyright: ignore[reportUnknownMemberType]
        return self.visual_encoder  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    @override
    def _build_hf_config(self) -> AutoConfig:
        self.hf_config: AutoConfig = AutoConfig.from_pretrained(
            self.hf_name, trust_remote_code=True
        )  # pyright: ignore[reportUnknownMemberType]
        return self.hf_config  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    @override
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.visual_encoder(images, output_hidden_states=True)  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
        return outputs.hidden_states[self.output_layer]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    @override
    def get_output_dim(self) -> int | None:
        if getattr(self.hf_config, "hidden_size", None) is not None:
            return self.hf_config.hidden_size  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue]
        else:
            return None

    @override
    def get_img_size(self) -> int | None:
        if getattr(self.hf_config, "image_size", None) is not None:
            return self.hf_config.image_size  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue]
        else:
            return None

    @override
    def get_patch_size(self) -> int | None:
        if getattr(self.hf_config, "patch_size", None) is not None:
            return self.hf_config.patch_size  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue]
        else:
            return None
