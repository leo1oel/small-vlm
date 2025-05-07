import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast, override

import torch.nn as nn
from torch import Tensor, device, dtype
from transformers import BaseImageProcessor, PretrainedConfig, PreTrainedModel

from ...config.config_schema import VisualEncoderConfig

log: logging.Logger = logging.getLogger(name=__name__)


@dataclass
class VisualModelConfig:
    hidden_size: int | None = None
    img_size: int | None = None
    patch_size: int | None = None


class VisualEncoder(nn.Module, ABC):
    def __init__(
        self, config: VisualEncoderConfig, torch_dtype: dtype, torch_device: device
    ) -> None:
        super().__init__()
        self.config: VisualEncoderConfig = config
        self.name: str = self.config.name
        self.hf_name: str = self.config.hf_name
        self.model_type: str = self.config.type
        self.torch_dtype: dtype = torch_dtype
        self.torch_device: device = torch_device

        # model config
        self.model_config: VisualModelConfig = VisualModelConfig(
            hidden_size=getattr(self.config, "hidden_size", None),
            img_size=getattr(self.config, "img_size", None),
            patch_size=getattr(self.config, "patch_size", None),
        )

        # output layer
        self.output_layer: int = getattr(self.config, "output_layer", -1)
        log.info(f"Using {self.output_layer} layer as output layer")

        # use cls token
        self.use_cls_token: bool = getattr(self.config, "use_cls_token", False)
        log.info(f"Use cls token: {self.use_cls_token}")

        self.initialize_components()

    # initialize all components
    def initialize_components(self) -> None:
        self._hf_config: PretrainedConfig = self._build_hf_config()

        self.verify_config()

        self._visual_encoder: PreTrainedModel = self._build_visual_encoder()
        self._preprocessor: BaseImageProcessor = self._build_preprocessor()

        # calculate token size
        self.token_size: int = (
            cast(int, self.model_config.img_size) // cast(int, self.model_config.patch_size)
        ) ** 2 + self.use_cls_token

    @property
    def preprocessor(self) -> BaseImageProcessor:
        return self._preprocessor

    @property
    def visual_encoder(self) -> PreTrainedModel:
        return self._visual_encoder

    @property
    def hf_config(self) -> PretrainedConfig:
        return self._hf_config

    @property
    def hidden_size(self) -> int:
        return cast(int, self.model_config.hidden_size)

    @hidden_size.setter
    def hidden_size(self, value: int) -> None:
        self.model_config.hidden_size = value

    @property
    def img_size(self) -> int:
        return cast(int, self.model_config.img_size)

    @img_size.setter
    def img_size(self, value: int) -> None:
        self.model_config.img_size = value

    @property
    def patch_size(self) -> int:
        return cast(int, self.model_config.patch_size)

    @patch_size.setter
    def patch_size(self, value: int) -> None:
        self.model_config.patch_size = value

    @abstractmethod
    def _build_preprocessor(self) -> BaseImageProcessor:
        pass

    @abstractmethod
    def _build_visual_encoder(self) -> PreTrainedModel:
        pass

    @abstractmethod
    def _build_hf_config(self) -> PretrainedConfig:
        pass

    @abstractmethod
    @override
    def forward(self, images: Tensor | list[Tensor]) -> Tensor | list[Tensor]:
        pass

    def verify_config(self) -> None:
        config_pairs = [
            ("hidden_size", self.get_config("hidden_size"), self.hidden_size),
            ("img_size", self.get_config("image_size"), self.img_size),
            ("patch_size", self.get_config("patch_size"), self.patch_size),
        ]

        for key, model_value, config_value in config_pairs:
            self._verify_param_match(key, model_value, config_value)

    def get_config(self, key: str) -> int | str | None:
        vision_config = getattr(self.hf_config, "vision_config", None)
        if vision_config and hasattr(vision_config, key):
            return getattr(vision_config, key)

        return getattr(self.hf_config, key, None)

    def _verify_param_match(
        self, key: str, model_value: int | str | None, config_value: int | str | None
    ) -> None:
        capitalized_key = key.capitalize()

        if model_value is None and config_value is None:
            log.warning(f"{capitalized_key} not found in config for {self.hf_name}")
        elif model_value is not None and config_value is None:
            setattr(self, key, int(model_value))
            if hasattr(self.config, key):
                setattr(self.config, key, int(model_value))
            log.info(f"{capitalized_key} not found in config, using hf config: {model_value}")
        elif model_value is None and config_value is not None:
            log.warning(f"{capitalized_key} not found in hf config for {self.hf_name}")
        elif model_value is not None and config_value is not None:
            if model_value != config_value:
                error_msg = f"{capitalized_key} mismatch: hf config: {model_value} != config: {config_value}"
                log.warning(error_msg)
            else:
                log.info(
                    f"{capitalized_key} verified: hf config: {model_value} == config: {config_value}"
                )
