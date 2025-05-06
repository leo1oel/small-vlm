import logging
from abc import ABC, abstractmethod
from typing import NamedTuple, override

import torch.nn as nn
from torch import Tensor, device, dtype

from ...config.config_schema import ConnectorConfig

log: logging.Logger = logging.getLogger(name=__name__)


class ProcessedVisualFeatures(NamedTuple):
    features: list[Tensor]


class ForwardOutput(NamedTuple):
    embeddings: Tensor
    attention_mask: Tensor


class Connector(nn.Module, ABC):
    def __init__(
        self,
        config: ConnectorConfig,
        image_hidden_size: int,
        text_hidden_size: int,
        torch_dtype: dtype,
        torch_device: device,
    ) -> None:
        super().__init__()
        self.config: ConnectorConfig = config
        self.name: str = self.config.name
        self.torch_dtype: dtype = torch_dtype
        self.torch_device: device = torch_device

        self.image_hidden_size: int = image_hidden_size
        self.text_hidden_size: int = text_hidden_size
        self.projection_layer: nn.Module = self.build_projection_layer().to(
            dtype=self.torch_dtype,
            device=self.torch_device,
        )
        self.initialize_layers()

    @abstractmethod
    def _build_projection_layer(self) -> nn.Module:
        pass

    def build_projection_layer(self) -> nn.Module:
        return self._build_projection_layer()

    @abstractmethod
    def _initialize_layers(self) -> None:
        pass

    def initialize_layers(self) -> None:
        self._initialize_layers()

    @abstractmethod
    def projection(self, visual_features: Tensor) -> Tensor:
        pass

    @override
    def forward(self, visual_features: Tensor):
        return self.projection(visual_features)
