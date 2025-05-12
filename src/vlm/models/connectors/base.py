import logging
from abc import ABC, abstractmethod
from typing import override

import torch.nn as nn
from torch import Tensor

from ...config.config_schema import ConnectorConfig

log: logging.Logger = logging.getLogger(name=__name__)


class Connector(nn.Module, ABC):
    def __init__(
        self,
        config: ConnectorConfig,
        image_hidden_size: int,
        text_hidden_size: int,
    ) -> None:
        super().__init__()
        self.config: ConnectorConfig = config
        self.name: str = self.config.name

        self.image_hidden_size: int = image_hidden_size
        self.text_hidden_size: int = text_hidden_size
        self.projection_layer: nn.Module = self.build_projection_layer()

    @abstractmethod
    def _build_projection_layer(self) -> nn.Module:
        pass

    def build_projection_layer(self) -> nn.Module:
        return self._build_projection_layer()

    @abstractmethod
    def projection(self, visual_features: Tensor) -> Tensor:
        pass

    @override
    def forward(self, visual_features: Tensor):
        return self.projection(visual_features)
