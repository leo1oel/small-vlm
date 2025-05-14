from abc import ABC, abstractmethod
from typing import Any, override

import torch.nn as nn
from torch import Tensor


class Connector(nn.Module, ABC):
    def __init__(
        self,
        config: Any,
        image_hidden_size: int,
        text_hidden_size: int,
    ) -> None:
        super().__init__()
        self.config: Any = config
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
