from typing import override

import torch
import torch.nn as nn

from ...config.config_schema import ConnectorConfig
from .base import Connector


class LinearConnector(Connector):
    def __init__(self, config: ConnectorConfig) -> None:
        super().__init__(config)

    @override
    def _build_projection_layer(self) -> nn.Module:
        return nn.Linear(self.config.visual_dim, self.config.text_dim)

    @override
    def _initialize_layers(self) -> None:
        nn.init.normal_(self.projection_layer.weight, mean=0.0, std=0.02)  # pyright: ignore
        nn.init.zeros_(self.projection_layer.bias)  # pyright: ignore

    @override
    def projection(self, visual_features: torch.Tensor) -> torch.Tensor:
        return self.projection_layer(visual_features)  # pyright: ignore
