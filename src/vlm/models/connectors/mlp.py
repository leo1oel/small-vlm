from typing import override

import torch
import torch.nn as nn

from ...config.config_schema import ConnectorConfig
from .base import Connector


class MLPConnector(Connector):
    def __init__(
        self, config: ConnectorConfig, image_hidden_size: int, text_hidden_size: int
    ) -> None:
        super().__init__(config, image_hidden_size, text_hidden_size)

    @override
    def _build_projection_layer(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.image_hidden_size, self.text_hidden_size),
            nn.ReLU(),
            nn.Linear(self.text_hidden_size, self.text_hidden_size),
        )

    @override
    def _initialize_layers(self) -> None:
        nn.init.normal_(self.projection_layer[0].weight, mean=0.0, std=0.02)  # pyright: ignore
        nn.init.zeros_(self.projection_layer[0].bias)  # pyright: ignore

        nn.init.normal_(self.projection_layer[2].weight, mean=0.0, std=0.02)  # pyright: ignore
        nn.init.zeros_(self.projection_layer[2].bias)  # pyright: ignore

    @override
    def projection(self, visual_features: torch.Tensor) -> torch.Tensor:
        return self.projection_layer(visual_features)  # pyright: ignore
