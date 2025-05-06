from typing import cast, override

import torch.nn as nn
from torch import Tensor, device, dtype

from ...config.config_schema import ConnectorConfig
from .base import Connector


class LinearConnector(Connector):
    def __init__(
        self,
        config: ConnectorConfig,
        image_hidden_size: int,
        text_hidden_size: int,
        torch_dtype: dtype,
        torch_device: device,
    ) -> None:
        super().__init__(config, image_hidden_size, text_hidden_size, torch_dtype, torch_device)

    @override
    def _build_projection_layer(self) -> nn.Module:
        return nn.Linear(
            self.image_hidden_size,
            self.text_hidden_size,
        )

    @override
    def _initialize_layers(self) -> None:
        linear_layer = cast(nn.Linear, self.projection_layer)
        nn.init.normal_(linear_layer.weight, mean=0.0, std=0.02)
        nn.init.zeros_(linear_layer.bias)

    @override
    def projection(self, visual_features: Tensor) -> Tensor:
        return self.projection_layer(visual_features)
