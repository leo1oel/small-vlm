from typing import override

import torch.nn as nn
from torch import Tensor, device, dtype

from ...config.config_schema import ConnectorConfig
from .base import Connector


class IdentityConnector(Connector):
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
        return nn.Identity()

    @override
    def _initialize_layers(self) -> None:
        pass

    @override
    def projection(self, visual_features: Tensor) -> Tensor:
        return visual_features
