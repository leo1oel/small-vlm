from typing import Any, override

import torch.nn as nn
from torch import Tensor

from .base import Connector


class LinearConnector(Connector):
    def __init__(
        self,
        config: Any,
        image_hidden_size: int,
        text_hidden_size: int,
    ) -> None:
        super().__init__(config, image_hidden_size, text_hidden_size)

    @override
    def _build_projection_layer(self) -> nn.Module:
        return nn.Linear(
            self.image_hidden_size,
            self.text_hidden_size,
        )

    @override
    def projection(self, visual_features: Tensor) -> Tensor:
        return self.projection_layer(visual_features)
