from abc import ABC, abstractmethod
from typing import override

import torch
import torch.nn as nn


class Connector(nn.Module, ABC):
    @abstractmethod
    @override
    def forward(self, visual_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def input_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        pass
