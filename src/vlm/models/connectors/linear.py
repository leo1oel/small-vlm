import torch

from ...config.config_schema import ConnectorConfig
from .base import Connector

class LinearConnector(Connector):
    def __init__(self, config: ConnectorConfig):
        super().__init__()
        self.config: ConnectorConfig = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
