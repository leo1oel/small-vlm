import logging
import re
from abc import ABC, abstractmethod
from typing import Any, override

import torch.nn as nn
from torch import Tensor

log: logging.Logger = logging.getLogger(__name__)


class Connector(nn.Module, ABC):
    """
    Abstract base class for all connector modules.
    Connectors are responsible for projecting visual features to a space
    compatible with text features.
    """

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
        self.projection_layer: nn.Module = self._build_projection_layer()

    @abstractmethod
    def _build_projection_layer(self) -> nn.Module:
        pass

    @override
    def forward(self, visual_features: Tensor) -> Tensor:
        return self.projection_layer(visual_features)


class IdentityConnector(Connector):
    def __init__(
        self,
        config: Any,
        image_hidden_size: int,
        text_hidden_size: int,
    ) -> None:
        if image_hidden_size != text_hidden_size:
            log.warning(
                f"IdentityConnector initialized with image_hidden_size ({image_hidden_size}) "
                f"!= text_hidden_size ({text_hidden_size}). Features will pass through unchanged."
            )
        super().__init__(config, image_hidden_size, text_hidden_size)

    @override
    def _build_projection_layer(self) -> nn.Module:
        return nn.Identity()


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


class MLPConnector(Connector):
    ACTIVATION_MAP: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,  # Swish/SiLU
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }

    @override
    def __init__(
        self,
        config: Any,
        image_hidden_size: int,
        text_hidden_size: int,
    ) -> None:
        self.num_layers: int = 2
        self.activation_name: str = "gelu"

        # Parse num_layers and activation_name from the connector's name string
        self._parse_config_name(config.name)

        super().__init__(config, image_hidden_size, text_hidden_size)

    def _parse_config_name(self, name: str) -> None:
        pattern = r"mlp_(\d+)_(\w+)"  # e.g., mlp_2_gelu, mlp_3_relu
        match = re.match(pattern, name)
        if match:
            try:
                self.num_layers = int(match.group(1))
                self.activation_name = match.group(2).lower()
                if self.activation_name not in self.ACTIVATION_MAP:
                    log.warning(
                        f"MLPConnector: Activation '{self.activation_name}' from name '{name}' is not recognized. "
                        f"Falling back to default activation '{MLPConnector.activation_name}'. "
                        f"Supported: {list(self.ACTIVATION_MAP.keys())}"
                    )
                    self.activation_name = "gelu"  # Fallback to default if parsed name is invalid
            except ValueError:
                log.warning(
                    f"MLPConnector: Could not parse num_layers from '{match.group(1)}' in name '{name}'. "
                    f"Using default num_layers: {self.num_layers}."
                )
        else:
            log.warning(
                f"MLPConnector name '{name}' does not match pattern 'mlp_NUMLAYERS_ACTIVATION'. "
                f"Using defaults: num_layers={self.num_layers}, activation_name='{self.activation_name}'."
            )

    @override
    def _build_projection_layer(self) -> nn.Module:
        if self.num_layers < 1:
            raise ValueError(
                f"MLPConnector: Number of layers must be at least 1, got {self.num_layers}"
            )

        activation_class = self.ACTIVATION_MAP.get(self.activation_name)
        if activation_class is None:
            # This case should ideally be handled by _parse_config_name fallback,
            # but as a safeguard:
            log.error(
                f"MLPConnector: Unsupported activation function '{self.activation_name}'. "
                f"Supported activations: {list(self.ACTIVATION_MAP.keys())}. "
                f"Defaulting to GELU."
            )
            activation_class = nn.GELU  # Fallback

        layers: list[nn.Module] = []

        for i in range(self.num_layers):
            # The first layer maps from image_hidden_size to text_hidden_size.
            # Subsequent hidden layers map from text_hidden_size to text_hidden_size.
            # The final layer also outputs text_hidden_size.
            input_dim = self.image_hidden_size if i == 0 else self.text_hidden_size
            output_dim = (
                self.text_hidden_size
            )  # All layers in the MLP project towards/maintain text_hidden_size

            layers.append(nn.Linear(input_dim, output_dim))

            # Add activation function for all layers except the last one
            if i < self.num_layers - 1:
                layers.append(activation_class())

        return nn.Sequential(*layers)


# --- Connector Mapping and Exports ---

# This map is used by your _build_connector function to instantiate the correct connector type.
# The keys ('identity', 'linear', 'mlp') should match the `connector_config.type` values.
connector_map: dict[str, type[Connector]] = {
    "identity": IdentityConnector,
    "linear": LinearConnector,
    "mlp": MLPConnector,
}
