from .base import Connector
from .identity import IdentityConnector
from .linear import LinearConnector
from .mlp_n_activation import MLPConnector

__all__ = ["Connector", "IdentityConnector", "LinearConnector", "MLPConnector", "connector_map"]

connector_map = {
    "identity": IdentityConnector,
    "linear": LinearConnector,
    "mlp": MLPConnector,
}
