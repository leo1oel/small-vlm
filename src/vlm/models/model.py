from ..config.config_schema import ModelConfig


class VLM:
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg: ModelConfig = cfg
