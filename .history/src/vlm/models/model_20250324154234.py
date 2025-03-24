from omegaconf import DictConfig

class VLM:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg: DictConfig = cfg
