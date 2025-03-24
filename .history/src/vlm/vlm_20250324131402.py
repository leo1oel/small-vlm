from pathlib import Path

import hydra
from omegaconf import DictConfig
import logging

log: logging.Logger = logging.getLogger(name=__name__)
config_path: Path = Path(__file__).resolve().parent.parent.parent / "conf"

def load_model(cfg: DictConfig) -> None:
    log.info(msg=f"Loading model: {cfg.model.name}")
    log.info(msg=f"Visual encoder: {cfg.model.visual_encoder.name}")
    log.info(msg=f"LLM: {cfg.model.llm.name}")
    log.info(msg=f"Connector: {cfg.model.connector.name}")

def vlm(cfg: DictConfig) -> None:
    load_model(cfg)


@hydra.main(version_base=None, config_path=str(config_path), config_name="config")
def main(cfg: DictConfig) -> None:
    vlm(cfg)


if __name__ == "__main__":
    main()
