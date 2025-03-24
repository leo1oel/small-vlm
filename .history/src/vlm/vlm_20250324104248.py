from pathlib import Path

import hydra
from omegaconf import DictConfig
import logging

logger: logging.Logger = logging.getLogger(name=__name__)
config_path: Path = Path(__file__).resolve().parent.parent.parent / "conf"

def vlm(cfg: DictConfig) -> None:
    logger.info(msg=f"模型配置: {cfg.model}")
    logger.debug(msg=f"模型配置: {cfg.model}")
    logger.warning(msg=f"模型配置: {cfg.model}")
    logger.error(msg=f"模型配置: {cfg.model}")


@hydra.main(version_base=None, config_path=str(config_path), config_name="config")
def main(cfg: DictConfig) -> None:
    vlm(cfg)


if __name__ == "__main__":
    main()
