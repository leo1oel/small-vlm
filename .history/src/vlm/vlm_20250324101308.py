from pathlib import Path

import hydra
from omegaconf import DictConfig
import logging

logger: logging.Logger = logging.getLogger(name=__name__)
config_path: Path = Path(__file__).resolve().parent.parent.parent / "conf"



def vlm(cfg: DictConfig) -> None:
    if hasattr(cfg, 'model'):
        logger.debug(f"模型配置: {cfg.model}")
    else:
        logger.warning("没有找到模型配置!")
        logger.debug(f"可用的配置键: {list(cfg.keys())}")


@hydra.main(version_base=None, config_path=str(config_path), config_name="config")
def main(cfg: DictConfig) -> None:
    vlm(cfg)


if __name__ == "__main__":
    main()
