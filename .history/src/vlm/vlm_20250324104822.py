from pathlib import Path

import hydra
from omegaconf import DictConfig
import logging

log: logging.Logger = logging.getLogger(name=__name__)
config_path: Path = Path(__file__).resolve().parent.parent.parent / "conf"

def vlm(cfg: DictConfig) -> None:
    log.info(msg=f"模型配置: {cfg}")
    log.debug(msg=f"模型配置: {cfg}")
    log.warning(msg=f"模型配置: {cfg}")
    log.error(msg=f"模型配置: {cfg}")


@hydra.main(version_base=None, config_path=str(config_path), config_name="config")
def main(cfg: DictConfig) -> None:
    log.info(msg=f"模型配置: {cfg}")
    vlm(cfg)


if __name__ == "__main__":
    main()
