from pathlib import Path

import hydra
from omegaconf import DictConfig
import logging

log: logging.Logger = logging.getLogger(name=__name__)
config_path: Path = Path(__file__).resolve().parent.parent.parent / "conf"

def load_model(cfg: DictConfig) -> None:
    log.info(f"Loading model: [bold yellow]{cfg.model.name}[/bold yellow]")
    log.info(f"Visual encoder: [cyan]{cfg.model.visual_encoder.name}[/cyan]")
    log.info(f"LLM: [green]{cfg.model.llm.name}[/green]")
    log.debug(f"Connector: [magenta]{cfg.model.connector.name}[/magenta]")

def vlm(cfg: DictConfig) -> None:
    load_model(cfg)


@hydra.main(version_base=None, config_path=str(config_path), config_name="config")
def main(cfg: DictConfig) -> None:
    vlm(cfg)


if __name__ == "__main__":
    main()
