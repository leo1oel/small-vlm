from pathlib import Path

import hydra
from omegaconf import DictConfig
import logging
from rich.console import Console
from rich.table import Table

log: logging.Logger = logging.getLogger(name=__name__)
config_path: Path = Path(__file__).resolve().parent.parent.parent / "conf"

def load_model(cfg: DictConfig) -> None:
    log.info(f"Loading model: [bold yellow on red]{cfg.model.name}[/bold yellow on red]")
    log.info(f"Visual encoder: [bold cyan on red]{cfg.model.visual_encoder.name}[/bold cyan on red]")
    log.info(f"LLM: [bold green on red]{cfg.model.llm.name}[/bold green on red]")
    log.info(f"Connector: [magenta on red]{cfg.model.connector.name}[/magenta on red]")

def vlm(cfg: DictConfig) -> None:
    load_model(cfg)


@hydra.main(version_base=None, config_path=str(config_path), config_name="config")
def main(cfg: DictConfig) -> None:
    vlm(cfg)


if __name__ == "__main__":
    main()
