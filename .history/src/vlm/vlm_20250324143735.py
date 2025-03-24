from pathlib import Path

import hydra
from omegaconf import DictConfig
import logging
from rich.console import Console
from rich.table import Table

log: logging.Logger = logging.getLogger(name=__name__)
config_path: Path = Path(__file__).resolve().parent.parent.parent / "conf"

def load_model(cfg: DictConfig) -> None:
    model_name: str = cfg.model.name
    model_config_path: Path = (config_path / "model" / f"{model_name}.yaml").resolve()
    file_url: str = f"file://{model_config_path}"
    log.info(f"Loading model: [red][link={file_url}]Example[/link][/red]")
    log.info(f"Visual encoder: [bold cyan][link=file://conf/model/default.yaml]{cfg.model.visual_encoder.name}[/link][/bold cyan]")
    log.info(f"LLM: [bold green]{cfg.model.llm.name}[/bold green]")
    log.info(f"Connector: [magenta on blue]{cfg.model.connector.name}[/magenta on blue]")

def vlm(cfg: DictConfig) -> None:
    load_model(cfg)


@hydra.main(version_base=None, config_path=str(config_path), config_name="config")
def main(cfg: DictConfig) -> None:
    vlm(cfg)


if __name__ == "__main__":
    main()
