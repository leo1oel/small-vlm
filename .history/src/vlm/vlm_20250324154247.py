from pathlib import Path

import hydra
from omegaconf import DictConfig
import logging
from .models.model import VLM

log: logging.Logger = logging.getLogger(name=__name__)
config_path: Path = Path(__file__).resolve().parent.parent.parent / "conf"

def print_model(cfg: DictConfig) -> None:
    model_name: str = cfg.model.name
    model_config_path: Path = (config_path / "model" / f"{model_name}.yaml").resolve()
    model_url: str = f"file://{model_config_path}"

    visual_encoder_name: str = cfg.model.visual_encoder.name
    visual_encoder_path: Path = (config_path / "model" / "visual_encoder" / f"{visual_encoder_name}.yaml").resolve()
    visual_url: str = f"file://{visual_encoder_path}"

    llm_name: str = cfg.model.llm.name
    llm_path: Path = (config_path / "model" / "llm" / f"{llm_name}.yaml").resolve()
    llm_url: str = f"file://{llm_path}"

    connector_name: str = cfg.model.connector.name
    connector_path: Path = (config_path / "model" / "connector" / f"{connector_name}.yaml").resolve()
    connector_url: str = f"file://{connector_path}"

    log.info(f"Loading model: [bold red][link={model_url}]{model_name}[/link][/bold red]")
    log.info(f"Visual encoder: [bold yellow][link={visual_url}]{visual_encoder_name}[/link][/bold yellow]")
    log.info(f"LLM: [bold blue][link={llm_url}]{llm_name}[/link][/bold blue]")
    log.info(f"Connector: [bold green][link={connector_url}]{connector_name}[/link][/bold green]")

def load_model(cfg: DictConfig) -> VLM:
    print_model(cfg)
    model = VLM(cfg)
    return model
    

def vlm(cfg: DictConfig) -> None:
    model = load_model(cfg)
    


@hydra.main(version_base=None, config_path=str(config_path), config_name="config")
def main(cfg: DictConfig) -> None:
    vlm(cfg)


if __name__ == "__main__":
    main()
