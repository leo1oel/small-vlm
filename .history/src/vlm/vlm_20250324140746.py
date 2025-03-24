from pathlib import Path

import hydra
from omegaconf import DictConfig
import logging
from rich.console import Console
from rich.table import Table

log: logging.Logger = logging.getLogger(name=__name__)
config_path: Path = Path(__file__).resolve().parent.parent.parent / "conf"

def load_model(cfg: DictConfig) -> None:
    log.info(f"Loading model: [bold yellow]{cfg.model.name}[/bold yellow]")
    log.info(f"Visual encoder: [cyan]{cfg.model.visual_encoder.name}[/cyan]")
    log.info(f"LLM: [green]{cfg.model.llm.name}[/green]")
    log.info(f"Connector: [magenta]{cfg.model.connector.name}[/magenta]")
    # 创建表格
    table = Table(title="模型配置")
    table.add_column("组件")
    table.add_column("名称")
    
    table.add_row("模型", cfg.model.name)
    table.add_row("视觉编码器", cfg.model.visual_encoder.name)
    table.add_row("LLM", cfg.model.llm.name)
    
    log.info(Console().render_str(table))

def vlm(cfg: DictConfig) -> None:
    load_model(cfg)


@hydra.main(version_base=None, config_path=str(config_path), config_name="config")
def main(cfg: DictConfig) -> None:
    vlm(cfg)


if __name__ == "__main__":
    main()
