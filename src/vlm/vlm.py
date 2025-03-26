import logging
from pathlib import Path

import hydra
from lightning.pytorch.utilities.model_summary.model_summary import ModelSummary

from .config.config_schema import AppConfig, ModelConfig, TrainerConfig, register_configs
from .inference.inference import inference
from .models.model import VLM
from .train.trainer import train

log: logging.Logger = logging.getLogger(name=__name__)
config_path: Path = Path(__file__).resolve().parent / "config"


def print_model(cfg: ModelConfig) -> None:
    model_name: str = cfg.name
    model_config_path: Path = (config_path / "model" / f"{model_name}.yaml").resolve()
    model_url: str = f"file://{model_config_path}"

    visual_encoder_name: str = cfg.visual_encoder.name
    visual_encoder_path: Path = (
        config_path / "model" / "visual_encoder" / f"{visual_encoder_name}.yaml"
    ).resolve()
    visual_url: str = f"file://{visual_encoder_path}"

    llm_name: str = cfg.llm.name
    llm_path: Path = (config_path / "model" / "llm" / f"{llm_name}.yaml").resolve()
    llm_url: str = f"file://{llm_path}"

    connector_name: str = cfg.connector.name
    connector_path: Path = (
        config_path / "model" / "connector" / f"{connector_name}.yaml"
    ).resolve()
    connector_url: str = f"file://{connector_path}"

    log.info(f"Loading model: [bold red][link={model_url}]{model_name}[/link][/bold red]")
    log.info(
        f"Visual encoder: [bold cyan][link={visual_url}]{visual_encoder_name}[/link][/bold cyan]"
    )
    log.info(f"LLM: [bold blue][link={llm_url}]{llm_name}[/link][/bold blue]")
    log.info(f"Connector: [bold yellow][link={connector_url}]{connector_name}[/link][/bold yellow]")


def load_model(model_cfg: ModelConfig, trainer_cfg: TrainerConfig) -> VLM:
    print_model(model_cfg)
    model: VLM = VLM(model_cfg, trainer_cfg)
    # example_input_array: tuple[torch.Tensor | list[torch.Tensor], torch.Tensor] = (
    #         [torch.randn(1, 3, 224, 224), torch.randn(3, 3, 224, 224)],  # 图像输入
    #         model.language_model.tokenizer(
    #             ["test <|image|>.", "test <|image|> multiple <|image|> images <|image|>."],
    #             padding=True,
    #             return_tensors="pt",
    #         ).input_ids,  # pyright: ignore
    #     )
    # model(example_input_array[0], example_input_array[1])
    summary = ModelSummary(model)  # pyright: ignore
    print(summary)
    return model


def vlm(cfg: AppConfig) -> None:
    model: VLM = load_model(cfg.model, cfg.trainer)
    if cfg.mode.is_training:
        train(cfg.trainer, model)
    else:
        inference(cfg.trainer, model)


@hydra.main(version_base=None, config_path=str(config_path), config_name="config")  # pyright: ignore
def main(cfg: AppConfig) -> None:
    vlm(cfg)


register_configs()

if __name__ == "__main__":
    main()
