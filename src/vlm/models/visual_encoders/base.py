import logging
from abc import ABC, abstractmethod
from typing import override

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoImageProcessor, AutoModel

from ...config.config_schema import VisualEncoderConfig

log: logging.Logger = logging.getLogger(name=__name__)


class VisualEncoder(nn.Module, ABC):
    def __init__(self, config: VisualEncoderConfig) -> None:
        super().__init__()
        self.config: VisualEncoderConfig = config
        self.name: str = self.config.name
        self.hf_name: str = self.config.hf_name
        self.model_type: str = self.config.type
        self.hidden_size: int | None = getattr(self.config, "hidden_size", None)
        self.img_size: int | None = getattr(self.config, "img_size", None)
        self.patch_size: int | None = getattr(self.config, "patch_size", None)
        self.output_layer: int = getattr(self.config, "output_layer", -1)
        log.info(
            f"[bold green]Using[/bold green] [bold blue] {self.output_layer} [/bold blue] [bold green]layer as output layer[/bold green]"
        )
        self.preprocessor: AutoImageProcessor = self.build_preprocessor()
        self.visual_encoder: AutoModel = self.build_visual_encoder()
        self.hf_config: AutoConfig = self.build_hf_config()
        self.verify_config()
        self.token_size: int = (self.img_size // self.patch_size) ** 2 + self.config.use_cls_token  # pyright: ignore

    @abstractmethod
    def _build_preprocessor(self) -> AutoImageProcessor:
        pass

    def build_preprocessor(self) -> AutoImageProcessor:
        log.info(
            f"[bold green]Building image preprocessor for[/bold green] [bold blue] {self.hf_name}[/bold blue]"
        )
        return self._build_preprocessor()

    @abstractmethod
    def _build_visual_encoder(self) -> AutoModel:
        pass

    def build_visual_encoder(self) -> AutoModel:
        log.info(
            f"[bold green]Building visual encoder for[/bold green] [bold blue] {self.hf_name}[/bold blue]"
        )
        return self._build_visual_encoder()

    @abstractmethod
    def _build_hf_config(self) -> AutoConfig:
        pass

    def build_hf_config(self) -> AutoConfig:
        log.info(
            f"[bold green]Building hf config for[/bold green] [bold blue] {self.hf_name}[/bold blue]"
        )
        return self._build_hf_config()

    @abstractmethod
    @override
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pass

    def verify_config(self) -> None:
        model_hidden_size: int | str | None = self.get_config("hidden_size")
        model_img_size: int | str | None = self.get_config("image_size")
        model_patch_size: int | str | None = self.get_config("patch_size")

        self.verify_equal("hidden_size", model_hidden_size, self.hidden_size)
        self.verify_equal("img_size", model_img_size, self.img_size)
        self.verify_equal("patch_size", model_patch_size, self.patch_size)

    def get_config(self, key: str) -> int | str | None:
        if getattr(self.hf_config, "vision_config", None) is not None:
            if getattr(self.hf_config.vision_config, key, None) is not None:  # pyright: ignore
                return getattr(self.hf_config.vision_config, key)  # pyright: ignore
        elif getattr(self.hf_config, key, None) is not None:
            return getattr(self.hf_config, key)  # pyright: ignore
        else:
            return None

    def verify_equal(
        self, key: str, model_value: int | str | None, config_value: int | str | None
    ) -> None:
        if model_value is None and config_value is None:
            log.warning(
                f"[bold yellow]{key.capitalize()} not found in config for[/bold yellow] [bold blue] {self.hf_name}[/bold blue]"
            )
        elif model_value is not None and config_value is None:
            setattr(self, key, int(model_value))
            log.info(
                f"[bold green]{key.capitalize()} not found in config, using hf config:[/bold green] [bold blue] {model_value}[/bold blue]"
            )
        elif model_value is None and config_value is not None:
            log.warning(
                f"[bold yellow]{key.capitalize()} not found in hf config for[/bold yellow {self.hf_name}"
            )
        elif model_value is not None and config_value is not None:
            if model_value != config_value:
                log.error(
                    f"[bold red]{key.capitalize()} mismatch: hf config:[/bold red] [bold blue] {model_value}[/bold blue] [bold red]!= config:[/bold red] [bold blue] {config_value}[/bold blue]"
                )
                raise ValueError(
                    f"{key.capitalize()} mismatch: hf config:[/bold red] [bold blue] {model_value}[/bold blue] [bold red]!= config:[/bold red] [bold blue] {config_value}[/bold blue]"
                )
            else:
                log.info(
                    f"[bold green]{key.capitalize()} verified: hf config:[/bold green] [bold blue] {model_value}[/bold blue] [bold green]== config:[/bold green] [bold blue] {config_value}[/bold blue]"
                )
