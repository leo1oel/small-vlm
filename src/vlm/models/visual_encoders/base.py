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
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.config: VisualEncoderConfig = config
        self.name: str = self.config.name
        self.hf_name: str = self.config.hf_name
        self.model_type: str = self.config.type
        self.feature_dim: int | None = getattr(self.config, "feature_dim", None)
        self.img_size: int | None = getattr(self.config, "img_size", None)
        self.patch_size: int | None = getattr(self.config, "patch_size", None)
        self.output_layer: int = getattr(self.config, "output_layer", -1)
        self.preprocessor: AutoImageProcessor = self._build_preprocessor()
        self.visual_encoder: AutoModel = self._build_visual_encoder()
        self.hf_config: AutoConfig = self._build_hf_config()
        self.verify_config()

    @abstractmethod
    def _build_preprocessor(self) -> AutoImageProcessor:
        pass

    @abstractmethod
    def _build_visual_encoder(self) -> AutoModel:
        pass

    @abstractmethod
    def _build_hf_config(self) -> AutoConfig:
        pass

    @abstractmethod
    @override
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pass

    def verify_config(self) -> None:
        model_feature_dim: int | None = self.get_output_dim()
        model_img_size: int | None = self.get_img_size()
        model_patch_size: int | None = self.get_patch_size()

        if self.feature_dim is None and model_feature_dim is None:
            log.warning(
                f"[bold yellow]Feature dimension not found in config for {self.hf_name}[/bold yellow]"
            )
        elif self.feature_dim is None and model_feature_dim is not None:
            self.feature_dim = model_feature_dim
            log.info(
                f"[bold green]Feature dimension not found in config, using hf config: {model_feature_dim}[/bold green]"
            )
        elif self.feature_dim is not None and model_feature_dim is None:
            log.info(
                f"[bold green]Feature dimension not found in hf config, using config: {self.feature_dim}[/bold green]"
            )
        elif self.feature_dim is not None and model_feature_dim is not None:
            if self.feature_dim != model_feature_dim:
                log.error(
                    f"[bold red]Feature dimension mismatch: {self.feature_dim} != {model_feature_dim}[/bold red]"
                )
                raise ValueError(
                    f"Feature dimension mismatch: {self.feature_dim} != {model_feature_dim}"
                )
            else:
                log.info(
                    f"[bold green]Feature dimension verified: {self.feature_dim} == {model_feature_dim}[/bold green]"
                )

        if self.img_size is None and model_img_size is None:
            log.warning(
                f"[bold yellow]Image size not found in config for {self.hf_name}[/bold yellow]"
            )
        elif self.img_size is None and model_img_size is not None:
            self.img_size = model_img_size
            log.info(
                f"[bold green]Image size not found in config, using hf config: {model_img_size}[/bold green]"
            )
        elif self.img_size is not None and model_img_size is None:
            log.warning(
                f"[bold yellow]Image size not found in hf config for {self.hf_name}[/bold yellow]"
            )
        elif self.img_size is not None and model_img_size is not None:
            if self.img_size != model_img_size:
                log.error(
                    f"[bold red]Image size mismatch: {self.img_size} != {model_img_size}[/bold red]"
                )
                raise ValueError(f"Image size mismatch: {self.img_size} != {model_img_size}")
            else:
                log.info(
                    f"[bold green]Image size verified: {self.img_size} == {model_img_size}[/bold green]"
                )

        if self.patch_size is None and model_patch_size is None:
            log.warning(
                f"[bold yellow]Patch size not found in config for {self.hf_name}[/bold yellow]"
            )
        elif self.patch_size is None and model_patch_size is not None:
            self.patch_size = model_patch_size
            log.info(
                f"[bold green]Patch size not found in config, using hf config: {model_patch_size}[/bold green]"
            )
        elif self.patch_size is not None and model_patch_size is None:
            log.warning(
                f"[bold yellow]Patch size not found in hf config for {self.hf_name}[/bold yellow]"
            )
        elif self.patch_size is not None and model_patch_size is not None:
            if self.patch_size != model_patch_size:
                log.error(
                    f"[bold red]Patch size mismatch: {self.patch_size} != {model_patch_size}[/bold red]"
                )
                raise ValueError(f"Patch size mismatch: {self.patch_size} != {model_patch_size}")
            else:
                log.info(
                    f"[bold green]Patch size verified: {self.patch_size} == {model_patch_size}[/bold green]"
                )

    @abstractmethod
    def get_output_dim(self) -> int | None:
        pass

    @abstractmethod
    def get_img_size(self) -> int | None:
        pass

    @abstractmethod
    def get_patch_size(self) -> int | None:
        pass
