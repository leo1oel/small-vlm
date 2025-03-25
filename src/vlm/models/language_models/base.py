import logging
from abc import ABC, abstractmethod
from typing import override

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ...config.config_schema import LLMConfig

log: logging.Logger = logging.getLogger(name=__name__)


class LanguageModel(nn.Module, ABC):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.config: LLMConfig = config
        self.name: str = self.config.name
        self.hf_name: str = self.config.hf_name
        self.model_type: str = self.config.type
        self.hidden_dim: int | None = getattr(self.config, "hidden_dim", None)
        self.vocab_size: int | None = getattr(self.config, "vocab_size", None)
        self.max_seq_length: int | None = getattr(self.config, "max_seq_length", None)
        self.output_layer: int = getattr(self.config, "output_layer", -1)
        self.tokenizer: AutoTokenizer = self._build_tokenizer()
        self.language_model: AutoModel = self._build_language_model()
        self.hf_config: AutoConfig = self._build_hf_config()
        self.verify_config()

    @abstractmethod
    def _build_tokenizer(self) -> AutoTokenizer:
        pass

    @abstractmethod
    def _build_language_model(self) -> AutoModel:
        pass

    @abstractmethod
    def _build_hf_config(self) -> AutoConfig:
        pass

    @abstractmethod
    @override
    def forward(
        self, input_ids: torch.Tensor, attention_mask: None | torch.Tensor = None
    ) -> torch.Tensor:
        pass

    def verify_config(self) -> None:
        model_hidden_dim: int | None = self.get_hidden_dim()
        model_vocab_size: int | None = self.get_vocab_size()
        model_max_seq_length: int | None = self.get_max_seq_length()

        if self.hidden_dim is None and model_hidden_dim is None:
            log.warning(
                f"[bold yellow]Hidden dimension not found in config for {self.hf_name}[/bold yellow]"
            )
        elif self.hidden_dim is None and model_hidden_dim is not None:
            self.hidden_dim = model_hidden_dim
            log.info(
                f"[bold green]Hidden dimension not found in config, using hf config: {model_hidden_dim}[/bold green]"
            )
        elif self.hidden_dim is not None and model_hidden_dim is None:
            log.info(
                f"[bold green]Hidden dimension not found in hf config, using config: {self.hidden_dim}[/bold green]"
            )
        elif self.hidden_dim is not None and model_hidden_dim is not None:
            if self.hidden_dim != model_hidden_dim:
                log.error(
                    f"[bold red]Hidden dimension mismatch: {self.hidden_dim} != {model_hidden_dim}[/bold red]"
                )
                raise ValueError(
                    f"Hidden dimension mismatch: {self.hidden_dim} != {model_hidden_dim}"
                )
            else:
                log.info(
                    f"[bold green]Hidden dimension verified: {self.hidden_dim} == {model_hidden_dim}[/bold green]"
                )

        if self.vocab_size is None and model_vocab_size is None:
            log.warning(
                f"[bold yellow]Vocabulary size not found in config for {self.hf_name}[/bold yellow]"
            )
        elif self.vocab_size is None and model_vocab_size is not None:
            self.vocab_size = model_vocab_size
            log.info(
                f"[bold green]Vocabulary size not found in config, using hf config: {model_vocab_size}[/bold green]"
            )
        elif self.vocab_size is not None and model_vocab_size is None:
            log.warning(
                f"[bold yellow]Vocabulary size not found in hf config for {self.hf_name}[/bold yellow]"
            )
        elif self.vocab_size is not None and model_vocab_size is not None:
            if self.vocab_size != model_vocab_size:
                log.error(
                    f"[bold red]Vocabulary size mismatch: {self.vocab_size} != {model_vocab_size}[/bold red]"
                )
                raise ValueError(
                    f"Vocabulary size mismatch: {self.vocab_size} != {model_vocab_size}"
                )
            else:
                log.info(
                    f"[bold green]Vocabulary size verified: {self.vocab_size} == {model_vocab_size}[/bold green]"
                )

        if self.max_seq_length is None and model_max_seq_length is None:
            log.warning(
                f"[bold yellow]Maximum sequence length not found in config for {self.hf_name}[/bold yellow]"
            )
        elif self.max_seq_length is None and model_max_seq_length is not None:
            self.max_seq_length = model_max_seq_length
            log.info(
                f"[bold green]Maximum sequence length not found in config, using hf config: {model_max_seq_length}[/bold green]"
            )
        elif self.max_seq_length is not None and model_max_seq_length is None:
            log.warning(
                f"[bold yellow]Maximum sequence length not found in hf config for {self.hf_name}[/bold yellow]"
            )
        elif self.max_seq_length is not None and model_max_seq_length is not None:
            if self.max_seq_length != model_max_seq_length:
                log.error(
                    f"[bold red]Maximum sequence length mismatch: {self.max_seq_length} != {model_max_seq_length}[/bold red]"
                )
                raise ValueError(
                    f"Maximum sequence length mismatch: {self.max_seq_length} != {model_max_seq_length}"
                )
            else:
                log.info(
                    f"[bold green]Maximum sequence length verified: {self.max_seq_length} == {model_max_seq_length}[/bold green]"
                )

    @abstractmethod
    def get_hidden_dim(self) -> int | None:
        pass

    @abstractmethod
    def get_vocab_size(self) -> int | None:
        pass

    @abstractmethod
    def get_max_seq_length(self) -> int | None:
        pass
