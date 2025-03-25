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
        super().__init__()
        self.config: LLMConfig = config
        self.name: str = self.config.name
        self.hf_name: str = self.config.hf_name
        self.model_type: str = self.config.type
        self.hidden_dim: int | None = getattr(self.config, "hidden_dim", None)
        self.vocab_size: int | None = getattr(self.config, "vocab_size", None)
        self.max_seq_length: int | None = getattr(self.config, "max_seq_length", None)
        self.output_layer: int = getattr(self.config, "output_layer", -1)
        self.image_token: str | None = getattr(self.config, "image_token", None)
        self.tokenizer: AutoTokenizer = self.build_tokenizer()
        self.language_model: AutoModel = self.build_language_model()
        self.hf_config: AutoConfig = self.build_hf_config()
        self.image_token_id: int | None = (
            self.add_image_token(self.image_token) if self.image_token else None
        )
        self.embeddings: nn.Embedding = self.build_embeddings()
        self.verify_config()

    def add_image_token(self, image_token: str) -> int:
        log.info(f"[bold green]Adding image token: {image_token}[/bold green]")
        self.tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})  # pyright: ignore
        image_token_id: int = self.tokenizer.convert_tokens_to_ids(image_token)  # pyright: ignore
        self.language_model.resize_token_embeddings(len(self.tokenizer))  # pyright: ignore
        return image_token_id  # pyright: ignore

    @abstractmethod
    def _build_embedding_layer(self) -> nn.Embedding:
        pass

    def build_embeddings(self) -> nn.Embedding:
        log.info(f"[bold green]Building embeddings for {self.hf_name}[/bold green]")
        return self._build_embedding_layer()

    @abstractmethod
    def _build_tokenizer(self) -> AutoTokenizer:
        pass

    def build_tokenizer(self) -> AutoTokenizer:
        log.info(
            f"[bold green]Building tokenizer for[/bold green] [bold blue] {self.hf_name}[/bold blue]"
        )
        return self._build_tokenizer()

    @abstractmethod
    def _build_language_model(self) -> AutoModel:
        pass

    def build_language_model(self) -> AutoModel:
        log.info(
            f"[bold green]Building language model for[/bold green] [bold blue] {self.hf_name}[/bold blue]"
        )
        return self._build_language_model()

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
    def forward(
        self,
        input_ids: None | torch.Tensor = None,
        input_embeds: None | torch.Tensor = None,
        attention_mask: None | torch.Tensor = None,
    ) -> torch.Tensor:
        pass

    def verify_config(self) -> None:
        model_hidden_dim: int | str | None = self.get_config("hidden_size")
        model_vocab_size: int | str | None = self.get_config("vocab_size")
        model_max_seq_length: int | str | None = self.get_config("max_position_embeddings")

        self.verify_equal("hidden_dim", model_hidden_dim, self.hidden_dim)
        self.verify_equal("vocab_size", model_vocab_size, self.vocab_size)
        self.verify_equal("max_seq_length", model_max_seq_length, self.max_seq_length)

    def get_config(self, key: str) -> int | str | None:
        if getattr(self.hf_config, key, None) is not None:
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
                f"[bold yellow]{key.capitalize()} not found in hf config for[/bold yellow] [bold blue] {self.hf_name}[/bold blue]"
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
