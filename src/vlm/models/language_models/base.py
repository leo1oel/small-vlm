import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
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
        self.hidden_size: int | None = getattr(self.config, "hidden_size", None)
        self.vocab_size: int | None = getattr(self.config, "vocab_size", None)
        self.max_seq_length: int | None = getattr(self.config, "max_seq_length", None)
        self.output_layer: int = getattr(self.config, "output_layer", -1)
        self.image_token: str | None = getattr(self.config, "image_token", None)
        self.pad_token: str = getattr(self.config, "pad_token", "PAD")
        self.tokenizer: AutoTokenizer = self.build_tokenizer()
        self.language_model: AutoModel = self.build_language_model()
        self.hf_config: AutoConfig = self.build_hf_config()
        self.image_token_id: int | None = (
            self.add_image_token(self.image_token) if self.image_token else None
        )
        self.pad_token_id: int = self.add_pad_token(self.pad_token)
        self.embeddings: nn.Embedding = self.build_embeddings()
        self.verify_config()
        self.transform: Callable[
            [list[dict[str, str]], int, bool], tuple[torch.Tensor, torch.Tensor]
        ] = self._build_transform()

    def add_image_token(self, image_token: str) -> int:
        log.info(f"[bold green]Adding image token: {image_token}[/bold green]")
        self.tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})  # pyright: ignore
        image_token_id: int = self.tokenizer.convert_tokens_to_ids(image_token)  # pyright: ignore
        self.language_model.resize_token_embeddings(len(self.tokenizer))  # pyright: ignore
        return image_token_id  # pyright: ignore

    def add_pad_token(self, pad_token: str) -> int:
        log.info(f"[bold green]Adding pad token: {pad_token}[/bold green]")
        self.tokenizer.add_special_tokens({"pad_token": pad_token})  # pyright: ignore
        pad_token_id: int = self.tokenizer.convert_tokens_to_ids(pad_token)  # pyright: ignore
        self.language_model.resize_token_embeddings(len(self.tokenizer))  # pyright: ignore
        return pad_token_id  # pyright: ignore

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

    def _build_transform(
        self,
    ) -> Callable[[list[dict[str, str]], int, bool], tuple[torch.Tensor, torch.Tensor]]:
        def transform(
            text: list[dict[str, str]], image_token_size: int, generation: bool = False
        ) -> tuple[torch.Tensor, torch.Tensor]:
            conversation = []
            for item in text:
                role = "user" if item["from"] == "human" else "assistant"
                conversation.append({"role": role, "content": item["value"]})
            input_ids: torch.Tensor = self.tokenizer.apply_chat_template(  # pyright: ignore
                conversation,
                tokenize=True,
                add_generation_prompt=generation,
                return_tensors="pt",
                padding=False,
                truncation=True,
            )[0]
            print(
                self.tokenizer.apply_chat_template(  # pyright: ignore
                    conversation,
                    tokenize=False,
                    add_generation_prompt=generation,
                    padding=False,
                    truncation=True,
                )
            )
            labels: torch.Tensor = torch.full_like(input_ids, -100)  # pyright: ignore
            labels[:-1] = input_ids[1:].clone()

            image_token_positions: list[int] = []
            for i, token_id in enumerate(input_ids):  # pyright: ignore
                if token_id == self.image_token_id:
                    image_token_positions.append(i)

            assistant_ranges: list[tuple[int, int]] = []
            in_assistant: bool = False
            start_idx: int | None = None

            for i, token_id in enumerate(input_ids):  # pyright: ignore
                token = self.tokenizer.decode([token_id])  # pyright: ignore

                if "<|assistant|>" in token:
                    in_assistant = True
                    start_idx = i
                elif "<|end|>" in token and in_assistant:
                    if start_idx is not None:
                        assistant_ranges.append((start_idx, i - 1))
                    in_assistant = False
                    start_idx = None

            for i in range(len(labels)):  # pyright: ignore
                is_in_assistant_range = any(start <= i <= end for start, end in assistant_ranges)
                if not is_in_assistant_range:
                    labels[i] = -100

            expanded_labels = []

            for i, token_id in enumerate(input_ids):  # pyright: ignore
                expanded_labels.append(labels[i].item())

                if token_id == self.image_token_id:
                    expanded_labels.extend([-100] * (image_token_size - 1))

            return (input_ids, torch.tensor(expanded_labels))  # pyright: ignore

        return transform

    @abstractmethod
    @override
    def forward(
        self,
        input_ids: None | torch.Tensor = None,
        inputs_embeds: None | torch.Tensor = None,
        attention_mask: None | torch.Tensor = None,
    ) -> torch.Tensor:
        pass

    def verify_config(self) -> None:
        model_hidden_size: int | str | None = self.get_config("hidden_size")
        model_vocab_size: int | str | None = self.get_config("vocab_size")
        model_max_seq_length: int | str | None = self.get_config("max_position_embeddings")

        self.verify_equal("hidden_size", model_hidden_size, self.hidden_size)
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
