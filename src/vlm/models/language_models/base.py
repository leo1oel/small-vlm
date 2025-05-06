import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, cast, override

import torch.nn as nn
from torch import FloatTensor, LongTensor, Tensor, device, dtype
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer

from ...config.config_schema import LLMConfig

log: logging.Logger = logging.getLogger(name=__name__)


@dataclass
class TokenConfig:
    """Special token configuration"""

    image_token: str = "<image>"
    image_patch_token: str = "<im_patch>"
    image_start_token: str = "<im_start>"
    image_end_token: str = "<im_end>"
    system_token: str | None = None
    user_token: str | None = None
    assistant_token: str = "<|assistant|>"
    image_token_id: int | None = None
    image_patch_token_id: int | None = None
    image_start_token_id: int | None = None
    image_end_token_id: int | None = None


@dataclass
class LanguageModelConfig:
    hidden_size: int | None = None
    vocab_size: int | None = None
    max_seq_length: int | None = None


class LanguageModel(nn.Module, ABC):
    def __init__(self, config: LLMConfig, torch_dtype: dtype, torch_device: device) -> None:
        super().__init__()
        self.config: LLMConfig = config
        self.name: str = self.config.name
        self.hf_name: str = self.config.hf_name
        self.model_type: str = self.config.type
        self.torch_dtype: dtype = torch_dtype
        self.torch_device: device = torch_device

        # model config
        self.model_config: LanguageModelConfig = LanguageModelConfig(
            hidden_size=getattr(self.config, "hidden_size", None),
            vocab_size=getattr(self.config, "vocab_size", None),
            max_seq_length=getattr(self.config, "max_seq_length", None),
        )

        # token config
        self.token_config: TokenConfig = TokenConfig(
            image_token=getattr(self.config, "image_token", "<image>"),
            image_patch_token=getattr(self.config, "image_patch_token", "<im_patch>"),
            image_start_token=getattr(self.config, "image_start_token", "<im_start>"),
            image_end_token=getattr(self.config, "image_end_token", "<im_end>"),
            system_token=getattr(self.config, "system_token", None),
            user_token=getattr(self.config, "user_token", None),
            assistant_token=getattr(self.config, "assistant_token", "<|assistant|>"),
        )

        self.initialize_components()

    # initialize all components
    def initialize_components(self) -> None:
        self._hf_config: PretrainedConfig = self._build_hf_config()
        self.verify_config()
        self._tokenizer: PreTrainedTokenizer = self._build_tokenizer()
        self._tokenizer.pad_token = self._tokenizer.unk_token
        self._add_special_tokens()
        self._language_model: PreTrainedModel = self._build_language_model()
        self.language_model.resize_token_embeddings(len(self.tokenizer))
        self._embeddings: nn.Module = self._build_embedding_layer()

    def _add_special_tokens(self) -> None:
        """Adds special tokens to the tokenizer if they don't exist."""
        # Create a mapping of tokens to their attribute names

        token_mapping = {
            self.token_config.system_token: "system_token_id",
            self.token_config.user_token: "user_token_id",
            self.token_config.assistant_token: "assistant_token_id",
        }
        if self.config.use_image_patch_token:
            token_mapping[self.token_config.image_patch_token] = "image_patch_token_id"
        if self.config.use_start_end_tokens:
            token_mapping[self.token_config.image_start_token] = "image_start_token_id"
            token_mapping[self.token_config.image_end_token] = "image_end_token_id"

        # Identify which tokens need to be added
        tokens_to_add: list[Any] = []
        for token in token_mapping:
            if token is None:
                continue
            if self.tokenizer.convert_tokens_to_ids(token) == self.tokenizer.unk_token_id:
                tokens_to_add.append(token)
                log.info(f"Token '{token}' does not exist in tokenizer, will be added")
            else:
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                log.info(f"Token '{token}' exists in tokenizer, ID: {token_id}")
                setattr(self.token_config, token_mapping[token], cast(int, token_id))

        # Add all new tokens at once if any
        if tokens_to_add:
            log.info(f"Adding tokens: {tokens_to_add}")
            self.tokenizer.add_tokens(tokens_to_add, special_tokens=True)

            # Now set the IDs for newly added tokens
            for token in tokens_to_add:
                token_id = cast(int, self.tokenizer.convert_tokens_to_ids(token))
                setattr(self.token_config, token_mapping[token], token_id)

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer

    @property
    def language_model(self) -> PreTrainedModel:
        return self._language_model

    @property
    def hf_config(self) -> PretrainedConfig:
        return self._hf_config

    @property
    def hidden_size(self) -> int:
        return cast(int, self.model_config.hidden_size)

    @hidden_size.setter
    def hidden_size(self, value: int) -> None:
        self.model_config.hidden_size = value

    @property
    def vocab_size(self) -> int:
        return cast(int, self.model_config.vocab_size)

    @vocab_size.setter
    def vocab_size(self, value: int) -> None:
        self.model_config.vocab_size = value

    @property
    def max_seq_length(self) -> int:
        return cast(int, self.model_config.max_seq_length)

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        self.model_config.max_seq_length = value

    @property
    def embeddings(self) -> nn.Module:
        return self._embeddings

    @property
    def image_token_id(self) -> int:
        return cast(int, self.token_config.image_token_id)

    @property
    def pad_token_id(self) -> int:
        return cast(int, self.tokenizer.pad_token_id)

    @abstractmethod
    def _build_embedding_layer(self) -> nn.Module:
        pass

    @abstractmethod
    def _build_tokenizer(self) -> PreTrainedTokenizer:
        pass

    @abstractmethod
    def _build_language_model(self) -> PreTrainedModel:
        pass

    @abstractmethod
    def _build_hf_config(self) -> PretrainedConfig:
        pass

    @override
    def forward(
        self,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: LongTensor | None = None,
        past_key_values: list[FloatTensor] | None = None,
        labels: LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        images: FloatTensor | None = None,
        image_sizes: list[list[int]] | None = None,
        return_dict: bool | None = None,
    ) -> Tensor:
        return self.language_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    # @abstractmethod
    # @override
    # def generate(
    #     self,
    #     inputs: Tensor | None = None,
    #     images: FloatTensor | None = None,
    #     image_sizes: list[list[int]] | None = None,
    #     **kwargs,
    # ) -> GenerateOutput | torch.LongTensor:
    #     pass

    def verify_config(self) -> None:
        config_pairs = [
            ("hidden_size", self.get_config("hidden_size"), self.hidden_size),
            ("vocab_size", self.get_config("vocab_size"), self.vocab_size),
            ("max_seq_length", self.get_config("max_position_embeddings"), self.max_seq_length),
        ]

        for key, model_value, config_value in config_pairs:
            self._verify_param_match(key, model_value, config_value)

    def _verify_param_match(
        self, key: str, model_value: int | str | None, config_value: int | str | None
    ) -> None:
        capitalized_key = key.capitalize()

        if model_value is None and config_value is None:
            log.warning(f"{capitalized_key} not found in config for {self.name}")
        elif model_value is not None and config_value is None:
            setattr(self, key, int(model_value))
            if hasattr(self.config, key):
                setattr(self.config, key, int(model_value))
            log.info(f"{capitalized_key} not found in config, using hf config: {model_value}")
        elif model_value is None and config_value is not None:
            log.warning(f"{capitalized_key} not found in hf config for {self.name}")
        elif model_value is not None and config_value is not None:
            if model_value != config_value:
                error_msg = f"{capitalized_key} mismatch: hf config: {model_value} != config: {config_value}"
                log.warning(error_msg)
            else:
                log.info(
                    f"{capitalized_key} verified: hf config: {model_value} == config: {config_value}"
                )

    def get_config(self, key: str) -> int | str | None:
        return getattr(self.hf_config, key, None)
