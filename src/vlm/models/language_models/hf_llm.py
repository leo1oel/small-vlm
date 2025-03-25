from typing import override

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from ...config.config_schema import LLMConfig
from .base import LanguageModel


class HFLLMLanguageModel(LanguageModel):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)

    @override
    def _build_tokenizer(self) -> AutoTokenizer:
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            self.hf_name, trust_remote_code=True
        )  # pyright: ignore[reportUnknownMemberType]
        return self.tokenizer  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    @override
    def _build_language_model(self) -> AutoModel:
        self.language_model: AutoModel = AutoModelForCausalLM.from_pretrained(
            self.hf_name, trust_remote_code=True
        )  # pyright: ignore[reportUnknownMemberType]
        return self.language_model  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    @override
    def _build_hf_config(self) -> AutoConfig:
        self.hf_config: AutoConfig = AutoConfig.from_pretrained(
            self.hf_name, trust_remote_code=True
        )  # pyright: ignore[reportUnknownMemberType]
        return self.hf_config  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    @override
    def forward(
        self, input_ids: torch.Tensor, attention_mask: None | torch.Tensor = None
    ) -> torch.Tensor:
        outputs = self.language_model(input_ids, attention_mask=attention_mask)  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
        return outputs.last_hidden_state  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    @override
    def get_hidden_dim(self) -> int | None:
        if getattr(self.hf_config, "hidden_size", None) is not None:
            return self.hf_config.hidden_size  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue]
        else:
            return None

    @override
    def get_vocab_size(self) -> int | None:
        if getattr(self.hf_config, "vocab_size", None) is not None:
            return self.hf_config.vocab_size  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue]
        else:
            return None

    @override
    def get_max_seq_length(self) -> int | None:
        if getattr(self.hf_config, "max_position_embeddings", None) is not None:
            return self.hf_config.max_position_embeddings  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue]
        else:
            return None
