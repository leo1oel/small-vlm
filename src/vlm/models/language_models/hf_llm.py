import logging
from typing import Any, cast, override

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from ...config.config_schema import LLMConfig
from .base import LanguageModel

log: logging.Logger = logging.getLogger(name=__name__)


class HFLLMLanguageModel(LanguageModel):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)

    @override
    def _build_embedding_layer(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    @override
    def _build_tokenizer(self) -> PreTrainedTokenizer:
        return cast(
            PreTrainedTokenizer,
            AutoTokenizer.from_pretrained(
                self.hf_name,
                trust_remote_code=True,
                use_fast=True,
                device="cuda" if torch.cuda.is_available() else "cpu",
            ),
        )

    @override
    def _build_language_model(self) -> PreTrainedModel:
        return cast(
            PreTrainedModel,
            AutoModelForCausalLM.from_pretrained(self.hf_name, trust_remote_code=True),
        )

    @override
    def _build_hf_config(self) -> PretrainedConfig:
        return cast(
            PretrainedConfig, AutoConfig.from_pretrained(self.hf_name, trust_remote_code=True)
        )

    @override
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            outputs: Any = self.language_model(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask
            )
        elif input_ids is not None:
            outputs = self.language_model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            error_msg = "Either input_ids or inputs_embeds must be provided"
            log.error(error_msg)
            raise ValueError(error_msg)
        return outputs[0]
