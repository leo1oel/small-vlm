import logging
from typing import Any, cast, override

import torch.nn as nn
from torch import FloatTensor, LongTensor, device, dtype
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.utils import ModelOutput

from ...config.config_schema import LLMConfig
from .base import LanguageModel

log: logging.Logger = logging.getLogger(name=__name__)


class HFLLMLanguageModel(LanguageModel):
    def __init__(
        self, config: LLMConfig, torch_dtype: dtype, torch_device: device, attn_implementation: str
    ) -> None:
        super().__init__(config, torch_dtype, torch_device, attn_implementation)

    @override
    def get_embedding_layer(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    @override
    def _build_tokenizer(self) -> PreTrainedTokenizer:
        return cast(
            PreTrainedTokenizer,
            AutoTokenizer.from_pretrained(
                self.hf_name,
                trust_remote_code=True,
                model_max_length=self.config.max_seq_length,
                use_fast=True,
            ),
        )

    @override
    def _build_language_model(self) -> PreTrainedModel:
        llm = cast(
            PreTrainedModel,
            AutoModelForCausalLM.from_pretrained(
                self.hf_name,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                attn_implementation=self.attn_implementation,
                torch_dtype=self.torch_dtype,
            ),
        )
        if llm.device == device("meta"):
            llm.to_empty(device=self.torch_device)
        else:
            llm.to(device=self.torch_device)
        return llm

    @override
    def _build_hf_config(self) -> PretrainedConfig:
        return cast(
            PretrainedConfig, AutoConfig.from_pretrained(self.hf_name, trust_remote_code=True)
        )

    @override
    def generate(
        self,
        position_ids: LongTensor | None = None,
        attention_mask: LongTensor | None = None,
        inputs_embeds: FloatTensor | None = None,
        **kwargs: Any,
    ) -> ModelOutput | LongTensor:
        return self.language_model.generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
