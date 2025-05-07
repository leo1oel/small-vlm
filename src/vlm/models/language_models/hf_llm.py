import logging
from typing import cast, override

import torch.nn as nn
from torch import device, dtype
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
    def __init__(self, config: LLMConfig, torch_dtype: dtype, torch_device: device) -> None:
        super().__init__(config, torch_dtype, torch_device)

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
                model_max_length=self.config.max_seq_length,
                use_fast=True,
            ),
        )

    @override
    def _build_language_model(self) -> PreTrainedModel:
        return cast(
            PreTrainedModel,
            AutoModelForCausalLM.from_pretrained(
                self.hf_name,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                attn_implementation=self.config.attn_implementation,
                torch_dtype=self.torch_dtype,
            ).to(device=self.torch_device),
        )

    @override
    def _build_hf_config(self) -> PretrainedConfig:
        return cast(
            PretrainedConfig, AutoConfig.from_pretrained(self.hf_name, trust_remote_code=True)
        )

    # @override
    # def generate(
    #     self,
    #     inputs: Tensor | None = None,
    #     images: FloatTensor | None = None,
    #     image_sizes: list[list[int]] | None = None,
    #     **kwargs,
    # ) -> Any:
    #     position_ids = kwargs.pop("position_ids", None)
    #     attention_mask = kwargs.pop("attention_mask", None)
    #     if "inputs_embeds" in kwargs:
    #         raise NotImplementedError("`inputs_embeds` is not supported")

    #     if images is not None:
    #         (inputs, position_ids, attention_mask, _, inputs_embeds, _) = (
    #             self.prepare_inputs_labels_for_multimodal(
    #                 inputs,
    #                 position_ids,
    #                 attention_mask,
    #                 None,
    #                 None,
    #                 images,
    #                 image_sizes=image_sizes,
    #             )
    #         )
    #     else:
    #         inputs_embeds = self.embeddings(inputs)

    #     return self.language_model.generate(
    #         position_ids=position_ids,
    #         attention_mask=attention_mask,
    #         inputs_embeds=inputs_embeds,
    #         **kwargs,
    #     )
