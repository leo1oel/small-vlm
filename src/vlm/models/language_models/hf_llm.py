import logging
from typing import override

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from ...config.config_schema import LLMConfig
from .base import LanguageModel

log: logging.Logger = logging.getLogger(name=__name__)


class HFLLMLanguageModel(LanguageModel):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)

    @override
    def _build_embedding_layer(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()  # pyright: ignore

    @override
    def _build_tokenizer(self) -> AutoTokenizer:
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            self.hf_name, trust_remote_code=True
        )
        return self.tokenizer  # pyright: ignore

    @override
    def _build_language_model(self) -> AutoModel:
        # try:
        #     self.language_model: AutoModel = AutoModelForCausalLM.from_pretrained(
        #         self.hf_name, trust_remote_code=True, attn_implementation="flash_attention_2"
        #     )
        #     log.info("[bold green]Successfully loaded model with flash_attention_2[/bold green]")
        # except Exception as e:
        #     log.warning(
        #         f"[bold yellow]Failed to load model with flash_attention_2: {e} Falling back to sdpa implementation...[/bold yellow]"
        #     )
        #     self.language_model = AutoModelForCausalLM.from_pretrained(
        #         self.hf_name, trust_remote_code=True, attn_implementation="sdpa"
        #     )
        #     log.info("[bold green]Successfully loaded model with sdpa[/bold green]")
        self.language_model: AutoModel = AutoModelForCausalLM.from_pretrained(
            self.hf_name, trust_remote_code=True
        )
        log.info("[bold green]Successfully loaded model[/bold green]")
        return self.language_model  # pyright: ignore

    @override
    def _build_hf_config(self) -> AutoConfig:
        self.hf_config: AutoConfig = AutoConfig.from_pretrained(
            self.hf_name, trust_remote_code=True
        )
        return self.hf_config  # pyright: ignore

    @override
    def forward(
        self,
        input_ids: None | torch.Tensor = None,
        inputs_embeds: None | torch.Tensor = None,
        attention_mask: None | torch.Tensor = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            outputs = self.language_model(  # pyright: ignore
                inputs_embeds=inputs_embeds, attention_mask=attention_mask
            )
        elif input_ids is not None:
            outputs = self.language_model(input_ids, attention_mask=attention_mask)  # pyright: ignore
        else:
            log.error("[bold red]Either input_ids or inputs_embeds must be provided[/bold red]")
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        return outputs[0]  # pyright: ignore
