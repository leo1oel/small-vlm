import logging
import pathlib
from typing import Any

import torch.nn as nn

from ..data.data_arguments import DataArguments
from ..data.dataset import make_supervised_data_module
from ..models.model import VLM
from ..utils import conversation as conversation_lib
from .training_arguments import TrainingArguments
from .vlm_trainer import VLMTrainer

log: logging.Logger = logging.getLogger(name=__name__)


def train(model: VLM, training_args: TrainingArguments, data_args: DataArguments):
    if training_args.gradient_checkpointing:

        def make_inputs_require_grad(module: nn.Module, input: Any, output: Any) -> None:  # pyright: ignore
            output.requires_grad_(True)

        model.language_model.embeddings.register_forward_hook(make_inputs_require_grad)

    conversation_lib.default_conversation = conversation_lib.conv_templates[training_args.version]

    tokenizer = model.language_model.tokenizer
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = VLMTrainer(model=model, processing_class=tokenizer, args=training_args, **data_module)
    log.info(f"Trainer: {trainer}")

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
