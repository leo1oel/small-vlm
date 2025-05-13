import logging
from typing import Any

import torch
import torch.nn as nn
import transformers

from ..models import VLMProcessor
from ..utils import conversation as conversation_lib
from .set_trainable import set_trainable_params
from .training_arguments import TrainingArguments
from .vlm_trainer import VLMTrainer

log: logging.Logger = logging.getLogger(name=__name__)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
    del state_dict
    trainer._save(output_dir, state_dict=cpu_state_dict)  # pyright: ignore


def train(model: Any, training_args: TrainingArguments, data_module: Any, processor: VLMProcessor):
    log.info("Using gradient checkpointing")
    if training_args.gradient_checkpointing:

        def make_inputs_require_grad(module: nn.Module, input: Any, output: Any) -> None:
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    conversation_lib.default_conversation = conversation_lib.conv_templates[training_args.version]

    model.config.use_cache = False
    set_trainable_params(model, training_args)

    log.info("Creating trainer")
    trainer = VLMTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        **data_module,
    )

    if training_args.resume_from_checkpoint:
        log.info(f"Resuming from checkpoint: {training_args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        log.info("Training without resuming from checkpoint")
        trainer.train()

    log.info("Saving state")
    trainer.save_state()
    model.config.use_cache = True

    log.info("Saving model")
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
