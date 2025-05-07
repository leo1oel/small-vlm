import logging
from typing import Any

import torch
import torch.nn as nn
import transformers

from ..data.data_arguments import DataArguments
from ..data.dataset import make_supervised_data_module
from ..models.model import VLM
from ..utils import conversation as conversation_lib

# from ..utils.rich_callback import ProgressCallback
from .training_arguments import TrainingArguments
from .vlm_trainer import VLMTrainer

logger: logging.Logger = logging.getLogger(name=__name__)


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


def train(model: VLM, training_args: TrainingArguments, data_args: DataArguments):
    logger.info("Using gradient checkpointing")
    if training_args.gradient_checkpointing:

        def make_inputs_require_grad(module: nn.Module, input: Any, output: Any) -> None:  # pyright: ignore
            output.requires_grad_(True)

        model.language_model.embeddings.register_forward_hook(make_inputs_require_grad)

    conversation_lib.default_conversation = conversation_lib.conv_templates[training_args.version]

    tokenizer = model.language_model.tokenizer

    logger.info("Creating data module")
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    model.language_model.language_model.config.use_cache = False

    logger.info("Creating trainer")
    trainer = VLMTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        # callbacks=[ProgressCallback],
        **data_module,
    )

    if training_args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {training_args.output_dir}")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        logger.info("Training from scratch")
        trainer.train()

    logger.info("Saving state")
    trainer.save_state()
    model.language_model.language_model.config.use_cache = True

    logger.info("Saving model")
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
