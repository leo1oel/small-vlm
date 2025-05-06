import logging
from typing import Any, override

import torch
import torch.nn as nn
import transformers

from ..data.data_arguments import DataArguments
from ..data.dataset import make_supervised_data_module
from ..models.model import VLM
from ..utils import conversation as conversation_lib
from .training_arguments import TrainingArguments
from .vlm_trainer import VLMTrainer

log: logging.Logger = logging.getLogger(name=__name__)


class LoggingCallback(transformers.TrainerCallback):
    @override
    def on_log(self, args, state, control, logs=None, **kwargs):  # pyright: ignore
        if state.is_world_process_zero:
            logs = logs or {}
            msg = ", ".join(
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in logs.items()
            )
            log.info(msg)


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
    log.info("Using gradient checkpointing")
    if training_args.gradient_checkpointing:

        def make_inputs_require_grad(module: nn.Module, input: Any, output: Any) -> None:  # pyright: ignore
            output.requires_grad_(True)

        model.language_model.embeddings.register_forward_hook(make_inputs_require_grad)

    conversation_lib.default_conversation = conversation_lib.conv_templates[training_args.version]

    tokenizer = model.language_model.tokenizer

    log.info("Creating data module")
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    torch_dtype = (
        torch.float32
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    model.language_model.language_model.to(dtype=torch_dtype, device=training_args.device)
    model.visual_encoder.visual_encoder.to(dtype=torch_dtype, device=training_args.device)
    model.connector.to(dtype=torch_dtype, device=training_args.device)

    log.info("Creating trainer")
    trainer = VLMTrainer(model=model, processing_class=tokenizer, args=training_args, **data_module)
    trainer.add_callback(LoggingCallback())
    log.info(f"Trainer: {trainer}")

    if training_args.resume_from_checkpoint:
        log.info(f"Resuming from checkpoint: {training_args.output_dir}")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        log.info("Training from scratch")
        trainer.train()

    log.info("Saving state")
    trainer.save_state()

    log.info("Saving model")
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
