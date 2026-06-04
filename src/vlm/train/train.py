import logging
from typing import Any

import torch
import torch.nn as nn
import transformers

from ..models import VLMProcessor
from ..utils import conversation as conversation_lib
from .set_trainable import set_trainable_params
from .training_arguments import TrainingArguments
from .vlm_trainer import EnergonStateCallback, VLMTrainer, restore_energon_state

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


def _validate_energon_args(training_args: TrainingArguments) -> None:
    """The streaming loader has no epoch length and does its own sampling, so a
    few HF Trainer features are structurally unavailable — fail loud upfront."""
    if training_args.max_steps <= 0:
        raise ValueError(
            "dataset.type='energon' requires trainer.max_steps > 0: the streaming "
            "loader has no epoch length, so scheduling/stopping must be step-based"
        )
    for flag in ("group_by_length", "group_by_modality_length", "sequential_sampling"):
        if getattr(training_args, flag, False):
            raise ValueError(
                f"trainer.{flag} is incompatible with dataset.type='energon' "
                "(no map-style dataset to sample from); use energon blend "
                "weights / shuffle_buffer_size instead"
            )
    if training_args.dataloader_num_workers:
        log.info(
            "dataset.type='energon': trainer.dataloader_num_workers is ignored — "
            "workers belong to the energon loader (dataset.num_workers)"
        )
    if not training_args.ignore_data_skip:
        # On resume, HF would re-iterate the stream via skip_first_batches to
        # reach the previous position; the energon loader state restore already
        # does that exactly and for free.
        training_args.ignore_data_skip = True
        log.info("dataset.type='energon': forcing ignore_data_skip=True (exact loader resume)")


def train(
    model: Any,
    training_args: TrainingArguments,
    data_module: Any,
    processor: VLMProcessor,
    energon_loader: Any = None,
    energon_num_workers: int = 0,
):
    log.info("Using gradient checkpointing")
    if training_args.gradient_checkpointing:

        def make_inputs_require_grad(module: nn.Module, input: Any, output: Any) -> None:
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    conversation_lib.default_conversation = conversation_lib.conv_templates[training_args.version]

    model.config.use_cache = False
    set_trainable_params(model, training_args)

    if energon_loader is not None:
        _validate_energon_args(training_args)

    log.info("Creating trainer")
    trainer = VLMTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        energon_train_loader=energon_loader,
        **data_module,
    )
    if energon_loader is not None:
        trainer.add_callback(EnergonStateCallback(energon_loader, energon_num_workers))

    import os

    from transformers.trainer_utils import get_last_checkpoint

    if training_args.resume_from_checkpoint:
        resume_ckpt = training_args.resume_from_checkpoint
        log.info(f"Resuming from checkpoint: {resume_ckpt}")
    else:
        resume_ckpt = None
        if os.path.isdir(training_args.output_dir):
            resume_ckpt = get_last_checkpoint(training_args.output_dir)
        if resume_ckpt is not None:
            log.info(f"Auto-resuming from last checkpoint: {resume_ckpt} (requeued job?)")
        else:
            log.info("Training from scratch (no checkpoint in output_dir)")

    if energon_loader is not None and resume_ckpt:
        restore_energon_state(energon_loader, resume_ckpt, training_args, energon_num_workers)

    trainer.train(resume_from_checkpoint=resume_ckpt)

    log.info("Saving state")
    trainer.save_state()
    model.config.use_cache = True

    log.info("Saving model")
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
