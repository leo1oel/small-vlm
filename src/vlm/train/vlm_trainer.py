import logging
import os
from typing import Any, override

import datasets
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from transformers.trainer import Trainer, has_length, is_datasets_available
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import LengthGroupedSampler

from ..data import MultiModalLengthGroupedSampler
from .optimizer import configure_optimizers

log = logging.getLogger(__name__)


def _energon_state_path(checkpoint_dir: str, process_index: int) -> str:
    return os.path.join(checkpoint_dir, f"energon_state_rank{process_index}.pt")


class EnergonStateCallback(TrainerCallback):
    """Persists the energon streaming loader's exact-resume state inside every
    HF checkpoint (one file per rank — energon state is rank-local). The
    symmetric restore happens in train.py before trainer.train()."""

    def __init__(self, loader: Any, num_workers: int):
        self.loader: Any = loader
        self.num_workers: int = num_workers

    @override
    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)  # rank0 may not have created it yet
        torch.save(
            {
                "loader_state": self.loader.save_state_rank(),
                # resume-topology invariants: energon state is only valid for
                # the same world_size and num_workers
                "world_size": args.world_size,
                "num_workers": self.num_workers,
            },
            _energon_state_path(checkpoint_dir, args.process_index),
        )


def restore_energon_state(loader: Any, checkpoint_dir: str, args: Any, num_workers: int) -> None:
    """Restore the rank-local energon loader state saved by EnergonStateCallback,
    asserting the resume-topology invariants."""
    path = _energon_state_path(checkpoint_dir, args.process_index)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"checkpoint {checkpoint_dir} has no energon state for rank "
            f"{args.process_index} ({path}) — was it saved with a different "
            "world_size or without the streaming dataset?"
        )
    payload = torch.load(path, weights_only=False)  # own checkpoint; holds dataclasses
    for key, current in (("world_size", args.world_size), ("num_workers", num_workers)):
        if payload[key] != current:
            raise RuntimeError(
                f"energon resume topology mismatch: checkpoint has {key}="
                f"{payload[key]} but the current run has {key}={current}; the "
                "loader state is only valid for an identical topology"
            )
    loader.restore_state_rank(payload["loader_state"])
    log.info(f"Restored energon loader state from {path}")


class VLMTrainer(Trainer):
    def __init__(self, *args: Any, energon_train_loader: Any = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.energon_train_loader: Any = energon_train_loader

    @override
    def get_train_dataloader(self):
        if self.energon_train_loader is not None:
            # Energon owns batching, workers and rank sharding (WorkerConfig
            # reads torch.distributed). Returning it directly bypasses the
            # whole HF DataLoader/accelerator.prepare path — which would
            # otherwise shard the already-sharded stream a second time.
            return self.energon_train_loader
        return super().get_train_dataloader()

    @override
    def _get_train_sampler(self, train_dataset: Any = None) -> torch.utils.data.Sampler | None:
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None or not has_length(train_dataset):
            return None

        if self.args.group_by_modality_length:
            log.info("Using modality length grouped sampling")
            lengths: Any = self.train_dataset.modality_lengths
            return MultiModalLengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        elif self.args.group_by_length:
            log.info("Using length grouped sampling")
            if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
                lengths = (
                    train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = (
                self.processing_class.model_input_names[0]
                if self.processing_class is not None
                else None
            )
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )
        elif self.args.sequential_sampling:
            log.info("Using sequential sampling")
            return SequentialSampler(train_dataset)
        else:
            return RandomSampler(train_dataset)

    @override
    def create_optimizer(self, model: torch.nn.Module | None = None):
        # v5 Trainer calls create_optimizer(model) in the FSDP delay-creation path
        # (trainer.py: self.create_optimizer(model)); accept and honor the arg.
        opt_model = self.model if model is None else model

        if self.optimizer is None:
            optimizer_grouped_parameters = configure_optimizers(opt_model, self.args)

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                self.args, opt_model
            )
            self.optimizer: Any = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer
