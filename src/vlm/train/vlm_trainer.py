from typing import Any, override

import torch
from transformers.trainer import Trainer, has_length

from ..data import MultiModalLengthGroupedSampler
from .optimizer import configure_optimizers


class VLMTrainer(Trainer):
    @override
    def _get_train_sampler(self) -> torch.utils.data.Sampler | None:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths: Any = self.train_dataset.modality_lengths
            return MultiModalLengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    @override
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model

        if self.optimizer is None:
            optimizer_grouped_parameters = configure_optimizers(opt_model, self.args)

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                self.args, opt_model
            )
            self.optimizer: Any = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer
