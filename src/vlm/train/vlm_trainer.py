import logging
from typing import Any, override

import datasets
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from transformers.trainer import has_length, is_datasets_available
from transformers.trainer_pt_utils import LengthGroupedSampler

from ..data import MultiModalLengthGroupedSampler
from .multitask_trainer import MultiTaskTrainer
from .optimizer import configure_optimizers

log = logging.getLogger(__name__)


class VLMTrainer(MultiTaskTrainer):
    @override
    def _get_train_sampler(self, train_dataset=None) -> torch.utils.data.Sampler | None:
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
    def create_optimizer(self):
        opt_model = self.model

        if self.optimizer is None:
            optimizer_grouped_parameters = configure_optimizers(opt_model, self.args)

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                self.args, opt_model
            )
            self.optimizer: Any = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer
