"""The trainer supporting multiple metrics record."""

import weakref
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import torch
from numpy import typing as npt
from torch.nn import Module
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils.data import Dataset, IterableDataset
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction, speed_metrics
from transformers.training_args import TrainingArguments

Number = int | float


class AdditionalState:
    def __init__(self, args: TrainingArguments) -> None:
        self.metrics: dict[str, list[Number | torch.Tensor | npt.NDArray]] = defaultdict(list)
        self.args = weakref.ref(args)

    def add_metrics(self, **metrics: Number | torch.Tensor | npt.NDArray):
        for k, v in metrics.items():
            self.metrics[k].append(v)

    def get_metrics(
        self,
        step_scale: float = 1.0,
        gather_func: Callable[[torch.Tensor | list[torch.Tensor]], torch.Tensor] | None = None,
        round_digits: int | None = None,
    ) -> dict[str, Number]:
        metrics: dict[str, list[Number]] = defaultdict(list)
        for k, values in self.metrics.items():
            for value in values:
                if isinstance(value, torch.Tensor):
                    if gather_func is not None:
                        value = gather_func(value).mean().item()
                    else:
                        value = value.mean().cpu().item()
                    val = value
                    val = val / self.args().gradient_accumulation_steps
                elif isinstance(value, int | float):
                    val = value
                    val = val / self.args().gradient_accumulation_steps
                elif isinstance(value, np.ndarray):
                    val = value.mean().item()
                    val = val / self.args().gradient_accumulation_steps
                else:
                    val = value
                metrics[k].append(val)

        step_metrics = {
            k: sum(v) / (len(v) / self.args().gradient_accumulation_steps)
            for k, v in metrics.items()
        }
        if round_digits is not None:
            step_metrics = {k: round(v, round_digits) for k, v in step_metrics.items()}

        return step_metrics

    def pop_metrics(
        self,
        gather_func: Callable[[torch.Tensor | list[torch.Tensor]], torch.Tensor] | None = None,
        round_digits: int | None = None,
    ):
        ret = self.get_metrics(gather_func, round_digits)

        self.clear()

        return ret

    def clear(self):
        self.metrics.clear()


class MultiTaskModuleMixin:
    def report_metrics(
        self,
        state: AdditionalState,
        **metrics: Number | torch.Tensor | npt.NDArray,
    ):
        state.add_metrics(**metrics)


DataCollator = Callable[[list[Any]], dict[str, Any]]


def _patching_module_base(module: Module, additional_state: AdditionalState):
    if (
        isinstance(module, Module)
        and hasattr(module, "supports_report_metrics")
        and module.supports_report_metrics
        and MultiTaskModuleMixin not in module.__class__.__bases__
    ):
        module.__class__.__bases__ = module.__class__.__bases__ + (MultiTaskModuleMixin,)
        module.report_metrics = partial(module.report_metrics, additional_state)


class MultiTaskTrainer(Trainer):
    def __init__(
        self,
        model: PreTrainedModel | Module = None,
        args: TrainingArguments = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        model_init: Callable[[], PreTrainedModel] | None = None,
        compute_loss_func: Callable | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        | None = None,
        # grad_norm_weighter: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.additional_state = AdditionalState(args)
        if model is not None:
            report_patching = partial(_patching_module_base, additional_state=self.additional_state)
            model.apply(report_patching)
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        # self.grad_norm_weighter = grad_norm_weighter

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None:
                logs.update(
                    speed_metrics("train", start_time, num_tokens=self.state.num_input_tokens_seen)
                )

        additional_logs = (
            self.additional_state.pop_metrics(gather_func=self._nested_gather)
            if hasattr(self, "additional_state")
            else dict()
        )

        epoch = logs.pop("epoch", None)
        logs.update(additional_logs)
        logs["epoch"] = epoch

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    # def create_accelerator_and_postprocess(self):
    #     super().create_accelerator_and_postprocess()
    #     self.grad_norm_weighter.accelerator = self.accelerator

    # def training_step(
    #     self,
    #     model: nn.Module,
    #     inputs: dict[str, Union[torch.Tensor, Any]],
    #     num_items_in_batch: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #     # Prepare buffers for context parallelism
    #     cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)

    #     # Context manager is no-op if CP isn't enabled
    #     with cp_context():
    #         model.train()
    #         if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
    #             self.optimizer.train()

    #         inputs = self._prepare_inputs(inputs)
    #         if is_sagemaker_mp_enabled():
    #             loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
    #             return loss_mb.reduce_mean().detach().to(self.args.device)

    #         with self.compute_loss_context_manager():
    #             loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

    #         del inputs
    #         if (
    #             self.args.torch_empty_cache_steps is not None
    #             and self.state.global_step % self.args.torch_empty_cache_steps == 0
    #         ):
    #             if is_torch_xpu_available():
    #                 torch.xpu.empty_cache()
    #             elif is_torch_mlu_available():
    #                 torch.mlu.empty_cache()
    #             elif is_torch_musa_available():
    #                 torch.musa.empty_cache()
    #             elif is_torch_npu_available():
    #                 torch.npu.empty_cache()
    #             elif is_torch_mps_available():
    #                 torch.mps.empty_cache()
    #             elif is_torch_hpu_available():
    #                 logger.warning(
    #                     "`torch_empty_cache_steps` is set but HPU device/backend does not support empty_cache()."
    #                 )
    #             else:
    #                 torch.cuda.empty_cache()

    #         kwargs = {}

    #         # For LOMO optimizers you need to explicitly use the learning rate
    #         if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
    #             kwargs["learning_rate"] = self._get_learning_rate()

    #         if self.args.n_gpu > 1:
    #             loss = loss.mean()  # mean() to average on multi-gpu parallel training

    #         if self.use_apex:
    #             from apex import amp

    #             with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #                 scaled_loss.backward()
    #         else:
    #             # Finally we need to normalize the loss for reporting if GA loss bug is not fixed during compute loss
    #             if (
    #                 not self.model_accepts_loss_kwargs or num_items_in_batch is None
    #             ) and self.compute_loss_func is None:
    #                 # If the model does not accept loss kwargs, we need to normalize the loss by the number of gradient accumulation steps
    #                 loss = loss / self.current_gradient_accumulation_steps

    #             # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
    #             # https://github.com/huggingface/transformers/pull/35808
    #             if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
    #                 kwargs["scale_wrt_gas"] = False

    #             if self.grad_norm_weighter:
    #                 self.grad_norm_weighter.backward(loss, **kwargs)
    #             else:
    #                 self.accelerator.backward(loss, **kwargs)
    #         return loss.detach()

    # def compute_loss(
    #     self,
    #     model: nn.Module,
    #     inputs: dict[str, Union[torch.Tensor, Any]],
    #     return_outputs: bool = False,
    #     num_items_in_batch: Optional[torch.Tensor] = None,
    # ):
    #     if self.grad_norm_weighter:
    #         outputs = model(**inputs)

    #         try:
    #             loss_a = outputs["loss_task_a"]
    #             loss_b = outputs["loss_task_b"]
    #             shared_activations = outputs["shared_activations"]
    #             losses_list = [loss_a, loss_b]
    #         except KeyError as e:
    #             raise KeyError(
    #                 f"Model output is missing a required key for GradNorm: {e}. "
    #                 "Ensure your model's forward pass returns a dictionary with all individual losses and 'shared_activations'."
    #             )

    #         # 3. Call the weighter. It handles the backward pass internally.
    #         total_weighted_loss = self.grad_norm_weighter(
    #             losses=losses_list, activations=shared_activations
    #         )

    #         # 4. Return the final weighted loss, which the Trainer uses for logging.
    #         return (total_weighted_loss, outputs) if return_outputs else total_weighted_loss

    #     else:
    #         if (
    #             self.label_smoother is not None or self.compute_loss_func is not None
    #         ) and "labels" in inputs:
    #             labels = inputs.pop("labels")
    #         else:
    #             labels = None
    #         if self.model_accepts_loss_kwargs:
    #             kwargs = {}
    #             if num_items_in_batch is not None:
    #                 kwargs["num_items_in_batch"] = num_items_in_batch
    #             inputs = {**inputs, **kwargs}
    #         outputs = model(**inputs)
    #         # Save past state if it exists
    #         # TODO: this needs to be fixed and made cleaner later.
    #         if self.args.past_index >= 0:
    #             self._past = outputs[self.args.past_index]

    #         if labels is not None:
    #             unwrapped_model = self.accelerator.unwrap_model(model)
    #             if _is_peft_model(unwrapped_model):
    #                 model_name = unwrapped_model.base_model.model._get_name()
    #             else:
    #                 model_name = unwrapped_model._get_name()
    #             # User-defined compute_loss function
    #             if self.compute_loss_func is not None:
    #                 loss = self.compute_loss_func(
    #                     outputs, labels, num_items_in_batch=num_items_in_batch
    #                 )
    #             elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
    #                 loss = self.label_smoother(outputs, labels, shift_labels=True)
    #             else:
    #                 loss = self.label_smoother(outputs, labels)
    #         else:
    #             if isinstance(outputs, dict) and "loss" not in outputs:
    #                 raise ValueError(
    #                     "The model did not return a loss from the inputs, only the following keys: "
    #                     f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
    #                 )
    #             # We don't use .loss here since the model may return tuples instead of ModelOutput.
    #             loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    #         if (
    #             self.args.average_tokens_across_devices
    #             and (self.model_accepts_loss_kwargs or self.compute_loss_func)
    #             and num_items_in_batch is not None
    #         ):
    #             loss *= self.accelerator.num_processes

    #         return (loss, outputs) if return_outputs else loss
