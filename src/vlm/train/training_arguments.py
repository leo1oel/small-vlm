from dataclasses import dataclass

import transformers

# from typing import Any
# from transformers.debug_utils import DebugOption
# from transformers.trainer_utils import FSDPOption, HubStrategy, IntervalStrategy, SchedulerType
# from transformers.training_args import OptimizerNames
from ..config import TrainerConfig


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    Training arguments for the VLM model.
    Inherits from HuggingFace's TrainingArguments.
    """

    # # Output
    # output_dir: str | None = None
    # overwrite_output_dir: bool = False

    # # Training modes
    # do_train: bool = True
    # do_eval: bool = False
    # do_predict: bool = False

    # # Evaluation
    # eval_strategy: IntervalStrategy | str = "no"
    # prediction_loss_only: bool = False

    # # Batch sizes
    # per_device_train_batch_size: int = 8
    # per_device_eval_batch_size: int = 8
    # per_gpu_train_batch_size: int | None = None
    # per_gpu_eval_batch_size: int | None = None

    # # Gradient and evaluation
    # gradient_accumulation_steps: int = 1
    # eval_accumulation_steps: int | None = None
    # eval_delay: float | None = 0
    # torch_empty_cache_steps: int | None = None

    # # Optimizer parameters
    # learning_rate: float = 5e-05
    # weight_decay: float = 0.0
    # adam_beta1: float = 0.9
    # adam_beta2: float = 0.999
    # adam_epsilon: float = 1e-08
    # max_grad_norm: float = 1.0

    # # Training length
    # num_train_epochs: float = 3.0
    # max_steps: int = -1

    # # Learning rate scheduler
    # lr_scheduler_type: SchedulerType | str = "linear"
    # lr_scheduler_kwargs: dict[str, Any] | str | None = None
    # warmup_ratio: float = 0.0
    # warmup_steps: int = 0

    # # Logging
    # log_level: str | None = "passive"
    # log_level_replica: str | None = "warning"
    # log_on_each_node: bool = True
    # logging_dir: str | None = None
    # logging_strategy: IntervalStrategy | str = "steps"
    # logging_first_step: bool = False
    # logging_steps: float = 500
    # logging_nan_inf_filter: bool = True

    # # Saving
    # save_strategy: str | str = "steps"
    # save_steps: float = 500
    # save_total_limit: int | None = None
    # save_safetensors: bool | None = True
    # save_on_each_node: bool = False
    # save_only_model: bool = False
    # restore_callback_states_from_checkpoint: bool = False

    # # Hardware
    # no_cuda: bool = False
    # use_cpu: bool = False
    # use_mps_device: bool = False

    # # Reproducibility
    # seed: int = 42
    # data_seed: int | None = None

    # # Precision and performance
    # jit_mode_eval: bool = False
    # use_ipex: bool = False
    # bf16: bool = False
    # fp16: bool = False
    # fp16_opt_level: str = "O1"
    # half_precision_backend: str = "auto"
    # bf16_full_eval: bool = False
    # fp16_full_eval: bool = False
    # tf32: bool | None = None

    # # Distributed training
    # local_rank: int = -1
    # ddp_backend: str | None = None
    # tpu_num_cores: int | None = None
    # tpu_metrics_debug: bool = False
    # debug: str | list[DebugOption] = ""

    # # Dataloader
    # dataloader_drop_last: bool = False
    # eval_steps: float | None = None
    # dataloader_num_workers: int = 0
    # dataloader_prefetch_factor: int | None = None
    # past_index: int = -1

    # # Misc training arguments
    # run_name: str | None = None
    # disable_tqdm: bool | None = None
    # remove_unused_columns: bool | None = True
    # label_names: list[str] | None = None
    # load_best_model_at_end: bool | None = False
    # metric_for_best_model: str | None = None
    # greater_is_better: bool | None = None
    # ignore_data_skip: bool = False

    # # FSDP
    # fsdp: list[FSDPOption] | str | None = ""
    # fsdp_min_num_params: int = 0
    # fsdp_config: dict[str, Any] | str | None = None
    # tp_size: int | None = 0
    # fsdp_transformer_layer_cls_to_wrap: str | None = None

    # # Accelerator
    # accelerator_config: dict[str, Any] | str | None = None
    # deepspeed: dict[str, Any] | str | None = None

    # # Optimizer
    # label_smoothing_factor: float = 0.0
    # optim: OptimizerNames | str = "adamw_torch"
    # optim_args: str | None = None
    # adafactor: bool = False

    # # Grouping
    # group_by_length: bool = False
    # length_column_name: str | None = "length"

    # # Reporting
    # report_to: None | str | list[str] = None

    # # DDP
    # ddp_find_unused_parameters: bool | None = None
    # ddp_bucket_cap_mb: int | None = None
    # ddp_broadcast_buffers: bool | None = None

    # # Dataloader options
    # dataloader_pin_memory: bool = True
    # dataloader_persistent_workers: bool = False

    # # Memory metrics
    # skip_memory_metrics: bool = True
    # use_legacy_prediction_loop: bool = False

    # # Hugging Face Hub
    # push_to_hub: bool = False
    # resume_from_checkpoint: str | None = None
    # hub_model_id: str | None = None
    # hub_strategy: HubStrategy | str = "every_save"
    # hub_token: str | None = None
    # hub_private_repo: bool | None = None
    # hub_always_push: bool = False

    # # Gradient checkpointing
    # gradient_checkpointing: bool = False
    # gradient_checkpointing_kwargs: dict[str, Any] | str | None = None

    # # Metrics
    # include_inputs_for_metrics: bool = False
    # include_for_metrics: list[str] = []
    # eval_do_concat_batches: bool = True

    # # FP16
    # fp16_backend: str = "auto"

    # # Hub legacy
    # push_to_hub_model_id: str | None = None
    # push_to_hub_organization: str | None = None
    # push_to_hub_token: str | None = None

    # # Mixed precision
    # mp_parameters: str = ""

    # # Batch size finding
    # auto_find_batch_size: bool = False
    # full_determinism: bool = False

    # # TorchDynamo
    # torchdynamo: str | None = None
    # ray_scope: str | None = "last"

    # # DDP timeout
    # ddp_timeout: int | None = 1800

    # # Torch compile
    # torch_compile: bool = False
    # torch_compile_backend: str | None = None
    # torch_compile_mode: str | None = None

    # # Tokens
    # include_tokens_per_second: bool | None = False
    # include_num_input_tokens_seen: bool | None = False

    # # NEFTune
    # neftune_noise_alpha: float | None = None

    # # Target modules
    # optim_target_modules: None | str | list[str] = None

    # # Evaluation
    # batch_eval_metrics: bool = False
    # eval_on_start: bool = False

    # # Kernel
    # use_liger_kernel: bool | None = False
    # eval_use_gather_object: bool | None = False
    # average_tokens_across_devices: bool | None = False

    # VLM specific
    unfreeze_language_model: bool = False
    unfreeze_connector: bool = False
    unfreeze_visual_encoder: bool = False
    language_model_lr: float | None = None
    connector_lr: float | None = None
    visual_encoder_lr: float | None = None
    group_by_modality_length: bool = False
    language_model_wd: float | None = None
    connector_wd: float | None = None
    visual_encoder_wd: float | None = None
    version: str = "v0"


def get_training_args(config: TrainerConfig) -> TrainingArguments:
    return TrainingArguments(
        remove_unused_columns=False,
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        deepspeed=config.deepspeed,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        learning_rate=config.learning_rate.default_lr,
        weight_decay=config.weight_decay.default_wd,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        tf32=config.tf32,
        bf16=config.bf16,
        fp16=config.fp16,
        dataloader_num_workers=config.dataloader_num_workers,
        report_to=config.report_to,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        unfreeze_language_model=config.unfreeze.train_language_model,
        unfreeze_connector=config.unfreeze.train_connector,
        unfreeze_visual_encoder=config.unfreeze.train_visual_encoder,
        language_model_lr=config.learning_rate.language_model_learning_rate,
        connector_lr=config.learning_rate.connector_learning_rate,
        visual_encoder_lr=config.learning_rate.visual_encoder_learning_rate,
        group_by_modality_length=config.group_by_modality_length,
        language_model_wd=config.weight_decay.language_model_weight_decay,
        connector_wd=config.weight_decay.connector_weight_decay,
        visual_encoder_wd=config.weight_decay.visual_encoder_weight_decay,
        version=config.version,
        gradient_checkpointing=config.gradient_checkpointing,
        run_name=config.run_name,
        resume_from_checkpoint=config.resume_from_checkpoint,
        seed=config.seed,
        data_seed=config.seed,
        # disable_tqdm=True,
    )
