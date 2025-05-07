import logging
from typing import Any

import torch.nn as nn
from torch.nn.parameter import Parameter
from transformers import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

from .training_arguments import TrainingArguments

log = logging.getLogger(__name__)


def configure_optimizers(model: PreTrainedModel | nn.Module, trainer_config: TrainingArguments):
    log.info("configure_optimizers")
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    param_groups: dict[str, dict[str, list[Parameter]]] = {}

    visual_encoder_params = collect_param_groups(model.visual_encoder, decay_parameters)
    param_groups["visual_encoder"] = visual_encoder_params

    language_model_params = collect_param_groups(model.language_model, decay_parameters)
    param_groups["language_model"] = language_model_params

    connector_params = collect_param_groups(model.connector, decay_parameters)
    param_groups["connector"] = connector_params

    return build_optimizer_params(trainer_config, param_groups)


def collect_param_groups(
    module: PreTrainedModel | nn.Module, decay_parameters: list[str]
) -> dict[str, list[Parameter]]:
    decay_params = [
        p for n, p in module.named_parameters() if p.requires_grad and n in decay_parameters
    ]

    no_decay_params = [
        p for n, p in module.named_parameters() if p.requires_grad and n not in decay_parameters
    ]

    if decay_params or no_decay_params:
        return {"decay": decay_params, "no_decay": no_decay_params}
    return {"decay": [], "no_decay": []}


def build_optimizer_params(
    trainer_config: TrainingArguments,
    param_groups: dict[str, dict[str, list[Parameter]]],
) -> list[dict[str, Any]]:
    optimizer_params: list[dict[str, Any]] = []

    optimizer_params.extend(
        get_module_param_groups(
            module_name="visual_encoder",
            param_groups=param_groups,
            weight_decay=trainer_config.visual_encoder_wd,
            learning_rate=trainer_config.visual_encoder_lr,
        )
    )

    optimizer_params.extend(
        get_module_param_groups(
            module_name="language_model",
            param_groups=param_groups,
            weight_decay=trainer_config.language_model_wd,
            learning_rate=trainer_config.language_model_lr,
        )
    )

    optimizer_params.extend(
        get_module_param_groups(
            module_name="connector",
            param_groups=param_groups,
            weight_decay=trainer_config.connector_wd,
            learning_rate=trainer_config.connector_lr,
        )
    )
    return optimizer_params


def get_module_param_groups(
    module_name: str,
    param_groups: dict[str, dict[str, list[Parameter]]],
    weight_decay: float,
    learning_rate: float,
) -> list[dict[str, Any]]:
    log.info(f"{module_name} lr: {learning_rate}, weight_decay: {weight_decay}")
    return [
        {
            "params": param_groups[module_name]["decay"],
            "weight_decay": weight_decay,
            "lr": learning_rate,
        },
        {
            "params": param_groups[module_name]["no_decay"],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
    ]
