import logging
from typing import Any

import torch.nn as nn
from torch.nn.parameter import Parameter
from transformers import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

from .set_trainable import group_params_by_prefix
from .training_arguments import TrainingArguments

log = logging.getLogger(__name__)


def configure_optimizers(model: PreTrainedModel | nn.Module, trainer_config: TrainingArguments):
    log.info("configure_optimizers")
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    grouped_params = group_params_by_prefix(model)

    param_groups: dict[str, dict[str, list[Parameter]]] = {}

    component_to_config = {
        "language_model": "model",
        "vision_model": "vision_model",
        "connector": "connector",
        "lm_head": "model",
        "visual_aux_head": "visual_aux_head",
        # Generation x-pred head + timestep embedder train with the LM lr/wd
        # ("model" config) — a single LR is fine for v1 (a dedicated higher gen
        # lr can be added as generation_lr later). Mapping it here ensures the
        # group is NOT silently dropped (configure_optimizers skips unmapped
        # groups).
        "generation": "model",
        # BREEN port (spec 2026-06-24): the learnable queries and the CLIP->LLM
        # distill head (+ norm_layer) are fresh visual-interface adapters — train
        # them with the connector/projector lr/wd (BREEN's higher "proj" LR in
        # S0; identical to the LM lr in the single-LR SFT configs, so no
        # regression for the existing eve/repa distill arms). MUST be mapped here
        # or configure_optimizers silently drops them (trainable but never
        # stepped).
        "learnable_query": "connector",
        "visual_distill_head": "connector",
        # The per-layer visual experts (mlp_visual / norm_visual / proj_visual +
        # gates) are in-stack capacity trained at the LM lr/wd ("model" config).
        # They used to fall through into the "language_model" group (which also
        # maps to "model"); set_trainable.group_params_by_prefix now gives them
        # their own truthful "visual_expert" group, so map it to the SAME "model"
        # config to keep the optimizer byte-identical. MUST be mapped here or
        # configure_optimizers silently drops them (trainable but never stepped).
        "visual_expert": "model",
    }

    for component, params_list in grouped_params.items():
        if not params_list:
            continue

        config_name = component_to_config.get(component)
        if not config_name:
            continue

        if config_name not in param_groups:
            param_groups[config_name] = {"decay": [], "no_decay": []}

        for name, param in params_list:
            if not param.requires_grad:
                continue

            if name in decay_parameters:
                param_groups[config_name]["decay"].append(param)
            else:
                param_groups[config_name]["no_decay"].append(param)

    filtered_param_groups = {}
    for group_name, group in param_groups.items():
        has_params = len(group["decay"]) > 0 or len(group["no_decay"]) > 0
        if has_params:
            filtered_param_groups[group_name] = group
            decay_count = sum(p.numel() for p in group["decay"])
            no_decay_count = sum(p.numel() for p in group["no_decay"])
            if decay_count or no_decay_count:
                log.info(
                    f"{group_name}: decay params: {decay_count:,}, no_decay params: {no_decay_count:,} (trainable)"
                )
        else:
            log.info(f"{group_name}: No trainable parameters found for this group, skipping")

    return build_optimizer_params(trainer_config, filtered_param_groups)


def build_optimizer_params(
    trainer_config: TrainingArguments,
    param_groups: dict[str, dict[str, list[Parameter]]],
) -> list[dict[str, Any]]:
    optimizer_params: list[dict[str, Any]] = []

    component_configs = {
        "vision_model": {
            "weight_decay": trainer_config.visual_encoder_wd,
            "learning_rate": trainer_config.visual_encoder_lr,
        },
        "model": {
            "weight_decay": trainer_config.language_model_wd,
            "learning_rate": trainer_config.language_model_lr,
        },
        "connector": {
            "weight_decay": trainer_config.connector_wd,
            "learning_rate": trainer_config.connector_lr,
        },
        "visual_aux_head": {
            # None falls back to the run's default lr / the LM weight decay —
            # the head is fresh, so a higher dedicated lr is a legitimate dial.
            "weight_decay": trainer_config.visual_aux_head_wd
            if trainer_config.visual_aux_head_wd is not None
            else trainer_config.language_model_wd,
            "learning_rate": trainer_config.visual_aux_head_lr
            if trainer_config.visual_aux_head_lr is not None
            else trainer_config.learning_rate,
        },
    }

    for module_name, config in component_configs.items():
        if module_name in param_groups:
            optimizer_params.extend(
                get_module_param_groups(
                    module_name=module_name,
                    param_groups=param_groups,
                    weight_decay=config["weight_decay"],
                    learning_rate=config["learning_rate"],
                )
            )

    if not optimizer_params:
        log.warning(
            "No parameter groups found for optimization. Check if any parameters are set to trainable."
        )

    return optimizer_params


def get_module_param_groups(
    module_name: str,
    param_groups: dict[str, dict[str, list[Parameter]]],
    weight_decay: float,
    learning_rate: float,
) -> list[dict[str, Any]]:
    log.info(f"{module_name} lr: {learning_rate}, weight_decay: {weight_decay}")

    groups = []

    if param_groups[module_name]["decay"]:
        groups.append(
            {
                "params": param_groups[module_name]["decay"],
                "weight_decay": weight_decay,
                "lr": learning_rate,
            }
        )

    if param_groups[module_name]["no_decay"]:
        groups.append(
            {
                "params": param_groups[module_name]["no_decay"],
                "weight_decay": 0.0,
                "lr": learning_rate,
            }
        )

    return groups
