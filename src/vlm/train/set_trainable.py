import logging
from collections import defaultdict
from typing import Any

log = logging.getLogger(__name__)


def format_param_stats(trainable: int, total: int) -> str:
    if total == 0:
        return f"{trainable:,} (0.00% of 0)"
    return f"{trainable:,} ({trainable / total:.2%} of {total:,})"


def count_params(params_list: list):
    trainable = sum(p.numel() for _, p in params_list if p.requires_grad)
    total = sum(p.numel() for _, p in params_list)
    return trainable, total


def group_params_by_prefix(model: Any):
    component_prefixes = {
        "vision_model": ["model.vision_model", "model.vision_tower", "vision_tower"],
        "language_model": [],
        "connector": ["model.connector", "connector"],
        "lm_head": ["lm_head"],
    }

    all_params = list(model.named_parameters())

    grouped_params = defaultdict(list)

    for name, param in all_params:
        assigned = False
        for component, prefixes in component_prefixes.items():
            if any(name.startswith(prefix) for prefix in prefixes):
                grouped_params[component].append((name, param))
                assigned = True
                break

        if not assigned and not any(
            name.startswith(prefix)
            for prefixes in [component_prefixes["vision_model"], component_prefixes["connector"]]
            for prefix in prefixes
        ):
            grouped_params["language_model"].append((name, param))

    return grouped_params


def log_trainable_params_detailed(model: Any):
    grouped_params = group_params_by_prefix(model)

    all_params = list(model.named_parameters())
    total_trainable, total_params = count_params(all_params)

    param_stats = {"total": {"": (total_trainable, total_params)}, "components": {}}

    for component, params_list in grouped_params.items():
        trainable, total = count_params(params_list)
        param_stats["components"][component] = {"": (trainable, total)}

    if total_params > 0:
        log.info(f"Trainable parameters: {format_param_stats(total_trainable, total_params)}")

        for component_name, stats in param_stats["components"].items():
            main_stat = stats[""]
            trainable, total = main_stat
            if total > 0:
                log.info(f" - {component_name}: {format_param_stats(*main_stat)}")

                for subname, substats in stats.items():
                    if subname != "" and substats[1] > 0:
                        log.info(f"   - {subname}: {format_param_stats(*substats)}")

    return param_stats


def set_trainable_params(model: Any, config: dict[str, bool]):
    for param in model.parameters():
        param.requires_grad = False

    grouped_params = group_params_by_prefix(model)

    if config.train_language_model:
        for _, param in grouped_params["language_model"]:
            param.requires_grad = True

        for _, param in grouped_params.get("lm_head", []):
            param.requires_grad = True

    if config.train_vision_model:
        for _, param in grouped_params["vision_model"]:
            param.requires_grad = True

    if config.train_connector:
        for _, param in grouped_params["connector"]:
            param.requires_grad = True

    if getattr(model.config, "use_start_end_tokens", False):
        for _, param in grouped_params.get("embeddings", []):
            param.requires_grad = True

    return log_trainable_params_detailed(model)
