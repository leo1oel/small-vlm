import logging
from collections import defaultdict
from typing import Any

log = logging.getLogger(__name__)


def format_param_count(count: int) -> str:
    """Format parameter count with M/B units"""
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.1f}B"
    elif count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.1f}K"
    else:
        return str(count)


def format_param_stats(trainable: int, total: int) -> str:
    if total == 0:
        return f"{trainable:,} (0.00% of 0)"

    trainable_str = format_param_count(trainable)
    total_str = format_param_count(total)
    percentage = trainable / total * 100

    return f"{trainable_str} ({percentage:.1f}% of {total_str})"


def count_params(params_list: list):
    trainable = sum(p.numel() for _, p in params_list if p.requires_grad)
    total = sum(p.numel() for _, p in params_list)
    return trainable, total


def group_params_by_prefix(model: Any):
    component_prefixes = {
        "vision_model": ["model.vision_model", "model.vision_tower", "vision_tower"],
        "language_model": [],
        # The audio connector deliberately shares the "connector" group: both
        # embedders are trained together (same freeze flag, LR, weight decay) in
        # the gemma4_unified-style stage-1 recipe. Split into its own group here
        # (plus optimizer/schema dials) if they ever need separate treatment.
        "connector": ["model.connector", "model.audio_connector", "connector"],
        "lm_head": ["lm_head"],
        # Visual-aux head (spec 2026-06-06): its own group — without this the
        # params would fall through to "language_model" (the unassigned-prefix
        # default below) and silently take the LM lr / freeze flag.
        "visual_aux_head": ["visual_aux_head"],
        # Generation pathway (spec 2026-06-20): x-prediction head + in-context
        # timestep embedder (named gen_x_head.* / gen_t_embed.*, directly on the
        # ForCausalLM). Own group so they never fall through to "language_model"
        # and silently inherit its freeze flag / LR — they are always trained
        # when present (see set_trainable_params).
        "generation": ["gen_x_head", "gen_t_embed", "gen_patch_embed"],
        # BREEN port (spec 2026-06-24): the learnable queries and the CLIP->LLM
        # distill head (incl. its norm_layer) are directly on the ForCausalLM.
        # Own groups so a frozen-LLM S0 trains them instead of freezing them
        # (without this they fall through to "language_model" — the silent
        # no-op the build-item-8 fix exists to prevent). Always trained when
        # present (see set_trainable_params). The per-layer visual FFN expert
        # (mlp_visual) and its sigmoid gates are substring-named inside the
        # decoder layers, so they can't be prefix-grouped — they are force-
        # enabled directly in set_trainable_params instead.
        "learnable_query": ["learnable_query"],
        "visual_distill_head": ["visual_distill_head"],
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
        log.info("Training parameters overview:")
        log.info(f"  Total parameters: {format_param_count(total_params)}")
        log.info(f"  Trainable: {format_param_stats(total_trainable, total_params)}")

        log.info("Component breakdown:")
        for component_name, stats in param_stats["components"].items():
            main_stat = stats[""]
            trainable, total = main_stat
            if total > 0:
                status = "trainable" if trainable > 0 else "frozen"
                log.info(f"  • {component_name}: {format_param_stats(*main_stat)} [{status}]")

                for subname, substats in stats.items():
                    if subname != "" and substats[1] > 0:
                        sub_status = "trainable" if substats[0] > 0 else "frozen"
                        log.info(f"    - {subname}: {format_param_stats(*substats)} [{sub_status}]")

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

    # The visual-aux head exists only when the objective is active and is
    # always trainable — it is the fresh module the aux loss exists to train,
    # in every recipe (incl. frozen-trunk retrofits later).
    for _, param in grouped_params.get("visual_aux_head", []):
        param.requires_grad = True

    # Generation modules (x-pred head + timestep embedder) are always trainable
    # when present — the fresh modules the flow-matching loss exists to train,
    # in every recipe (incl. a frozen-trunk generation adapter later). Without
    # this they would inherit train_language_model; a frozen LM would then leave
    # the zero-initialized x-head identically zero -> pred_x0==0 -> loss never
    # moves (silent no-op).
    for _, param in grouped_params.get("generation", []):
        param.requires_grad = True

    # BREEN port (spec 2026-06-24): the learnable queries, the CLIP->LLM distill
    # head (+ norm_layer), the per-layer visual FFN expert (mlp_visual) and its
    # sigmoid gates are the fresh visual capacity every BREEN recipe trains —
    # including a frozen-LLM S0 (train_language_model=false). Force them on so
    # they never inherit the frozen LM flag (the build-item-8 silent-no-op trap).
    # No-op when the modules are absent (baseline / non-BREEN runs).
    for _, param in grouped_params.get("learnable_query", []):
        param.requires_grad = True
    for _, param in grouped_params.get("visual_distill_head", []):
        param.requires_grad = True
    for name, param in model.named_parameters():
        if ".mlp_visual." in name or "expert_gate" in name:
            param.requires_grad = True

    if getattr(model.config, "use_start_end_tokens", False):
        for _, param in grouped_params.get("embeddings", []):
            param.requires_grad = True

    return log_trainable_params_detailed(model)


def apply_delta_tuning(model: Any):
    """End-to-end (single-run, no staging) delta tuning to defeat gradient
    starvation (see devtools/grad_probe.py + memory visual-pathway-diagnosis).

    Overrides requires_grad AFTER set_trainable_params: keeps ONLY the visual
    pathway (connector + per-layer visual FFN expert `mlp_visual`) and the shared
    self-attention trainable; FREEZES the pure-language params (text FFN
    gate/up/down, token embeddings, lm_head, all norms). This is the end-to-end
    equivalent of Mono-InternVL EViP's frozen-LLM concept stage: by freezing the
    text FFN the language shortcut can't update, so gradient flows to the visual
    pathway instead of starving it — without multi-stage training. Attention is
    left trainable for vision-language alignment (Mono S1.3). visual_aux_head, if
    present, stays trainable."""
    import os as _os

    # DELTA_TUNING=2: also freeze attention (Mono-InternVL S1.1/S1.2 — entire LLM
    # frozen, only visual pathway trains). v1 (=1) left attention trainable, which
    # is itself a language shortcut and failed to stop starvation (measured).
    freeze_attn = _os.environ.get("DELTA_TUNING") == "2"
    trainable = frozen = 0
    for name, p in model.named_parameters():
        keep = (
            "connector" in name
            or ".mlp_visual." in name
            or "visual_aux_head" in name
            or "gen_x_head" in name
            or "gen_t_embed" in name
            or "gen_patch_embed" in name
            # BREEN port: queries / distill head / expert gates are visual pathway.
            or "learnable_query" in name
            or "visual_distill_head" in name
            or "expert_gate" in name
            or (not freeze_attn and ".self_attn." in name)
        )
        p.requires_grad = keep
        if keep:
            trainable += p.numel()
        else:
            frozen += p.numel()
    log.info(
        f"[delta-tuning] trainable={format_param_count(trainable)} "
        f"frozen={format_param_count(frozen)} "
        f"(visual pathway + attention trainable; text FFN/embed/lm_head/norm frozen)"
    )
    return log_trainable_params_detailed(model)
