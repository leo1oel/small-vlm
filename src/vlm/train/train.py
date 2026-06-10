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


def validate_aux_exit_config(
    aux_exit_layers: Any, num_hidden_layers: int, loss_chunk_size: int
) -> list[int]:
    """Validate + normalize the aux-exit deep-supervision layer list (spec:
    docs/superpowers/specs/2026-06-05-aux-exit-loss-design.md). k is the
    1-based index of the decoder layer whose output feeds the shared
    norm+lm_head exit; k == num_hidden_layers would duplicate the main loss.
    Returns plain sorted unique ints (no OmegaConf types — the result lands
    on model.config and must serialize into checkpoint config.json)."""
    layers = sorted({int(k) for k in (aux_exit_layers or [])})
    if not layers:
        return []
    bad = [k for k in layers if not 1 <= k <= num_hidden_layers - 1]
    if bad:
        raise ValueError(
            f"trainer.aux_exit_layers {bad} out of range [1, {num_hidden_layers - 1}] "
            f"for a {num_hidden_layers}-layer backbone"
        )
    if loss_chunk_size <= 0:
        raise ValueError(
            "trainer.aux_exit_layers requires trainer.loss_chunk_size > 0 — the aux "
            "loss is implemented only in the chunked-CE training path"
        )
    return layers


def validate_visual_aux_config(
    objective: Any,
    layer: Any,
    num_hidden_layers: int,
    loss_chunk_size: int,
    encoder_free: bool,
) -> tuple[str, int | None]:
    """Validate the visual-aux dials (spec:
    docs/superpowers/specs/2026-06-06-visual-aux-loss-design.md). Returns the
    normalized (objective, layer) as plain python types (config.json-safe)."""
    objective = str(objective or "none")
    if objective == "none":
        return "none", None
    if objective not in ("aim_pixel", "nepa"):
        raise ValueError(f"model.visual_aux.objective {objective!r} not in (none, aim_pixel, nepa)")
    if loss_chunk_size <= 0:
        raise ValueError(
            "model.visual_aux.objective requires trainer.loss_chunk_size > 0 — the "
            "visual aux loss is implemented only in the chunked-CE training path"
        )
    if not encoder_free:
        raise ValueError(
            "visual_aux supports only the encoder-free (raw_patch) path: targets "
            "are raw patch rows / connector embedding rows, which the classic "
            "vision-tower path does not produce"
        )
    if layer is None:
        return objective, None
    k = int(layer)
    if not 1 <= k <= num_hidden_layers - 1:
        raise ValueError(
            f"trainer.visual_aux_layer {k} out of range [1, {num_hidden_layers - 1}] "
            f"for a {num_hidden_layers}-layer backbone"
        )
    return objective, k


def validate_cross_modal_mask_config(
    mode: Any,
    window: Any,
    bidirectional: Any,
    attn_implementation: Any,
    num_hidden_layers: int,
) -> tuple[str, list[int]]:
    """Validate the cross-modal 4D mask dials for the early-fusion access arms
    (plan docs/superpowers/plans/2026-06-10-early-fusion-access-arms.md).
    Returns the normalized (mode, window) as plain python types (config.json-
    safe; window is meaningful only for img2q_window but always returned so the
    checkpoint self-describes). "none" is the bit-identical baseline path.

    Adapts the sibling validators' call style: it reads the individual dials
    (model.cross_modal_mask.* + trainer.attn_implementation) rather than a full
    cfg, so it can run in train() against the real backbone layer count, with
    persistence flowing through model.config like visual_aux_objective."""
    mode = str(mode or "none")
    if mode == "none":
        return "none", [int(x) for x in (window or [1, 9])]
    if mode not in ("prefix_lm", "img2q_window"):
        raise ValueError(
            f"cross_modal_mask.mode must be none|prefix_lm|img2q_window, got {mode!r}"
        )
    if bool(bidirectional):
        raise ValueError(
            "cross_modal_mask.bidirectional=true is not implemented in v1 "
            "(mutual windowing needs per-layer decode masking)"
        )
    attn = str(attn_implementation or "")
    if mode == "prefix_lm" and attn not in ("sdpa", "sdpa_xmodal"):
        raise ValueError("prefix_lm needs trainer.attn_implementation=sdpa (4D masks bypass FA2)")
    win = [int(x) for x in (window or [1, 9])]
    if mode == "img2q_window":
        if attn != "sdpa_xmodal":
            raise ValueError("img2q_window needs trainer.attn_implementation=sdpa_xmodal")
        lo, hi = int(win[0]), int(win[1])
        if not (1 <= lo <= hi <= num_hidden_layers):
            raise ValueError(
                f"cross_modal_mask.window {win} out of range 1..{num_hidden_layers}"
            )
    return mode, win


def validate_energon_args(training_args: TrainingArguments) -> None:
    """The streaming loader has no epoch length and does its own sampling, so a
    few HF Trainer features are structurally unavailable — fail loud upfront.

    Called from vlm() before the (slow) model load so misconfigurations abort
    in seconds, and again from train() to guard direct callers."""
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
    # Record the template in the checkpoint config (serialized into config.json
    # by save_pretrained) so inference can rebuild the exact training-time
    # prompt format without guessing — vlm.inference.resolve_conv_mode reads it.
    model.config.conversation_version = training_args.version
    # Training-only chunked CE switch; the model's forward reads it (0 = off).
    model.config.loss_chunk_size = training_args.loss_chunk_size
    # Aux-exit deep supervision (early-fusion ablation): validated against the
    # real layer count, then copied next to loss_chunk_size for the model's
    # chunked-CE forward to read. Plain python types only (config.json-safe).
    model.config.aux_exit_layers = validate_aux_exit_config(
        training_args.aux_exit_layers,
        num_hidden_layers=len(model.model.layers),
        loss_chunk_size=int(training_args.loss_chunk_size),
    )
    model.config.aux_exit_weight = float(training_args.aux_exit_weight)
    model.config.aux_exit_detach = bool(training_args.aux_exit_detach)
    if model.config.aux_exit_layers and model.config.aux_exit_weight <= 0.0:
        # Loud, because this arm would silently train as a pure baseline
        # duplicate — almost certainly a typo in the run config.
        log.warning(
            "aux_exit_layers=%s but aux_exit_weight=%s: the aux loss is DISABLED "
            "(weight must be > 0) — this run is an exact baseline duplicate",
            model.config.aux_exit_layers,
            model.config.aux_exit_weight,
        )

    # Visual-aux (spec 2026-06-06): objective/head dials were placed on
    # model.config at build time (load_model); validate them against the real
    # backbone and copy the trainer-side dials next to them.
    va_objective, va_layer = validate_visual_aux_config(
        getattr(model.config, "visual_aux_objective", "none"),
        training_args.visual_aux_layer,
        num_hidden_layers=len(model.model.layers),
        loss_chunk_size=int(training_args.loss_chunk_size),
        encoder_free=model.model.vision_model is None,
    )
    model.config.visual_aux_weight = float(training_args.visual_aux_weight)
    model.config.visual_aux_layer = va_layer
    if va_objective != "none" and getattr(model, "visual_aux_head", None) is None:
        raise ValueError(
            "visual_aux objective is set but the model has no visual_aux_head — "
            "was the model loaded from an understanding-only checkpoint?"
        )
    if va_objective != "none" and model.config.visual_aux_weight <= 0.0:
        # Loud, mirroring the aux-exit guard: this arm would silently train
        # as a pure baseline duplicate — almost certainly a run-config typo.
        log.warning(
            "visual_aux_objective=%s but visual_aux_weight=%s: the visual aux loss "
            "is DISABLED (weight must be > 0) — this run is an exact baseline duplicate",
            va_objective,
            model.config.visual_aux_weight,
        )

    # Cross-modal 4D mask (early-fusion access arms, plan 2026-06-10): the mode/
    # window/bidirectional dials were placed on model.config at build time
    # (load_model, like visual_aux_objective). Validate against the real layer
    # count + the requested attn impl, then copy the normalized result back as
    # flat fields so the checkpoint config.json self-describes for inference.
    cmm_mode, cmm_window = validate_cross_modal_mask_config(
        getattr(model.config, "cross_modal_mask_mode", "none"),
        getattr(model.config, "cross_modal_mask_window", [1, 9]),
        getattr(model.config, "cross_modal_mask_bidirectional", False),
        attn_implementation=training_args.attn_implementation,
        num_hidden_layers=len(model.model.layers),
    )
    model.config.cross_modal_mask_mode = cmm_mode
    model.config.cross_modal_mask_window = cmm_window

    model.config.use_cache = False
    set_trainable_params(model, training_args)

    if energon_loader is not None:
        validate_energon_args(training_args)

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
        # One-shot escape hatch for checkpoints whose LOADER state is
        # unrestorable (e.g. buffer restore-keys written by older code):
        # `touch <output_dir>/.skip_energon_restore_once` before submitting.
        # Weights/optimizer/scheduler still resume; only the stream restarts.
        # The marker is consumed, so later preemption-requeues restore fully.
        skip_marker = os.path.join(training_args.output_dir, ".skip_energon_restore_once")
        skip_stream_restore = os.path.isfile(skip_marker)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # Every rank must sample the marker before rank 0 consumes it.
            torch.distributed.barrier()
        if skip_stream_restore:
            if training_args.process_index == 0:
                try:
                    os.remove(skip_marker)
                except FileNotFoundError:
                    pass
            log.warning(
                "one-shot marker found: SKIPPING energon stream restore — the data "
                f"stream restarts fresh; model/optimizer still resume from {resume_ckpt}"
            )
        else:
            restore_energon_state(energon_loader, resume_ckpt, training_args, energon_num_workers)

    trainer.train(resume_from_checkpoint=resume_ckpt)

    log.info("Saving state")
    trainer.save_state()
    model.config.use_cache = True

    log.info("Saving model")
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
