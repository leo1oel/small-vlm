"""Inference for VLM checkpoints trained with this framework.

Two model families share this entry point:

  - **Encoder-free unified models** (``vision_model is None``): images go
    through :class:`RawImageProcessor` (aspect-preserving, variable-length raw
    patches + ``image_position_ids``), audio through ``load_audio_frames``
    (raw 16 kHz waveform frames). Trained with the ``plain`` (stage-1) or
    ``qwen_2_5`` (SFT) conversation templates via the energon streaming path,
    which does NOT run ``preprocess_multimodal`` — placeholders stay where the
    text puts them.
  - **Legacy encoder models** (CLIP/SigLIP/DINO tower): square-pad pixel
    batches via ``process_images``; the local-json training path runs
    ``preprocess_multimodal`` (single ``<image>`` moved to the front of the
    user turn, optional ``<im_start>``/``<im_end>`` wrapping), which is
    mirrored here for the legacy templates.

Input construction mirrors the training pipeline in ``vlm.data.dataset``
exactly (same placeholder handling, same tokenization, same image/audio
preprocessing); tests/test_inference.py pins the parity.

Typical use::

    from vlm.inference import eval_model
    eval_model("outputs/.../checkpoint", query="What is in the image?",
               image_path="cat.jpg")

or, keeping the model loaded across calls::

    from vlm.inference import generate_response, load_model
    model, processor, _ = load_model("outputs/.../checkpoint")
    text = generate_response(model, processor, "What is in the image?",
                             images="cat.jpg")
"""

import logging
import re
import warnings
import zlib
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from ..data.data_arguments import DataArguments
from ..data.dataset import (
    apply_image_position,
    inject_query_placeholders,
    load_audio_frames,
    tokenizer_multimodal_token,
)
from ..models import VLMProcessor, get_dynamic_vlm
from ..models.image_processing_raw import RawImageProcessor
from ..utils import conversation as conversation_lib
from ..utils.conversation import conv_templates
from .generator import process_images

log: logging.Logger = logging.getLogger(name=__name__)

# preprocess_qwen hardcodes this system message for every training sample,
# regardless of the conv template's own `system` string (which is therefore
# dead at train time) — inference must use the same one. Pinned by
# tests/test_inference.py::test_qwen_prompt_matches_training.
QWEN_SYSTEM_MESSAGE = "You are a helpful assistant."


def _media_regex(data_args: DataArguments) -> str:
    """Same pattern as vlm.data.dataset._media_pattern (kept in sync; the
    prompt-parity tests would catch a divergence)."""
    tokens = [data_args.image_token, data_args.audio_token]
    # BREEN port: "<query>" is a splice placeholder (the model expands it into
    # the learnable query block), so the plain template must keep/extract it
    # alongside the media tokens — mirrors _media_pattern's learnable_query
    # gating. Without this the plain path drops the injected "<query>" tokens.
    if getattr(data_args, "learnable_query_enabled", False):
        tokens.append(data_args.query_token)
    return "(" + "|".join(re.escape(t) for t in tokens) + ")"


def _auto_detect_conv_mode(model_path: str) -> str:
    """Auto-detect conversation mode based on model path (legacy heuristic).

    Only used as a last resort for checkpoints that predate the
    ``conversation_version`` config field written at training time.
    """
    model_path_lower = model_path.lower()

    # Check for specific model types in order of specificity
    if "llama-3" in model_path_lower or "llama3" in model_path_lower:
        return "llava_llama_3"
    elif "llama-2" in model_path_lower or "llama2" in model_path_lower:
        return "llava_llama_2"
    elif "llama" in model_path_lower:
        return "llava_llama_2"  # Default for llama models
    elif "qwen3" in model_path_lower or "qwen-3" in model_path_lower:
        return "qwen_2_5"  # Qwen3 uses the same ChatML dialect
    elif "qwen2.5" in model_path_lower or "qwen-2.5" in model_path_lower:
        return "qwen_2_5"
    elif "qwen2" in model_path_lower or "qwen-2" in model_path_lower:
        return "qwen_2"
    elif "qwen1.5" in model_path_lower or "qwen-1.5" in model_path_lower:
        return "qwen_1_5"
    elif "qwen" in model_path_lower:
        return "qwen_1_5"  # Default for qwen models
    elif "mistral" in model_path_lower and "instruct" in model_path_lower:
        return "mistral_instruct"
    elif "mistral" in model_path_lower and "orca" in model_path_lower:
        return "mistral_orca"
    elif "mistral" in model_path_lower and "zephyr" in model_path_lower:
        return "mistral_zephyr"
    elif "mistral" in model_path_lower:
        return "mistral_instruct"  # Default for mistral models
    elif "gemma" in model_path_lower and "instruct" in model_path_lower:
        return "gemma_instruct"
    elif "vicuna" in model_path_lower:
        return "vicuna_v1"
    elif "mpt" in model_path_lower:
        return "mpt"
    elif "llava" in model_path_lower and "v0" in model_path_lower:
        return "llava_v0"
    elif "llava" in model_path_lower:
        return "llava_v1"  # Default for llava models
    else:
        # Default fallback
        warnings.warn(
            f"Could not auto-detect conv_mode for {model_path}, using 'v1' as default",
            stacklevel=2,
        )
        return "v1"


def resolve_conv_mode(
    model_config: Any,
    pretrained: str | None = None,
    conv_mode: str | None = None,
) -> str:
    """Pick the conversation template, in priority order:

    1. explicit ``conv_mode`` argument;
    2. ``conversation_version`` recorded in the checkpoint config (written by
       train.py since 2026-06 — exactly the trainer.version used in training);
    3. path-name heuristic (legacy checkpoints only; warns).
    """
    if conv_mode is not None:
        if conv_mode not in conv_templates:
            raise ValueError(f"unknown conv_mode {conv_mode!r}; valid: {sorted(conv_templates)}")
        return conv_mode
    recorded = getattr(model_config, "conversation_version", None)
    if recorded is not None:
        if recorded not in conv_templates:
            raise ValueError(
                f"checkpoint records conversation_version={recorded!r} which is not "
                f"in conv_templates; pass conv_mode explicitly"
            )
        return recorded
    if pretrained is None:
        raise ValueError(
            "cannot resolve the conversation template: the checkpoint config has no "
            "conversation_version and no conv_mode/pretrained path was given. Pass "
            "conv_mode= explicitly (training used trainer.version, e.g. 'plain' for "
            "stage-1 pretrain, 'qwen_2_5' for SFT)."
        )
    mode = _auto_detect_conv_mode(pretrained)
    warnings.warn(
        f"checkpoint config has no conversation_version; guessed conv_mode={mode!r} "
        f"from the path name. Pass conv_mode= explicitly if this is wrong "
        f"(training used trainer.version: 'plain' for stage-1, 'qwen_2_5' for SFT).",
        stacklevel=2,
    )
    return mode


def _data_args_from_config(config: Any) -> DataArguments:
    """Rebuild the (token/placeholder-related) DataArguments the checkpoint was
    trained with from its saved config. The flattened VLM config carries the
    LanguageModelConfig fields; audio dials live in the nested audio_config."""
    audio_cfg = getattr(config, "audio_config", None)
    audio_enabled = bool(audio_cfg is not None and getattr(audio_cfg, "enabled", False))
    return DataArguments(
        image_token=getattr(config, "image_token", "<image>"),
        image_token_index=getattr(config, "image_token_index", -200),
        use_start_end_tokens=getattr(config, "use_start_end_tokens", False),
        image_start_token=getattr(config, "image_start_token", "<im_start>"),
        image_end_token=getattr(config, "image_end_token", "<im_end>"),
        ignore_index=getattr(config, "ignore_index", -100),
        audio_token=getattr(config, "audio_token", "<audio>"),
        audio_token_index=getattr(config, "audio_token_index", -201),
        audio_enabled=audio_enabled,
        audio_sampling_rate=int(getattr(audio_cfg, "sampling_rate", 16000) or 16000)
        if audio_cfg is not None
        else 16000,
        audio_samples_per_token=int(getattr(audio_cfg, "samples_per_token", 640) or 640)
        if audio_cfg is not None
        else 640,
        max_audio_tokens=getattr(audio_cfg, "max_audio_tokens", 750)
        if audio_cfg is not None
        else 750,
        # BREEN port: a query-trained checkpoint must inject "<query>" at
        # inference too (the model expects the query block to summarize the
        # image). Self-describing from the saved config.
        learnable_query_enabled=bool(getattr(config, "learnable_query", False)),
        query_token=getattr(config, "query_token", "<query>"),
        query_token_index=getattr(config, "query_token_index", -202),
        query_placement=str(getattr(config, "learnable_query_placement", "after_image")),
    )


def load_model(
    pretrained: str,
    bf16: bool = True,
    fp16: bool = False,
    attn_implementation: str = "sdpa",
    device: str | torch.device | None = None,
):
    """
    Load VLM model and processor for inference.

    Args:
        pretrained: Path to a trained checkpoint dir (or Hub repo id)
        bf16: Use bfloat16 precision
        fp16: Use float16 precision
        attn_implementation: Attention implementation type (default sdpa,
            matching training; pass "eager" for maximum compatibility)
        device: Target device; defaults to cuda when available, else cpu

    Returns:
        Tuple of (model, processor, config_dict)
    """
    # Fail fast with a clear message for vanished/typo'd local paths (path-like
    # inputs that don't exist would otherwise surface as confusing hub-repo-id
    # validation errors from transformers' loaders).
    if ("/" in str(pretrained) or Path(str(pretrained)).is_absolute()) and not Path(
        str(pretrained)
    ).exists():
        if str(pretrained).count("/") != 1 or str(pretrained).startswith((".", "/", "~")):
            raise FileNotFoundError(
                f"checkpoint path does not exist: {pretrained} "
                "(deleted by checkpoint rotation / moved? pass a hub repo id as "
                "'namespace/name' if you meant one)"
            )
    processor = VLMProcessor.from_pretrained(pretrained)
    VLMForCausalLM, VLMConfig = get_dynamic_vlm(pretrained)
    # Self-describing mask arms (plan 2026-06-10): the img2q_window arm's
    # per-layer masks are consumed only by the registered "sdpa_xmodal"
    # attention function — plain sdpa would silently drop them and evaluate
    # the model as text-blind causal. _attn_implementation never serializes
    # into config.json, so auto-upgrade from the persisted mask mode instead.
    ckpt_config = VLMConfig.from_pretrained(pretrained)
    if (
        str(getattr(ckpt_config, "cross_modal_mask_mode", "none") or "none") == "img2q_window"
        and attn_implementation == "sdpa"
    ):
        attn_implementation = "sdpa_xmodal"
    model: VLMForCausalLM = VLMForCausalLM.from_pretrained(
        pretrained,
        dtype=torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32,
        attn_implementation=attn_implementation,
    )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Qwen3-backbone native checkpoints serialize bos_token_id=None (the base
    # Qwen3 LM carries 151643, but the trained VLM config drops it). The
    # encoder-free generate() hands HF inputs_embeds with no input_ids; HF still
    # needs a bos to seed its output-id bookkeeping whenever it cannot start the
    # sequence from inputs_embeds, and raises "bos_token_id has to be defined"
    # when it is None. On the current splice path inputs_embeds is always real so
    # the splice fix (single-token prefill no longer dropped) keeps this latent,
    # but the seed is one HF-version/code change away from firing. Pin a valid id
    # (pad, else eos) so a None-bos checkpoint can always generate_response. This
    # is bookkeeping only — the id is never prepended on the inputs_embeds path
    # and is stripped on decode (skip_special_tokens) — and defined-bos backbones
    # (e.g. Llama) are left untouched by the `is None` guard.
    gen_cfg = model.generation_config
    if gen_cfg.bos_token_id is None:
        seed_bos = gen_cfg.pad_token_id
        if seed_bos is None:
            seed_bos = gen_cfg.eos_token_id
        if isinstance(seed_bos, list | tuple):
            seed_bos = seed_bos[0] if seed_bos else None
        if seed_bos is not None:
            gen_cfg.bos_token_id = int(seed_bos)

    config_dict = {
        "image_token_index": getattr(model.config, "image_token_index", -200),
        "image_start_token": getattr(model.config, "image_start_token", "<im_start>"),
        "image_end_token": getattr(model.config, "image_end_token", "<im_end>"),
        "image_token": getattr(model.config, "image_token", "<image>"),
        "use_start_end_tokens": getattr(model.config, "use_start_end_tokens", False),
        "audio_token": getattr(model.config, "audio_token", "<audio>"),
        "audio_token_index": getattr(model.config, "audio_token_index", -201),
        "conversation_version": getattr(model.config, "conversation_version", None),
        "encoder_free": getattr(model.model, "vision_model", None) is None,
        "has_audio": getattr(model.model, "audio_connector", None) is not None,
    }

    return model, processor, config_dict


def ensure_placeholders(query: str, n_images: int, n_audios: int, data_args: DataArguments) -> str:
    """Reconcile media placeholders in the query with the media actually
    passed — single-string mirror of the training-side
    ``inject_missing_media_tokens`` (same rules, same error)."""
    prefix = ""
    for token, n in ((data_args.image_token, n_images), (data_args.audio_token, n_audios)):
        found = query.count(token)
        if found == n:
            continue
        if n > 0 and found == 0:
            prefix += (token + "\n") * n
        else:
            raise ValueError(
                f"query has {found} {token!r} placeholder(s) but {n} media item(s) "
                "were passed; provide one placeholder per item (or none, to have "
                "them prepended automatically)"
            )
    return prefix + query


def build_prompt(conv_mode: str, query: str, data_args: DataArguments) -> str:
    """Render the generation prompt exactly as training rendered its inputs.

    - ``plain`` (stage-1 pretrain): training drops the human text and keeps
      only the media placeholders (preprocess_plain); the model continues with
      the caption/transcript directly.
    - qwen family (SFT): ChatML with the *training* system message — mirrors
      preprocess_qwen's forced chat template, ending with the generation
      prefix ``<|im_start|>assistant\\n``.
    - everything else: legacy conv-template machinery, with the
      preprocess_multimodal normalizations the local-json training path
      applies (single image token moved to the front; optional
      ``<im_start>``/``<im_end>`` wrapping).
    """
    conv = conv_templates[conv_mode]

    if conv.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        placeholders = re.findall(_media_regex(data_args), query)
        if not placeholders:
            raise ValueError(
                "the 'plain' template needs at least one image or audio input "
                "(stage-1 models are caption/transcript models)"
            )
        stripped = re.sub(_media_regex(data_args), "", query).strip()
        if stripped:
            log.warning(
                "the 'plain' template ignores query text (training drops the human "
                "turn); discarding %r",
                stripped,
            )
        return "".join(placeholders)

    if conv.version == "qwen":
        # Mirror preprocess_qwen: forced ChatML template, hardcoded system
        # message, then the add_generation_prompt suffix.
        return (
            f"<|im_start|>system\n{QWEN_SYSTEM_MESSAGE}<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    # Legacy templates (v1 / llama / mpt / gemma ...): the local-json training
    # path runs preprocess_multimodal first — replicate it on the user turn,
    # including its n_image == 1 guard (dataset.py:385). Single-image LLaVA
    # convention hoists the lone "<image>" to the front; for an interleaved
    # multi-image turn (n_image > 1) every placeholder MUST stay in place.
    # Stripping all and prepending one would leave a single sentinel for N
    # images, and the splice (one feature per sentinel) would consume image 1
    # and silently drop images 2..N (the encoder still passes all N).
    n_image = query.count(data_args.image_token)
    if n_image == 1:
        query = query.replace(data_args.image_token, "").strip()
        query = data_args.image_token + "\n" + query
        query = query.strip()
    if n_image >= 1 and "mmtag" in conv.version:
        query = query.replace(data_args.image_token, "<Image>" + data_args.image_token + "</Image>")
    if data_args.use_start_end_tokens:
        query = query.replace(
            data_args.image_token,
            data_args.image_start_token + data_args.image_token + data_args.image_end_token,
        )

    conv = conv.copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return list(value)
    return [value]


def prepare_media_inputs(
    model: Any,
    processor: VLMProcessor,
    images: list,
    audios: list,
    data_args: DataArguments,
    image_aspect_ratio: str | None = None,
) -> dict[str, Any]:
    """Turn PIL images / audio sources into the kwargs ``model.generate``
    expects, matching the training-side preprocessing for the model family.

    ``image_aspect_ratio`` (classic encoder path only): defaults to the value
    recorded in the checkpoint config at training time (vlm.py), falling back
    to "pad" for older checkpoints — pass it explicitly if the checkpoint
    predates the recording and trained with dataset.image_aspect_ratio=square.

    Returns a dict with (a subset of) ``images``, ``image_position_ids``,
    ``image_sizes``, ``audios``.
    """
    gen_kwargs: dict[str, Any] = {}
    device = model.device
    encoder_free = getattr(model.model, "vision_model", None) is None

    if images:
        pil_images = []
        for image in images:
            if isinstance(image, str | Path):
                image = Image.open(image)
            pil_images.append(image.convert("RGB"))

        if encoder_free:
            if not isinstance(processor.image_processor, RawImageProcessor):
                raise ValueError(
                    "encoder-free model but the processor is not a RawImageProcessor — "
                    "checkpoint/processor mismatch (was the processor saved with the model?)"
                )
            # Training parity: process_raw_image == processor.preprocess, no
            # square-padding, aspect preserved, variable-length patch lists.
            out = processor.image_processor.preprocess(pil_images)
            gen_kwargs["images"] = [t.to(device) for t in out["pixel_values"]]
            gen_kwargs["image_position_ids"] = [t.to(device) for t in out["image_position_ids"]]
            gen_kwargs["image_sizes"] = [img.size for img in pil_images]
        else:
            from types import SimpleNamespace

            if image_aspect_ratio is None:
                image_aspect_ratio = getattr(model.config, "image_aspect_ratio", None) or "pad"
            if len(pil_images) > 1:
                # Training forces square-padding for multi-image samples on the
                # classic path (energon _process_images; dataset.py overwrite).
                image_aspect_ratio = "pad"
            images_tensor = process_images(
                pil_images,
                processor.image_processor,
                SimpleNamespace(image_aspect_ratio=image_aspect_ratio),
            )
            if isinstance(images_tensor, list):
                images_tensor = torch.stack(images_tensor, dim=0)
            dtype = next(model.parameters()).dtype
            gen_kwargs["images"] = images_tensor.to(device, dtype=dtype)
            gen_kwargs["image_sizes"] = [img.size for img in pil_images]

    if audios:
        if getattr(model.model, "audio_connector", None) is None:
            raise ValueError(
                "audio inputs were passed but this checkpoint has no audio pathway "
                "(config.audio_config is missing/disabled)"
            )
        gen_kwargs["audios"] = [
            load_audio_frames(source, data_args).to(device) for source in audios
        ]

    return gen_kwargs


def generate_response(
    model: Any,
    processor: VLMProcessor,
    query: str = "",
    images: Any = None,
    audios: Any = None,
    conv_mode: str | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    num_beams: int = 1,
    max_new_tokens: int = 512,
    stop_strings: list[str] | None = None,
    image_aspect_ratio: str | None = None,
    do_sample: bool | None = None,
) -> str:
    """Run one generation against a loaded model/processor pair.

    Args:
        model, processor: as returned by :func:`load_model`.
        query: user text; may contain ``<image>``/``<audio>`` placeholders for
            interleaved media (otherwise placeholders are prepended, one per
            item, images first — mirroring training's media-token injection).
        images: a PIL image, path, or a list thereof.
        audios: an audio path/file-like (wav/flac/ogg/mp3), or a list thereof.
        conv_mode: conversation template key; defaults to the
            ``conversation_version`` recorded in the checkpoint config.
        stop_strings: extra generation stop strings; for the ``plain``
            template defaults to ``["\\n"]`` (the training separator).
        image_aspect_ratio: legacy encoder models only; defaults to the value
            recorded in the checkpoint config at training time (fallback
            "pad"). Multi-image calls force "pad", mirroring training. The
            encoder-free path ignores this.
        do_sample: sampling switch. ``None`` (default) derives it from
            ``temperature > 0`` (greedy when temperature is 0); pass ``True``/
            ``False`` to control sampling independently of temperature (e.g.
            sample at the model's default temperature, or force greedy).

    Returns:
        The decoded response text.
    """
    images = _as_list(images)
    audios = _as_list(audios)
    data_args = _data_args_from_config(model.config)
    conv_mode = resolve_conv_mode(model.config, pretrained=None, conv_mode=conv_mode)

    if audios and not (
        conv_templates[conv_mode].sep_style == conversation_lib.SeparatorStyle.PLAIN
        or conv_templates[conv_mode].version == "qwen"
    ):
        # mirror check_audio_template_supported (training-side guard)
        raise NotImplementedError(
            "audio inputs are only supported with the 'plain' or 'qwen' conversation "
            f"templates, got conv_mode={conv_mode!r}"
        )

    # Self-describing checkpoints: rebuild the prompt with the SAME image layout
    # the model was trained on (plan 2026-06-10). Mirror the training order in
    # energon encode_sample: inject missing media placeholders FIRST, then
    # reposition, then inject <query>. Doing ensure_placeholders first matters
    # so a checkpoint trained with image_position != "keep" repositions even
    # when the caller relied on auto-injection (the query had no <image>). Only
    # the single-image case is repositioned (mirrors apply_image_position's
    # training-side guard); the seed mirrors the energon path (crc32 of the
    # user-typed query, before injection, so the layout is stable whether or not
    # the placeholder was supplied explicitly).
    image_token = str(getattr(model.config, "image_token", "<image>") or "<image>")
    image_position = str(getattr(model.config, "image_position", "keep") or "keep")
    position_seed = zlib.crc32(query.encode())
    query = ensure_placeholders(query, len(images), len(audios), data_args)
    if image_position != "keep" and query.count(image_token) == 1:
        _turns = [{"from": "human", "value": query}]
        apply_image_position(
            _turns,
            mode=image_position,
            image_token=image_token,
            seed=position_seed,
            # Mirror the training call sites (#8): protect the audio placeholder so
            # sandwich/random does not duplicate <audio> when image+audio are given
            # together, which would break its 1:1 feature count at splice time.
            protected_tokens=(data_args.audio_token,),
        )
        query = _turns[0]["value"]
    # BREEN port: emit one "<query>" per image at the trained placement, mirroring
    # the training data path so the splice inserts the learnable query block.
    if data_args.learnable_query_enabled and images:
        _turns = [{"from": "human", "value": query}]
        inject_query_placeholders(_turns, n_images=len(images), data_args=data_args)
        query = _turns[0]["value"]
    prompt = build_prompt(conv_mode, query, data_args)

    tokenizer = processor.tokenizer
    input_ids = tokenizer_multimodal_token(prompt, tokenizer, data_args, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).to(model.device)
    # Defense-in-depth: the splice inserts one image feature per <image> sentinel,
    # so a sentinel/image-count mismatch silently drops (or misaligns) images.
    # build_prompt + ensure_placeholders keep these equal; assert it so any future
    # regression fails loud instead of quietly answering on a subset of the images.
    n_sentinels = int((input_ids == data_args.image_token_index).sum().item())
    if n_sentinels != len(images):
        raise ValueError(
            f"prompt has {n_sentinels} {data_args.image_token!r} sentinel(s) but "
            f"{len(images)} image(s) were passed; the model splices one feature per "
            "sentinel, so a mismatch would silently drop/misalign images"
        )
    attention_mask = torch.ones_like(input_ids)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    gen_kwargs = prepare_media_inputs(
        model, processor, images, audios, data_args, image_aspect_ratio=image_aspect_ratio
    )

    conv = conv_templates[conv_mode]
    if stop_strings is None:
        if conv.sep_style == conversation_lib.SeparatorStyle.PLAIN:
            # preprocess_plain supervises `answer + "\n"`; everything after the
            # first newline is out-of-distribution continuation.
            stop_strings = ["\n"]
        elif getattr(conv, "stop_str", None):
            stop = conv.stop_str
            stop_strings = list(stop) if isinstance(stop, list) else [str(stop)]
        else:
            stop_strings = []
    eos_token_id = None
    if getattr(conv, "stop_token_ids", None):
        # e.g. llava_llama_3 records <|eot_id|>; merge with the model's eos.
        base_eos = model.generation_config.eos_token_id
        base_eos = [base_eos] if isinstance(base_eos, int) else list(base_eos or [])
        eos_token_id = sorted(set(base_eos) | set(conv.stop_token_ids))

    do_sample_effective = (temperature > 0) if do_sample is None else bool(do_sample)
    generate_kwargs: dict[str, Any] = dict(
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=do_sample_effective,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        **gen_kwargs,
    )
    if do_sample_effective:
        # temperature=0 with do_sample=True is invalid (division by zero); leave
        # it unset so the model's generation_config default applies in that case.
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
    if stop_strings:
        generate_kwargs["stop_strings"] = stop_strings
        generate_kwargs["tokenizer"] = tokenizer
    if eos_token_id is not None:
        generate_kwargs["eos_token_id"] = eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(input_ids, **generate_kwargs)

    # generate() runs from inputs_embeds, so output_ids contains only the new
    # tokens (no prompt echo).
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    for stop in stop_strings:
        outputs = outputs.split(stop)[0]
    return outputs.strip()


def eval_model(
    pretrained: str,
    query: str = "",
    image_path: str | list[str] | None = None,
    audio_path: str | list[str] | None = None,
    conv_mode: str | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    num_beams: int = 1,
    max_new_tokens: int = 512,
    image_aspect_ratio: str | None = None,
    bf16: bool = True,
    fp16: bool = False,
    attn_implementation: str = "sdpa",
    device: str | torch.device | None = None,
) -> str:
    """One-shot convenience wrapper: load the checkpoint, run a single
    generation, print and return the response.

    ``image_aspect_ratio`` (legacy encoder checkpoints only) is forwarded to
    :func:`generate_response`; see its docstring. Defaults to the value recorded
    in the checkpoint config; pass square/pad for checkpoints that predate the
    recording. The encoder-free path ignores it."""
    model, processor, _config = load_model(
        pretrained, bf16=bf16, fp16=fp16, attn_implementation=attn_implementation, device=device
    )
    conv_mode = resolve_conv_mode(model.config, pretrained=pretrained, conv_mode=conv_mode)

    # Back-compat: the original eval_model accepted "<image-placeholder>".
    if "<image-placeholder>" in query:
        data_args = _data_args_from_config(model.config)
        query = query.replace("<image-placeholder>", data_args.image_token)

    outputs = generate_response(
        model,
        processor,
        query=query,
        images=image_path,
        audios=audio_path,
        conv_mode=conv_mode,
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        image_aspect_ratio=image_aspect_ratio,
    )
    print(outputs)
    return outputs
