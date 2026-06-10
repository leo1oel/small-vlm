import copy
import json
import logging
import math
import os
import random
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, override

import soundfile as sf
import torch
import torchaudio
import transformers
import yaml
from PIL import Image
from torch.utils.data import Dataset
from transformers.image_processing_utils import BaseImageProcessor

from ..models import VLMProcessor
from ..models.image_processing_raw import RawImageProcessor
from ..utils import conversation as conversation_lib
from .data_arguments import DataArguments

log: logging.Logger = logging.getLogger(name=__name__)


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(
    target: torch.Tensor,
    tokenized_lens: list[int],
    speakers: list[str],
    data_args: DataArguments,
):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = data_args.ignore_index
    for tokenized_len, speaker in zip(tokenized_lens, speakers, strict=False):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = data_args.ignore_index
        cur_idx += tokenized_len


def _add_speaker_and_signal(
    header: str,
    source: Sequence[dict],
    get_conversation: bool = True,
):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def tokenizer_image_token(
    prompt: str,
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: DataArguments,
    return_tensors: str | None = None,
):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X: list[list[int]], sep: list[int]):
        return [ele for sublist in zip(X, [sep] * len(X), strict=False) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [data_args.image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


# ---------------------------------------------------------------------------
# Shared per-sample multimodal core — pure functions used by BOTH the local
# json dataset below and the energon streaming path (energon_dataset.py).
# ---------------------------------------------------------------------------


def _media_pattern(data_args: DataArguments) -> str:
    return (
        "(" + "|".join(re.escape(t) for t in (data_args.image_token, data_args.audio_token)) + ")"
    )


def tokenizer_multimodal_token(
    prompt: str,
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: DataArguments,
    return_tensors: str | None = None,
):
    """Generalization of tokenizer_image_token to multiple media placeholders:
    '<image>' -> image_token_index (-200), '<audio>' -> audio_token_index (-201).
    Behavior-identical to tokenizer_image_token for image-only prompts
    (including the keep-first-BOS-only handling)."""
    sentinel = {
        data_args.image_token: data_args.image_token_index,
        data_args.audio_token: data_args.audio_token_index,
    }
    input_ids: list[int] = []
    # re.split with a capturing group: even chunks are text (possibly ""), the
    # placeholders themselves come through as their own chunks.
    for i, chunk in enumerate(re.split(_media_pattern(data_args), prompt)):
        if chunk in sentinel:
            input_ids.append(sentinel[chunk])
            continue
        ids = tokenizer(chunk).input_ids
        # the tokenizer may prepend BOS to every chunk; keep it on the first only
        if i != 0 and len(ids) > 0 and ids[0] == tokenizer.bos_token_id:
            ids = ids[1:]
        input_ids.extend(ids)

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def inject_missing_media_tokens(
    conversations: list[dict],
    n_images: int,
    n_audios: int,
    data_args: DataArguments,
) -> None:
    """Reconcile media placeholders with the media a sample actually carries
    (per modality, in place). Datasets like ASR transcripts or bare caption
    pairs don't ship '<image>'/'<audio>' markers:
      - media present, no placeholder anywhere -> prepend to the first
        human/user turn (deterministic order: images then audios);
      - placeholder count matches -> leave the text as-is;
      - any other mismatch -> raise (ambiguous sample; surface, don't guess).
    """

    def text_key(turn: dict) -> str:
        return "value" if "value" in turn else "content"

    prefix = ""
    for token, n in ((data_args.image_token, n_images), (data_args.audio_token, n_audios)):
        found = sum(str(turn[text_key(turn)]).count(token) for turn in conversations)
        if found == n:
            continue
        if n > 0 and found == 0:
            prefix += (token + "\n") * n
        elif found > n:
            # Text mentions the literal placeholder more often than the sample
            # carries media (~0.1% of OneVision/FineVision: artifacts like a
            # prompt QUOTING "<audio>"). Neutralize the surplus occurrences —
            # bracket-swapped so the multimodal tokenizer no longer splits on
            # them — keeping the first n aligned with the actual media. This
            # must NOT raise: a raise inside energon's buffer-restore path
            # (which has no skip handler, unlike the streaming path) turns one
            # such buffered sample into a deterministic resume crash-loop.
            neutral = "[" + token.strip("<>") + "]"
            remaining = found - n
            for turn in reversed(conversations):
                if remaining <= 0:
                    break
                text = str(turn[text_key(turn)])
                while remaining > 0 and token in text:
                    head, _, tail = text.rpartition(token)
                    text = head + neutral + tail
                    remaining -= 1
                turn[text_key(turn)] = text
        else:
            raise ValueError(
                f"sample has {n} media item(s) for {token!r} but {found} placeholder(s) in its text"
            )
    if not prefix:
        return
    for turn in conversations:
        if (turn.get("from") or turn.get("role")) in ("human", "user"):
            turn[text_key(turn)] = prefix + turn[text_key(turn)]
            return
    raise ValueError("no human/user turn to inject media placeholders into")


def load_audio_frames(audio_source: Any, data_args: DataArguments) -> torch.Tensor:
    """Decode audio (a path or a file-like object) into model-ready frames:
    mono float32 @ audio_sampling_rate, shaped (T, samples_per_token).
    Longer than max_audio_tokens -> truncated (head kept); the tail partial
    frame is zero-padded (= the audio feature extractor's padding_value).
    Decode = soundfile (wav/flac/ogg/mp3 via libsndfile); resample =
    torchaudio.functional (pure torch sinc kernel, no torchcodec/ffmpeg)."""
    data, in_sr = sf.read(audio_source, dtype="float32", always_2d=True)  # (S, C)
    wav = torch.from_numpy(data).mean(dim=1)  # mono (S,)
    if in_sr != data_args.audio_sampling_rate:
        wav = torchaudio.functional.resample(wav, in_sr, data_args.audio_sampling_rate)
    samples_per_token = data_args.audio_samples_per_token
    if data_args.max_audio_tokens is not None:
        wav = wav[: data_args.max_audio_tokens * samples_per_token]
    if wav.numel() == 0:
        raise ValueError(f"empty audio: {audio_source}")
    remainder = wav.numel() % samples_per_token
    if remainder:
        wav = torch.cat([wav, wav.new_zeros(samples_per_token - remainder)])
    return wav.view(-1, samples_per_token)


def process_raw_image(image: Image.Image, processor: RawImageProcessor) -> tuple:
    """Encoder-free image entry: variable-resolution NaFlex patchification.
    No square-padding/resizing here — the processor preserves aspect ratio.
    Returns a 4-tuple (patches (N, patch_dim), position_ids (N, 2), size,
    modality) — one element longer than the classic 3-tuple, which is how the
    collator tells the two pipelines apart."""
    out = processor.preprocess(image)
    return out["pixel_values"][0], out["image_position_ids"][0], image.size, "image"


def expand2square(pil_img: Image.Image, background_color: tuple[int, int, int]) -> Image.Image:
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_classic_image(
    image: Image.Image, processor: BaseImageProcessor, image_aspect_ratio: str
) -> tuple:
    """Encoder-path (CLIP/SigLIP/DINO) image entry: fixed-resolution pixel
    grid via the HF image processor, optional square-padding first. Returns
    the classic 3-tuple (pixel_values, size, modality). Shared by the
    local-json dataset and the energon task encoder so both produce the
    exact same model inputs."""
    image_size = image.size
    if image_aspect_ratio == "pad":
        image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
    pixel_values = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    return pixel_values, image_size, "image"


def make_dummy_image_entry(image_processor: BaseImageProcessor) -> tuple:
    """Image entry for a sample with no image (text-only / audio-only) so the
    vision connector still runs every step (deepspeed/DDP unused-parameter
    guard); the splice consumes its feature zero-width — exact-zero gradient."""
    if isinstance(image_processor, RawImageProcessor):
        patches, position_ids = image_processor.get_dummy_inputs()
        return patches, position_ids, (1, 1), "text"
    crop_size = image_processor.crop_size
    if crop_size is None:
        crop_size = image_processor.size
    return (
        torch.zeros(1, 3, crop_size["height"], crop_size["width"]),
        (crop_size["width"], crop_size["height"]),
        "text",
    )


def make_dummy_audio_frames(data_args: DataArguments) -> torch.Tensor:
    """Audio counterpart of make_dummy_image_entry: one zero frame, consumed
    zero-width by the splice."""
    return torch.zeros(1, data_args.audio_samples_per_token)


def check_audio_template_supported() -> None:
    """Audio placeholders are wired into the 'plain' and 'qwen' templates only
    (the ones this project trains with); other templates would silently
    tokenize '<audio>' as literal text, so fail loudly instead."""
    conv = conversation_lib.default_conversation
    if conv.sep_style == conversation_lib.SeparatorStyle.PLAIN or conv.version == "qwen":
        return
    raise NotImplementedError(
        f"audio samples are only supported with the 'plain' or 'qwen' conversation "
        f"templates, but the active template is version={conv.version!r}"
    )


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources  # pyright: ignore

    for source in sources:
        for sentence in source:
            if data_args.image_token in sentence["value"]:
                sentence["value"] = sentence["value"].replace(data_args.image_token, "").strip()
                sentence["value"] = data_args.image_token + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(
                        data_args.image_token, "<Image>" + data_args.image_token + "</Image>"
                    )
            replace_token = data_args.image_token
            if data_args.use_start_end_tokens:
                replace_token = (
                    data_args.image_start_token + replace_token + data_args.image_end_token
                )
            sentence["value"] = sentence["value"].replace(data_args.image_token, replace_token)

    return sources  # pyright: ignore


def apply_image_position(
    conversations: list[dict],
    mode: str,
    image_token: str,
    seed: int | None = None,
) -> None:
    """Reposition the image placeholder inside human turns, in place (plan
    docs/superpowers/plans/2026-06-10-early-fusion-access-arms.md).

    Only human/user turns containing EXACTLY ONE image token and non-empty
    text are rewritten; everything else (gpt turns, image-only turns,
    multi-image turns) is left untouched. Modes:
      keep           - no-op (default; preserves both paths' current layout)
      question_first - "Q\\n<image>"
      sandwich       - "Q\\n<image>\\nQ"  (question repeated after the image)
      random         - seed-deterministic choice of first / middle / last

    Note: the question text is derived by removing the single image token and
    the horizontal spacing / own-line newline directly adjacent to it. Internal
    newlines in the prompt are preserved, so multi-line MCQ options survive
    intact ("<image>\\nQ?\\nA. cat\\nB. dog" -> "Q?\\nA. cat\\nB. dog"). A
    token embedded mid-line ("Look at <image> and answer.") collapses to a
    single separating space ("Look at and answer.") rather than gluing words.
    """
    if mode == "keep":
        return
    if mode not in ("question_first", "sandwich", "random"):
        raise ValueError(f"unknown image_position mode: {mode!r}")
    rng = random.Random(seed)
    for turn in conversations:
        if (turn.get("from") or turn.get("role")) not in ("human", "user"):
            continue
        key = "value" if "value" in turn else "content"
        text = str(turn[key])
        if text.count(image_token) != 1:
            continue
        # Remove the token plus its own-line newline (token-on-its-own-line
        # case), else replace token + adjacent horizontal whitespace with a
        # single space. Collapse stray horizontal-space runs but keep internal
        # newlines so multi-line MCQ options survive.
        tok = re.escape(image_token)
        question = re.sub(r"[ \t]*" + tok + r"[ \t]*\n", "", text)
        question = re.sub(r"[ \t]*" + tok + r"[ \t]*", " ", question)
        question = re.sub(r"[ \t]+", " ", question)
        question = "\n".join(line.strip() for line in question.split("\n")).strip()
        if not question:
            continue
        if mode == "question_first":
            new = f"{question}\n{image_token}"
        elif mode == "sandwich":
            new = f"{question}\n{image_token}\n{question}"
        else:  # random
            words = question.split()
            placements = ["first", "last"] + (["middle"] if len(words) >= 2 else [])
            choice = rng.choice(placements)
            if choice == "first":
                new = f"{image_token}\n{question}"
            elif choice == "last":
                new = f"{question}\n{image_token}"
            else:
                mid = len(words) // 2
                head, tail = " ".join(words[:mid]), " ".join(words[mid:])
                new = f"{head}\n{image_token}\n{tail}"
        turn[key] = new


def preprocess_llama_2(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: DataArguments,
    has_image: bool = False,
) -> dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt", data_args=data_args)
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets, strict=False):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = data_args.ignore_index
        for _, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = data_args.ignore_index

            cur_len += round_len
        target[cur_len:] = data_args.ignore_index

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = data_args.ignore_index
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


# Per-process cache: (id(tokenizer), has_image, media_tokens) -> prepared deepcopy.
# The deepcopy isolation is deliberate — adding '<image>' to the SHARED tokenizer
# would change len(tokenizer) and trigger an unwanted embedding resize in vlm.py.
_PREPROCESS_TOKENIZER_CACHE: dict[tuple, transformers.PreTrainedTokenizer] = {}


def _get_preprocess_tokenizer(
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool,
    media_tokens: tuple[str, ...] = ("<image>",),
) -> transformers.PreTrainedTokenizer:
    key = (id(tokenizer), has_image, media_tokens)
    cached = _PREPROCESS_TOKENIZER_CACHE.get(key)
    if cached is None:
        cached = copy.deepcopy(tokenizer)
        if has_image:
            cached.add_tokens(list(media_tokens), special_tokens=True)
        _PREPROCESS_TOKENIZER_CACHE[key] = cached
    return cached


def preprocess_qwen(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: DataArguments,
    has_image: bool = False,
    system_message: str = "You are a helpful assistant.",
) -> dict:
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image/audio tokens to tokenizer as special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    # NOTE: has_image is really "has media" here — the dataset passes True for
    # samples carrying images and/or audio.
    tokenizer = _get_preprocess_tokenizer(
        tokenizer, has_image, (data_args.image_token, data_args.audio_token)
    )

    # Only resolve the media token ids when they were actually added — for
    # text-only batches convert_tokens_to_ids would return unk/None and could
    # falsely match real text tokens below.
    image_token_index = (
        tokenizer.convert_tokens_to_ids(data_args.image_token) if has_image else None
    )
    audio_token_index = (
        tokenizer.convert_tokens_to_ids(data_args.audio_token) if has_image else None
    )
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx = [198, im_start, im_end]
    # nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for _, source in enumerate(sources):
        # A leading system turn (e.g. ~9% of LLaVA-OneVision-1.5-Instruct, of
        # which about half are empty-string artifacts) becomes this sample's
        # system message when non-empty; empty ones fall back to the default.
        # Either way it must not reach the roles[...] lookup below, which only
        # knows human/gpt (KeyError: 'system' used to crash the worker here).
        sample_system = system_message
        first_role = source[0].get("from") or source[0].get("role")
        if first_role == "system":
            sys_text = source[0]["value"] if "value" in source[0] else source[0].get("content")
            if isinstance(sys_text, str) and sys_text.strip():
                sample_system = sys_text
            source = source[1:]
        if source and roles.get(source[0]["from"], source[0]["from"]) != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        # transformers v5 defaults apply_chat_template(return_dict=True), which
        # returns a BatchEncoding; pass return_dict=False to keep the v4 behavior
        # of returning a plain list[int] that we concatenate below.
        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": sample_system}], return_dict=False
        )
        target += [data_args.ignore_index] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except KeyError:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv, return_dict=False)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [data_args.ignore_index] * len(encode_id)
            else:
                target += encode_id

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if image_token_index is not None and encode_id == image_token_index:
                input_id[idx] = data_args.image_token_index
            elif audio_token_index is not None and encode_id == audio_token_index:
                input_id[idx] = data_args.audio_token_index
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_llama3(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: DataArguments,
    has_image: bool = False,
    system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> dict:
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = _get_preprocess_tokenizer(tokenizer, has_image)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    # bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    # start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    # end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    # eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = [
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "\n\n",
    ]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    # After update, calling tokenizer of llama3 will
    # auto add bos id for the tokens. ヽ(｀⌒´)ﾉ
    # def safe_tokenizer_llama3(text):
    #     input_ids = tokenizer(text).input_ids
    #     if input_ids[0] == bos_token_id:
    #         input_ids = input_ids[1:]
    #     return input_ids

    # nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    # Apply prompt templates
    input_ids, targets = [], []
    for _, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        # transformers v5 defaults apply_chat_template(return_dict=True), which
        # returns a BatchEncoding; pass return_dict=False to keep the v4 behavior
        # of returning a plain list[int] that we concatenate below.
        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}], return_dict=False
        )
        target += [data_args.ignore_index] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except KeyError:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)

            conv = [{"role": role, "content": content}]
            # First is bos token we don't need here
            encode_id = tokenizer.apply_chat_template(conv, return_dict=False)[1:]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [data_args.ignore_index] * len(encode_id)
            else:
                target += encode_id

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = data_args.image_token_index
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_gemma(
    sources: list[list[dict[str, str]]],
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: DataArguments,
    has_image: bool = False,
) -> dict:
    conv: conversation_lib.Conversation = conversation_lib.default_conversation.copy()
    roles: dict[str, str] = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations: list[str] = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source: list[dict[str, str]] = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role: str = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids: torch.Tensor = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets: torch.Tensor = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.GEMMA

    # Mask target
    sep: str = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets, strict=True):
        total_len: int = int(target.ne(tokenizer.pad_token_id).sum())

        rounds: list[str] = conversation.split(conv.sep)
        re_rounds = []
        for conv_idx in range(0, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))

        cur_len = 1  # Ignore <bos>
        target[:cur_len] = data_args.ignore_index
        for _, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep  # Re-append sep because split on this
            # Now "".join(parts)==rou

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1  # Ignore <bos>
                instruction_len = (
                    len(tokenizer_image_token(parts[0], tokenizer)) - 1
                )  # Ignore <bos>
            else:
                round_len = len(tokenizer(rou).input_ids) - 1  # Ignore <bos>
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1  # Ignore <bos>

            round_len += 2  # sep: <end_of_turn>\n takes 2 tokens
            target[cur_len : cur_len + instruction_len] = data_args.ignore_index
            cur_len += round_len

        target[cur_len:] = data_args.ignore_index

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = data_args.ignore_index
                print(f"warning: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: DataArguments,
    has_image: bool = False,
) -> dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt", data_args=data_args)
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets, strict=False):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = data_args.ignore_index
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer, data_args=data_args))
                instruction_len = (
                    len(tokenizer_image_token(parts[0], tokenizer, data_args=data_args)) - 2
                )
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = data_args.ignore_index

            cur_len += round_len
        target[cur_len:] = data_args.ignore_index

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = data_args.ignore_index
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: DataArguments,
    has_image: bool = False,
) -> dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets, strict=False):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))  # user + gpt
        cur_len = 0
        target[:cur_len] = data_args.ignore_index
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, "legacy", False):
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = data_args.ignore_index

            cur_len += round_len
        target[cur_len:] = data_args.ignore_index

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = data_args.ignore_index
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: DataArguments,
) -> dict:
    # add end signal and concatenate together. The human side is reduced to
    # just the media placeholders (in their original order) — the plain
    # pretrain recipe drops the human text entirely. Image-only samples
    # produce exactly the legacy "<image>" behavior.
    conversations = []
    for source in sources:
        assert len(source) == 2
        placeholders = re.findall(_media_pattern(data_args), source[0]["value"])
        assert placeholders, "plain preprocessing expects at least one media placeholder"
        source[0]["value"] = "".join(placeholders)
        conversation = (
            source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
        )
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [
        tokenizer_multimodal_token(prompt, tokenizer, return_tensors="pt", data_args=data_args)
        for prompt in conversations
    ]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources, strict=False):
        tokenized_len = len(
            tokenizer_multimodal_token(source[0]["value"], tokenizer, data_args=data_args)
        )
        target[:tokenized_len] = data_args.ignore_index

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: DataArguments,
    has_image: bool = False,
) -> dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with ignore_index.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer, data_args)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, data_args, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, data_args, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, data_args, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, data_args, has_image=has_image)
    if conversation_lib.default_conversation.version == "gemma":
        return preprocess_gemma(sources, tokenizer, data_args, has_image=has_image)
    if conversation_lib.default_conversation.version == "llama_v3":
        return preprocess_llama3(sources, tokenizer, data_args, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts: list[str]) -> list[int]:
        return [len(tokenizer_image_token(prompt, tokenizer, data_args)) for prompt in prompts]

    if has_image:
        input_ids = [
            tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            for prompt in conversations
        ]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources, strict=False):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])  # pyright: ignore
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)[  # pyright: ignore
                "input_ids_lens"
            ]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers, data_args)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, processor: VLMProcessor, data_args: DataArguments):
        super().__init__()
        self.tokenizer: transformers.PreTrainedTokenizer = processor.tokenizer
        self.image_processor: BaseImageProcessor = processor.image_processor
        self.data_args: DataArguments = data_args
        self.list_data_dict: list = []

        # Handle multiple JSON files specified in the data_path
        if "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            file_names = file_pattern.split(",")
            log.info(f"Loading {file_names} from {base_path}")
            data_args.dataset_paths = []
            for file_name in file_names:
                data_args.dataset_paths.append(f"{base_path}{file_name}.json")
                full_path = f"{base_path}{file_name}.json"
                log.info(f"Loading {full_path}")
                with open(full_path) as file:
                    cur_data_dict = json.load(file)
                    log.info(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                    self.list_data_dict.extend(cur_data_dict)
        elif data_path.endswith(".yaml"):
            with open(data_path) as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999
                data_args.dataset_paths = [dataset.get("json_path") for dataset in datasets]
                for dataset in datasets:
                    json_path = dataset.get("json_path")
                    sampling_strategy = dataset.get("sampling_strategy", "all")
                    sampling_number = None

                    log.info(f"Loading {json_path} with {sampling_strategy} sampling strategy")

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path) as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path) as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(
                                int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100
                            )
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]

                    log.info(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            data_args.dataset_paths = [data_path]
            log.info(f"Loading {data_path}")
            with open(data_path) as file:
                cur_data_dict = json.load(file)
                log.info(f"Loaded {len(cur_data_dict)} samples from {data_path}")
                self.list_data_dict.extend(cur_data_dict)

        log.info(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        log.info("Formatting inputs...Skip in lazy mode")

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            # heuristic media token estimates, only used for length grouping
            media_tokens = (128 if "image" in sample else 0) + (128 if "audio" in sample else 0)
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"]) + media_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            assert cur_len > 0, f"Conversation length is 0 for {sample}"

            if (
                "image" in sample
                or "video" in sample
                or "audio" in sample
                or self.data_args.early_mix_text
            ):
                length_list.append(cur_len)
            else:
                length_list.append(-cur_len)
        return length_list

    def process_image(self, image_file: str, overwrite_image_aspect_ratio: str | None = None):
        image_folder = self.data_args.image_folder
        processor = self.image_processor
        # print(f"\n\nInspecting the image path, folder = {image_folder}, image={image_file}\n\n")
        try:
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        except Exception as exn:
            print(f"Failed to open image {image_file}. Exception:", exn)
            raise exn

        if isinstance(processor, RawImageProcessor):
            # Encoder-free path: variable resolution, aspect ratio preserved by
            # the processor itself — expand2square/aspect handling must NOT run.
            return process_raw_image(image, processor)

        image_aspect_ratio = self.data_args.image_aspect_ratio
        if overwrite_image_aspect_ratio is not None:
            image_aspect_ratio = overwrite_image_aspect_ratio
        # (highres / anyres / crop_split variants removed with the legacy
        # multi-crop pipelines; see git history if they come back.)
        return process_classic_image(image, processor, image_aspect_ratio)

    @override
    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3
        # num_final_retries = 300

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                # print(self._get_item(i))
                # exit(1)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",  # pyright: ignore
                    e,
                )
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i: int | str) -> dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        sample = self.list_data_dict[i]
        if "video" in sample:
            raise NotImplementedError("video samples are not supported")

        # --- media loading: image / audio are independently optional ---
        image = None
        if "image" in sample:
            image_file = sample["image"]
            if isinstance(image_file, list):
                image = [self.process_image(f) for f in image_file]
                # Handling multi images
                # overwrite to process with simple pad
                # (CLIP path only: the raw processor is natively variable-size)
                if len(image_file) > 1 and not isinstance(self.image_processor, RawImageProcessor):
                    image = [self.process_image(f, "pad") for f in image_file]
                    image = [[im[0], im[1], "image"] for im in image]
            else:
                image = [self.process_image(image_file)]

        audio = None
        if "audio" in sample:
            if not self.data_args.audio_enabled:
                raise ValueError(
                    f"sample {sample.get('id', i)} carries audio but the model's audio "
                    "pathway is off — set model.audio.enabled=true"
                )
            check_audio_template_supported()
            audio_files = (
                sample["audio"] if isinstance(sample["audio"], list) else [sample["audio"]]
            )
            audio_folder = self.data_args.audio_folder or ""
            audio = [
                load_audio_frames(os.path.join(audio_folder, f), self.data_args)
                for f in audio_files
            ]

        sources = copy.deepcopy([e["conversations"] for e in sources])
        inject_missing_media_tokens(
            sources[0],
            n_images=len(image) if image is not None else 0,
            n_audios=len(audio) if audio is not None else 0,
            data_args=self.data_args,
        )
        if image is not None:
            sources = preprocess_multimodal(sources, self.data_args)
            for source in sources:
                apply_image_position(
                    source,
                    mode=self.data_args.image_position,
                    image_token=self.data_args.image_token,
                    seed=i,
                )

        # elif "video" in sources[0]:
        #     video_file = self.list_data_dict[i]["video"]
        #     video_folder = self.data_args.video_folder
        #     video_file = os.path.join(video_folder, video_file)
        #     # suffix = video_file.split(".")[-1]
        #     if not os.path.exists(video_file):
        #         print(f"File {video_file} not exist!")

        #     try:
        #         if "shareVideoGPTV" in video_file:
        #             frame_files = [
        #                 os.path.join(video_file, f)
        #                 for f in os.listdir(video_file)
        #                 if os.path.isfile(os.path.join(video_file, f))
        #             ]
        #             frame_files.sort()  # Ensure the frames are sorted if they are named sequentially

        #             # TODO: Hard CODE: Determine the indices for uniformly sampling 10 frames
        #             if self.data_args.force_sample:
        #                 num_frames_to_sample = self.data_args.frames_upbound
        #             else:
        #                 num_frames_to_sample = 10

        #             avg_fps = 2

        #             total_frames = len(frame_files)
        #             sampled_indices = np.linspace(
        #                 0, total_frames - 1, num_frames_to_sample, dtype=int
        #             )

        #             frame_time = [i / 2 for i in sampled_indices]
        #             frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

        #             video_time = total_frames / avg_fps

        #             # Read and store the sampled frames
        #             video = []
        #             for idx in sampled_indices:
        #                 frame_path = frame_files[idx]
        #                 try:
        #                     with Image.open(frame_path) as img:
        #                         frame = img.convert("RGB")
        #                         video.append(frame)
        #                 except OSError:
        #                     print(f"Failed to read frame at path: {frame_path}")
        #         else:
        #             video, video_time, frame_time, num_frames_to_sample = process_video_with_decord(
        #                 video_file, self.data_args
        #             )

        #         processor = self.image_processor
        #         image = processor.preprocess(video, return_tensors="pt")["pixel_values"]
        #         if self.data_args.add_time_instruction:
        #             time_instruction = f"The video lasts for {video_time:.2f} seconds, and {num_frames_to_sample} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
        #             sources[0]["conversations"][0]["value"] = (
        #                 f"{self.data_args.image_token}\n{time_instruction}\n{sources[0]['conversations'][0]['value'].replace(self.data_args.image_token, '')}"
        #             )
        #         image = [(image, video[0].size, "video")]
        #         sources = preprocess_multimodal(
        #             copy.deepcopy([e["conversations"] for e in sources]), self.data_args
        #         )
        #         # print(sources)
        #     except Exception as e:
        #         print(f"Error: {e}")
        #         print(f"Failed to read video file: {video_file}")
        #         return self._get_item(i + 1)

        # has_image historically gates the media-aware tokenization path; audio
        # placeholders ride the same path, so it really means "has media".
        has_media = image is not None or audio is not None
        data_dict = preprocess(sources, self.tokenizer, self.data_args, has_image=has_media)

        if "prompt" in data_dict:
            prompt = data_dict["prompt"]
        else:
            prompt = None

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if image is not None:
            data_dict["image"] = image  # pyright: ignore
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal:
            # dummy entry, consumed zero-width by the splice
            data_dict["image"] = [make_dummy_image_entry(self.image_processor)]
        # audio exist in the data (dummy when the audio pathway is on but the
        # sample has none — same zero-width trick as the image dummy)
        if audio is not None:
            data_dict["audio"] = audio  # pyright: ignore
        elif self.data_args.audio_enabled:
            data_dict["audio"] = [make_dummy_audio_frames(self.data_args)]  # pyright: ignore
        # prompt exist in the data
        if prompt is not None:
            data_dict["prompt"] = prompt

        data_dict["id"] = self.list_data_dict[i].get("id", i)

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    ignore_index: int = -100

    def pad_sequence(
        self, input_ids: torch.Tensor | list[torch.Tensor], batch_first: bool, padding_value: int
    ):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0  # This gets the best result. Don't know why.
        input_ids = self.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = self.pad_sequence(labels, batch_first=True, padding_value=self.ignore_index)
        batch = dict(
            input_ids=input_ids,
            labels=labels.long() if labels.dtype == torch.int32 else labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "image" in instances[0]:
            flat = [im for instance in instances for im in instance["image"]]
            if len(flat[0]) == 4:
                # encoder-free entries: (patches, position_ids, size, modality).
                # Lists, never stacked — images vary in patch count N.
                batch["images"] = [im[0] for im in flat]
                batch["image_position_ids"] = [im[1] for im in flat]
                batch["image_sizes"] = [im[2] for im in flat]
            else:
                # classic 3-tuple entries: (pixel_values, size, modality)
                batch["image_sizes"] = [im[1] for im in flat]
                # batch["modalities"] = [im[2] for im in flat]
                images = [im[0] for im in flat]
                batch["images"] = images

                # if all(x is not None and x.shape == images[0].shape for x in images):
                #     batch["images"] = torch.stack(images)
                # else:
                #     batch["images"] = images

        if "audio" in instances[0]:
            # one (T_i, samples_per_token) tensor per audio, flattened in batch
            # order — exactly the queue layout the splice consumes
            batch["audios"] = [a for instance in instances for a in instance["audio"]]

        # if "prompt" in instances[0]:
        #     batch["prompts"] = [instance["prompt"] for instance in instances]

        return batch


def make_supervised_data_module(
    processor: VLMProcessor,
    data_args: DataArguments,
) -> dict:
    """Make dataset and collator for supervised fine-tuning."""
    if not data_args.data_path:
        raise ValueError(
            "dataset.path is required for dataset.type='json' (it is optional in "
            "the schema only because dataset.type='energon' does not use it)"
        )
    train_dataset = LazySupervisedDataset(
        processor=processor, data_path=data_args.data_path, data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=processor.tokenizer, ignore_index=data_args.ignore_index
    )
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
