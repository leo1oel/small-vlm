import copy
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, override

import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset
from transformers.image_processing_utils import BaseImageProcessor

from ..utils import conversation as conversation_lib
from .data_arguments import DataArguments


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_max_length,
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
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
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
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert data_args.image_token in source[0]["value"]
        source[0]["value"] = data_args.image_token
        conversation = (
            source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
        )
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt", data_args=data_args)
        for prompt in conversations
    ]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources, strict=False):
        tokenized_len = len(
            tokenizer_image_token(source[0]["value"], tokenizer, data_args=data_args)
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
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer, data_args)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, data_args, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, data_args, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, data_args, has_image=has_image)
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
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments
    ):
        super().__init__()
        list_data_dict = json.load(open(data_path))

        # rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer: transformers.PreTrainedTokenizer = tokenizer
        self.list_data_dict: list[Any] = list_data_dict
        self.data_args: DataArguments = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    @override
    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        sources: list[dict] = self.list_data_dict[i]
        if isinstance(i, int):  # pyright: ignore
            sources = [sources]  # pyright: ignore
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            image_folder = self.data_args.image_folder
            processor: BaseImageProcessor = self.data_args.image_preprocessor  # pyright: ignore
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            sources = preprocess_multimodal(  # pyright: ignore
                copy.deepcopy([e["conversations"] for e in sources]), self.data_args
            )
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            self.data_args,
            has_image=("image" in self.list_data_dict[i]),
        )
        if isinstance(i, int):  # pyright: ignore
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image  # pyright: ignore
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = torch.zeros(3, crop_size["height"], crop_size["width"])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    data_args: DataArguments

    def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.data_args.ignore_index
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: DataArguments,
) -> dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
