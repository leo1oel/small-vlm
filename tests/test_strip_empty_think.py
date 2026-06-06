"""Tests for dataset.strip_empty_think: dropping the empty `<think>\n\n</think>`
prefix that distillation-style caption data (e.g. Bee-Training-Data-Stage1)
prepends to every assistant turn. The strip happens in the energon
messages -> conversations bridge so the boilerplate never reaches the loss."""

import pytest

try:
    from vlm.config import DatasetConfig, ModelConfig
    from vlm.data.data_arguments import DataArguments, get_data_args
    from vlm.data.energon_dataset import messages_to_conversations
except ModuleNotFoundError as e:  # pragma: no cover - slim envs
    pytest.skip(f"vlm package not importable here: {e}", allow_module_level=True)

CAPTION = "The image shows a cup of coffee."
EMPTY_THINK = "<think>\n\n</think>\n\n"


def _convs(assistant_content, strip: bool, user_content="What is this?"):
    data_args = DataArguments(strip_empty_think=strip)
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    return messages_to_conversations(messages, data_args)


def test_strips_empty_think_prefix_from_assistant():
    convs = _convs(EMPTY_THINK + CAPTION, strip=True)
    assert convs[1] == {"from": "gpt", "value": CAPTION}


def test_default_off_passes_through_verbatim():
    convs = _convs(EMPTY_THINK + CAPTION, strip=False)
    assert convs[1]["value"] == EMPTY_THINK + CAPTION


def test_non_empty_think_block_is_preserved():
    reasoned = "<think>\nlet me look closely\n</think>\n\n" + CAPTION
    convs = _convs(reasoned, strip=True)
    assert convs[1]["value"] == reasoned


def test_user_turns_are_never_touched():
    convs = _convs(CAPTION, strip=True, user_content=EMPTY_THINK + "describe")
    assert convs[0]["value"] == EMPTY_THINK + "describe"


def test_strips_from_typed_content_list():
    content = [{"type": "text", "text": EMPTY_THINK + CAPTION}]
    convs = _convs(content, strip=True)
    assert convs[1] == {"from": "gpt", "value": CAPTION}


def test_tolerates_whitespace_variants_inside_empty_block():
    convs = _convs("<think>  \n </think>\n" + CAPTION, strip=True)
    assert convs[1]["value"] == CAPTION


def test_assistant_without_prefix_unchanged():
    convs = _convs(CAPTION, strip=True)
    assert convs[1]["value"] == CAPTION


def test_only_boilerplate_strips_to_empty_string():
    # Pin the degenerate case: a turn that is ONLY the empty think block
    # becomes "" (empty loss target). Absent from Bee Stage-1 (0/43k measured)
    # — accepted behavior, not silently guarded.
    convs = _convs(EMPTY_THINK, strip=True)
    assert convs[1]["value"] == ""


def test_get_data_args_plumbs_strip_empty_think():
    dataset_config = DatasetConfig(strip_empty_think=True)
    model_config = ModelConfig()
    model_config.visual_encoder.hf_name = None  # encoder-free: no HF lookup
    data_args = get_data_args(dataset_config, model_config)
    assert data_args.strip_empty_think is True
