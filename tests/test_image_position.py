"""Tests for the image-position ordering transform (plan 2026-06-10)."""

from vlm.data.dataset import apply_image_position

TOK = "<image>"


def conv(*turns):
    return [{"from": f, "value": v} for f, v in turns]


def test_keep_is_identity():
    c = conv(("human", f"{TOK}\nWhat is this?"), ("gpt", "A cat."))
    apply_image_position(c, mode="keep", image_token=TOK, seed=0)
    assert c[0]["value"] == f"{TOK}\nWhat is this?"


def test_question_first_moves_image_after_text():
    c = conv(("human", f"{TOK}\nWhat is this?"), ("gpt", "A cat."))
    apply_image_position(c, mode="question_first", image_token=TOK, seed=0)
    assert c[0]["value"] == f"What is this?\n{TOK}"
    assert c[1]["value"] == "A cat."  # gpt turns untouched


def test_sandwich_repeats_question():
    c = conv(
        ("human", f"{TOK}\nWhat is this?"),
    )
    apply_image_position(c, mode="sandwich", image_token=TOK, seed=0)
    assert c[0]["value"] == f"What is this?\n{TOK}\nWhat is this?"


def test_image_token_inside_text_is_extracted():
    c = conv(
        ("human", f"Look at {TOK} and answer."),
    )
    apply_image_position(c, mode="question_first", image_token=TOK, seed=0)
    # hard contract: exactly one token, moved to the end, no glued words
    assert c[0]["value"].count(TOK) == 1
    assert c[0]["value"].endswith(TOK)


def test_multiline_mcq_options_preserved():
    c = conv(
        ("human", f"{TOK}\nWhich animal?\nA. cat\nB. dog"),
    )
    apply_image_position(c, mode="question_first", image_token=TOK, seed=0)
    assert c[0]["value"] == f"Which animal?\nA. cat\nB. dog\n{TOK}"
    c2 = conv(
        ("human", f"{TOK}\nWhich animal?\nA. cat\nB. dog"),
    )
    apply_image_position(c2, mode="sandwich", image_token=TOK, seed=0)
    assert c2[0]["value"] == (
        f"Which animal?\nA. cat\nB. dog\n{TOK}\nWhich animal?\nA. cat\nB. dog"
    )


def test_double_application_is_guarded_by_fresh_copies():
    """The data paths deep-copy/rebuild conversations per access, so the
    transform never sees its own output. Pin the upstream contract that
    makes this safe: LazySupervisedDataset deep-copies before transforming."""
    import inspect

    from vlm.data import dataset as ds

    src = inspect.getsource(ds.LazySupervisedDataset)
    deepcopy_pos = src.find("copy.deepcopy")
    apply_pos = src.find("apply_image_position")
    assert deepcopy_pos != -1 and apply_pos != -1 and deepcopy_pos < apply_pos


def test_random_is_deterministic_per_seed():
    base = conv(
        ("human", f"{TOK}\nIs the red car left of the blue truck?"),
    )
    a = [dict(t) for t in base]
    b = [dict(t) for t in base]
    apply_image_position(a, mode="random", image_token=TOK, seed=1234)
    apply_image_position(b, mode="random", image_token=TOK, seed=1234)
    assert a[0]["value"] == b[0]["value"]
    assert a[0]["value"].count(TOK) == 1


def test_random_varies_across_seeds():
    vals = set()
    for seed in range(40):
        c = conv(
            ("human", f"{TOK}\nIs the red car left of the blue truck?"),
        )
        apply_image_position(c, mode="random", image_token=TOK, seed=seed)
        vals.add(c[0]["value"])
    assert len(vals) >= 2  # first/middle/last all reachable over 40 seeds


def test_image_only_turn_untouched():
    c = conv(
        ("human", TOK),
    )
    apply_image_position(c, mode="sandwich", image_token=TOK, seed=0)
    assert c[0]["value"] == TOK


def test_multi_image_turn_untouched():
    v = f"{TOK}\n{TOK}\nCompare these."
    c = conv(
        ("human", v),
    )
    apply_image_position(c, mode="sandwich", image_token=TOK, seed=0)
    assert c[0]["value"] == v


def test_no_image_turn_untouched():
    c = conv(("human", "Hello"), ("gpt", "Hi"))
    apply_image_position(c, mode="sandwich", image_token=TOK, seed=0)
    assert c[0]["value"] == "Hello"


def test_unknown_mode_raises():
    import pytest

    c = conv(
        ("human", f"{TOK}\nQ?"),
    )
    with pytest.raises(ValueError):
        apply_image_position(c, mode="banana", image_token=TOK, seed=0)


def test_content_key_supported():
    c = [{"role": "user", "content": f"{TOK}\nQ?"}]
    apply_image_position(c, mode="question_first", image_token=TOK, seed=0)
    assert c[0]["content"] == f"Q?\n{TOK}"


def test_dataset_config_has_image_position_default_keep():
    from vlm.config.config_schema import DatasetConfig
    from vlm.data.data_arguments import DataArguments

    assert DatasetConfig().image_position == "keep"
    assert DataArguments(image_position="sandwich").image_position == "sandwich"
