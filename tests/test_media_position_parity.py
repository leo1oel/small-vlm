"""#13 — `apply_image_position` must not duplicate / drop non-image media
placeholders; #12 — the collator must not silently truncate a media sentinel.

Both are energon-path parity fixes (the energon task encoder wires
`protected_tokens` into `apply_image_position` and `media_token_ids` into
`DataCollatorForSupervisedDataset`). The unit tests here exercise the two
library functions directly so they are hermetic (no tokenizer / Azure).
"""

import pytest

torch = pytest.importorskip("torch")

try:
    from vlm.data.dataset import DataCollatorForSupervisedDataset, apply_image_position
except ModuleNotFoundError as e:  # pragma: no cover - slim envs
    pytest.skip(f"vlm package not importable here: {e}", allow_module_level=True)

IMG = "<image>"
AUD = "<audio>"


# ---------------------------------------------------------------------------
# #13 apply_image_position: protected (audio) tokens keep their count.
# ---------------------------------------------------------------------------


def test_sandwich_does_not_duplicate_protected_audio():
    convs = [{"from": "human", "value": f"{IMG}\n{AUD}\nwhat is said?"}]
    apply_image_position(
        convs, mode="sandwich", image_token=IMG, seed=0, protected_tokens=(AUD,)
    )
    text = convs[0]["value"]
    # the image question is duplicated (sandwich) but the audio placeholder is not
    assert text.count(AUD) == 1
    assert text.count(IMG) == 1
    assert text.count("what is said?") == 2  # question repeated around the image


def test_sandwich_without_protection_would_duplicate_audio():
    # Guard documenting the bug (#13): WITHOUT protected_tokens the audio token
    # rides along in the duplicated question text.
    convs = [{"from": "human", "value": f"{IMG}\n{AUD}\nwhat is said?"}]
    apply_image_position(convs, mode="sandwich", image_token=IMG, seed=0)
    assert convs[0]["value"].count(AUD) == 2  # the bug the fix prevents


def test_question_first_preserves_audio_count():
    convs = [{"from": "human", "value": f"{IMG}\n{AUD}\ndescribe"}]
    apply_image_position(
        convs, mode="question_first", image_token=IMG, seed=0, protected_tokens=(AUD,)
    )
    assert convs[0]["value"].count(AUD) == 1
    assert convs[0]["value"].count(IMG) == 1


def test_random_preserves_audio_count():
    for seed in range(8):
        convs = [{"from": "human", "value": f"{IMG}\n{AUD}\na b c d"}]
        apply_image_position(
            convs, mode="random", image_token=IMG, seed=seed, protected_tokens=(AUD,)
        )
        assert convs[0]["value"].count(AUD) == 1, seed
        assert convs[0]["value"].count(IMG) == 1, seed


def test_image_only_turn_with_audio_left_untouched():
    # No real question text -> the rewrite is a no-op and the ORIGINAL turn
    # (image + audio) must survive verbatim, counts intact.
    convs = [{"from": "human", "value": f"{IMG}\n{AUD}"}]
    apply_image_position(
        convs, mode="sandwich", image_token=IMG, seed=0, protected_tokens=(AUD,)
    )
    assert convs[0]["value"].count(AUD) == 1
    assert convs[0]["value"].count(IMG) == 1


def test_no_protected_tokens_matches_legacy_behavior():
    # Plain image+text turn, no audio: protected_tokens is irrelevant and the
    # sandwich output is exactly the legacy layout.
    convs = [{"from": "human", "value": f"{IMG}\nwhat is this?"}]
    apply_image_position(
        convs, mode="sandwich", image_token=IMG, seed=0, protected_tokens=(AUD,)
    )
    assert convs[0]["value"] == "what is this?\n<image>\nwhat is this?"


# ---------------------------------------------------------------------------
# #12 collator: truncation past model_max_length must not drop a media sentinel.
# ---------------------------------------------------------------------------


class _Tok:
    """Minimal tokenizer stand-in for the collator (only the attributes it reads)."""

    def __init__(self, model_max_length: int):
        self.model_max_length = model_max_length
        self.pad_token_id = 0
        self.padding_side = "right"


IMG_ID, AUD_ID = -200, -201


def _instance(input_ids: list[int], idx: str = "s0") -> dict:
    t = torch.tensor(input_ids, dtype=torch.long)
    return {"input_ids": t, "labels": t.clone(), "id": idx}


def test_collator_raises_when_truncation_drops_sentinel():
    # image sentinel sits at position 5, model_max_length=4 -> it is dropped.
    collator = DataCollatorForSupervisedDataset(
        tokenizer=_Tok(4), media_token_ids=[IMG_ID, AUD_ID]
    )
    inst = _instance([1, 2, 3, 4, 5, IMG_ID, 6])
    with pytest.raises(ValueError, match="media sentinel"):
        collator([inst])


def test_collator_ok_when_sentinel_survives_truncation():
    # sentinel at position 1, well within model_max_length -> no raise.
    collator = DataCollatorForSupervisedDataset(
        tokenizer=_Tok(8), media_token_ids=[IMG_ID, AUD_ID]
    )
    inst = _instance([1, IMG_ID, 2, 3, 4, 5, 6, 7, 8, 9])  # len 10 > 8, but sentinel kept
    batch = collator([inst])
    assert batch["input_ids"].shape[1] == 8


def test_collator_no_guard_without_media_token_ids():
    # Back-compat: media_token_ids unset -> no guard, legacy truncation only.
    collator = DataCollatorForSupervisedDataset(tokenizer=_Tok(4))
    inst = _instance([1, 2, 3, 4, 5, IMG_ID, 6])
    batch = collator([inst])  # no raise
    assert batch["input_ids"].shape[1] == 4


def test_collator_no_raise_when_within_limit():
    collator = DataCollatorForSupervisedDataset(
        tokenizer=_Tok(16), media_token_ids=[IMG_ID]
    )
    inst = _instance([1, IMG_ID, 2, 3])
    batch = collator([inst])
    assert batch["input_ids"].shape[1] == 4
