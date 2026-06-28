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
# #12 collator: truncation past model_max_length must not silently drop a media
# sentinel. The guard is crash-loop-safe — it realigns the truncated sample's
# media feature lists with the surviving sentinels rather than RAISING (a raise
# inside energon's buffer-restore path, which has no skip handler, would
# deterministically crash-loop on resume).
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


def _img_entry(size_tag: int) -> tuple:
    """Minimal 3-tuple image feature entry (pixel_values, size, modality); the
    size carries an identifier so a test can see which features survived."""
    return (torch.zeros(1, 3, 2, 2), size_tag, "image")


def test_collator_neutralizes_truncated_media_instead_of_raising():
    # Two image sentinels; model_max_length=4 keeps only the first. The sample's
    # second image feature is an orphan that would mis-splice -> the guard trims
    # the feature list to the surviving sentinel and does NOT raise.
    collator = DataCollatorForSupervisedDataset(
        tokenizer=_Tok(4),
        media_token_ids=[IMG_ID, AUD_ID],
        media_feature_token_ids={"image": IMG_ID, "audio": AUD_ID},
    )
    inst = _instance([1, IMG_ID, 2, 3, 4, IMG_ID, 5])
    inst["image"] = [_img_entry(10), _img_entry(11)]
    batch = collator([inst])  # no raise
    assert batch["input_ids"].shape[1] == 4
    assert batch["image_sizes"] == [10]  # orphaned 2nd image feature pruned


def test_collator_keeps_one_dummy_when_all_sentinels_truncated():
    # The single image sentinel sits past the cutoff -> 0 survive. The sample's
    # one image feature is kept (consumed zero-width by the splice), not dropped.
    collator = DataCollatorForSupervisedDataset(
        tokenizer=_Tok(4),
        media_token_ids=[IMG_ID],
        media_feature_token_ids={"image": IMG_ID},
    )
    inst = _instance([1, 2, 3, 4, 5, IMG_ID, 6])
    inst["image"] = [_img_entry(10)]
    batch = collator([inst])  # no raise
    assert batch["input_ids"].shape[1] == 4
    assert batch["image_sizes"] == [10]  # the lone zero-width dummy survives


def test_collator_realign_preserves_cross_sample_alignment():
    # Sample A has two image sentinels, the second truncated away, so it must
    # contribute exactly one image feature; sample B is untouched. Without the
    # realignment A's orphaned 2nd feature would be spliced into B's sentinel.
    collator = DataCollatorForSupervisedDataset(
        tokenizer=_Tok(4),
        media_token_ids=[IMG_ID],
        media_feature_token_ids={"image": IMG_ID},
    )
    a = _instance([IMG_ID, 1, 2, 3, IMG_ID], idx="a")
    a["image"] = [_img_entry(10), _img_entry(11)]
    b = _instance([IMG_ID, 9], idx="b")
    b["image"] = [_img_entry(20)]
    batch = collator([a, b])
    # A trimmed to its one surviving sentinel (size 10); B intact (size 20).
    assert batch["image_sizes"] == [10, 20]


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
