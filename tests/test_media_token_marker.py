"""#11 — literal media-token text must not steal a real typed media position.

The energon typed-content bridge (messages_to_conversations) converts each typed
image/audio item into a placeholder at its exact position, and marks those
GENERATED placeholders with private-use sentinels. inject_missing_media_tokens
then treats the marked positions as authoritative: it neutralizes any UNMARKED
literal '<image>'/'<audio>' quoted in user text and unwraps the marks to a plain
token. Before the fix, the "neutralize surplus from the end" rule could
neutralize the REAL trailing placeholder and leave the quoted literal as the
model sentinel.

The non-marked (local-json) path is unchanged — covered here too as a guard.
"""

import pytest

try:
    from vlm.data.data_arguments import DataArguments
    from vlm.data.dataset import (
        _MARK_TMP,
        MEDIA_PLACEHOLDER_MARK_L,
        MEDIA_PLACEHOLDER_MARK_R,
        inject_missing_media_tokens,
    )
    from vlm.data.energon_dataset import messages_to_conversations
except ModuleNotFoundError as e:  # pragma: no cover - slim envs
    pytest.skip(f"vlm package not importable here: {e}", allow_module_level=True)

IMG = "<image>"
AUD = "<audio>"
_MARK_CHARS = (MEDIA_PLACEHOLDER_MARK_L, MEDIA_PLACEHOLDER_MARK_R, _MARK_TMP)


def _no_marks(text: str) -> bool:
    return not any(ch in text for ch in _MARK_CHARS)


def _bridge_and_inject(messages, n_images, n_audios, **da_kw):
    """Run the FULL energon typed-content path: messages -> marked conversations
    -> inject (unwrap marks / neutralize quoted literals)."""
    data_args = DataArguments(**da_kw)
    convs = messages_to_conversations(messages, data_args)
    inject_missing_media_tokens(convs, n_images=n_images, n_audios=n_audios, data_args=data_args)
    return convs


# ---------------------------------------------------------------------------
# #11 core: a quoted literal before a real typed image must NOT win.
# ---------------------------------------------------------------------------


def test_quoted_image_literal_does_not_steal_real_typed_position():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "the token <image> appears in my text"},
                {"type": "image", "path": "imgs/a.jpg"},
            ],
        },
        {"role": "assistant", "content": "a red square"},
    ]
    convs = _bridge_and_inject(messages, n_images=1, n_audios=0)
    user = convs[0]["value"]
    # Exactly one real sentinel survives, at the REAL (structural, trailing)
    # position; the quoted literal is neutralized to "[image]".
    assert user.count(IMG) == 1
    assert user.rstrip().endswith(IMG)
    assert "[image]" in user
    assert "the token [image] appears in my text" in user
    assert _no_marks(user)


def test_quoted_audio_literal_does_not_steal_real_typed_position():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "say <audio> out loud"},
                {"type": "audio", "path": "wavs/x.flac"},
            ],
        },
        {"role": "assistant", "content": "ok"},
    ]
    convs = _bridge_and_inject(messages, n_images=0, n_audios=1, audio_enabled=True)
    user = convs[0]["value"]
    assert user.count(AUD) == 1
    assert user.rstrip().endswith(AUD)
    assert "[audio]" in user
    assert _no_marks(user)


def test_normal_typed_image_no_quote_is_preserved():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": "imgs/a.jpg"},
                {"type": "text", "text": "describe this"},
            ],
        },
        {"role": "assistant", "content": "a cat"},
    ]
    convs = _bridge_and_inject(messages, n_images=1, n_audios=0)
    user = convs[0]["value"]
    assert user.count(IMG) == 1
    assert user.startswith(IMG)  # marked at its structural (leading) position
    assert "[image]" not in user
    assert _no_marks(user)


def test_two_typed_images_with_one_quoted_literal():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": "imgs/a.jpg"},
                {"type": "text", "text": "compare with the <image> mentioned here"},
                {"type": "image", "path": "imgs/b.jpg"},
            ],
        },
        {"role": "assistant", "content": "they differ"},
    ]
    convs = _bridge_and_inject(messages, n_images=2, n_audios=0)
    user = convs[0]["value"]
    # Two real sentinels survive (the two typed items); the quoted literal is
    # neutralized.
    assert user.count(IMG) == 2
    assert "[image]" in user
    assert _no_marks(user)


# ---------------------------------------------------------------------------
# Marks never leak past inject (they would corrupt tokenization otherwise).
# ---------------------------------------------------------------------------


def test_marks_present_before_inject_absent_after():
    data_args = DataArguments()
    messages = [
        {"role": "user", "content": [{"type": "image", "path": "imgs/a.jpg"}]},
        {"role": "assistant", "content": "x"},
    ]
    convs = messages_to_conversations(messages, data_args)
    assert MEDIA_PLACEHOLDER_MARK_L in convs[0]["value"]  # marked by the bridge
    inject_missing_media_tokens(convs, n_images=1, n_audios=0, data_args=data_args)
    assert _no_marks(convs[0]["value"])  # unwrapped to a plain token


# ---------------------------------------------------------------------------
# Non-marked (local-json) path is unchanged — regression guard.
# ---------------------------------------------------------------------------


def test_legacy_matching_count_unchanged():
    data_args = DataArguments()
    convs = [{"from": "human", "value": f"{IMG}\nwhat is this?"}, {"from": "gpt", "value": "a"}]
    inject_missing_media_tokens(convs, n_images=1, n_audios=0, data_args=data_args)
    assert convs[0]["value"] == f"{IMG}\nwhat is this?"


def test_legacy_prepend_when_missing():
    data_args = DataArguments()
    convs = [{"from": "human", "value": "describe"}, {"from": "gpt", "value": "a"}]
    inject_missing_media_tokens(convs, n_images=1, n_audios=0, data_args=data_args)
    assert convs[0]["value"] == f"{IMG}\ndescribe"


def test_legacy_surplus_neutralized_from_end():
    # Local-json path (no marks): the historical "neutralize surplus from the
    # end" behavior is preserved when there is no structural mark to trust.
    data_args = DataArguments()
    convs = [{"from": "human", "value": f"{IMG} and {IMG}"}, {"from": "gpt", "value": "a"}]
    inject_missing_media_tokens(convs, n_images=1, n_audios=0, data_args=data_args)
    assert convs[0]["value"] == f"{IMG} and [image]"


def test_legacy_too_few_raises():
    data_args = DataArguments()
    convs = [{"from": "human", "value": f"{IMG}"}, {"from": "gpt", "value": "a"}]
    with pytest.raises(ValueError):
        inject_missing_media_tokens(convs, n_images=2, n_audios=0, data_args=data_args)
