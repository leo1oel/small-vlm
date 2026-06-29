"""Tests for the energon pipeline's corrupt-sample resilience.

Honey-Data-1M (and likely bee_stage2) carry occasional corrupt images: a
truncated/broken PNG makes PIL raise ``SyntaxError``, which megatron-energon
classifies as a fatal ``SYSTEM_EXCEPTION`` and re-raises as ``FatalSampleError``,
killing the DataLoader worker -> the rank -> the whole multi-day run (observed
live: encabl-native-s2 died at step ~5000/11905 on shard 00059.tar sample
0000599478). One bad image anywhere in the data destroyed an unattended run.

The fix wraps ``TaskEncoder.encode_sample`` with ``@skip_corrupt_samples``,
turning any per-sample failure into energon's ``SkipSample`` (log once + count +
drop the one sample + keep training). These tests pin:
  - the crux: a ``SyntaxError`` is fatal WITHOUT the wrapper but skipped WITH it,
    driven through energon's REAL ``handle_errors`` (not a mock);
  - that an isolated skip is logged once with key/shard/exception and bumps the
    drop counter, while a SYSTEMATIC failure (>= ``VLM_MAX_CONSECUTIVE_SKIPS``
    consecutive skips with no success in between) escalates to a fatal
    ``FatalSampleError``, and a single successful encode resets that counter;
  - the real chat/caption + generation task encoders over a good image (happy
    path unchanged) and an undecodable image (skipped).
"""

import io
import logging
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
PIL_Image = pytest.importorskip("PIL.Image")

try:
    from megatron.energon import SkipSample
    from megatron.energon.errors import ErrorContext, FatalSampleError

    from vlm.data import energon_dataset as ed
    from vlm.data.data_arguments import DataArguments
    from vlm.data.energon_dataset import (
        MMChatRawSample,
        VLMChatTaskEncoder,
        VLMGenTaskEncoder,
        skip_corrupt_samples,
    )
    from vlm.data.energon_wds import WDSChatTaskEncoder, WDSGenTaskEncoder
    from vlm.models.image_processing_raw import RawImageProcessor
    from vlm.utils import conversation as conversation_lib
except ModuleNotFoundError as e:  # pragma: no cover - slim envs
    pytest.skip(f"energon/vlm not importable here: {e}", allow_module_level=True)


def _load_tokenizer():
    # Single assignment to _TOKENIZER (a helper, not try/except reassignment)
    # so the tokenizer-free crux tests still run when the HF cache is absent.
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except Exception:  # pragma: no cover - no HF cache
        return None


_TOKENIZER = _load_tokenizer()

needs_tokenizer = pytest.mark.skipif(_TOKENIZER is None, reason="Qwen3 tokenizer unavailable")


@pytest.fixture(autouse=True)
def _reset_consecutive_skips():
    # The per-worker consecutive-skip counter is module-global; reset it around
    # every test so the fatal threshold can't be tripped by accumulation across
    # unrelated tests (and so a small monkeypatched threshold starts clean).
    ed._consecutive_skipped = 0  # pyright: ignore[reportPrivateUsage]
    yield
    ed._consecutive_skipped = 0  # pyright: ignore[reportPrivateUsage]


# PIL's exact broken-PNG message from the reported crash; SyntaxError is in
# energon's SYSTEM_EXCEPTIONS, so a mere error *handler* can't rescue it — only
# converting to SkipSample at the source can.
BROKEN_PNG_MSG = r"broken PNG file (chunk b'WU\x95\xe3')"


def _bare_sample(key: str = "s0") -> MMChatRawSample:
    return MMChatRawSample(
        __key__=key,
        __restore_key__=(),
        __subflavors__=None,
        messages=[],
        image_bytes=[],
        audio_bytes=[],
        source=None,
    )


def _raw_chat_encoder():
    """A real VLMChatTaskEncoder wired exactly as the loader wires it."""
    conversation_lib.default_conversation = conversation_lib.conv_templates["qwen_2_5"]
    processor = SimpleNamespace(
        tokenizer=_TOKENIZER,
        image_processor=RawImageProcessor(
            patch_size=16, pooling_kernel_size=3, max_soft_tokens=280
        ),
    )
    return VLMChatTaskEncoder(processor, DataArguments(audio_enabled=False))


def _good_png_bytes() -> bytes:
    buf = io.BytesIO()
    PIL_Image.new("RGB", (64, 48), (255, 0, 0)).save(buf, "PNG")
    return buf.getvalue()


def _chat_sample(key: str, image_bytes: list[bytes]) -> MMChatRawSample:
    return MMChatRawSample(
        __key__=key,
        __restore_key__=(),
        __subflavors__=None,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": "x.png"},
                    {"type": "text", "text": "describe"},
                ],
            },
            {"role": "assistant", "content": "a red rectangle"},
        ],
        image_bytes=image_bytes,
        audio_bytes=[],
        source="test",
    )


# ---------------------------------------------------------------------------
# crux: fatal without the wrapper, non-fatal with it (real handle_errors)
# ---------------------------------------------------------------------------


def test_syntaxerror_is_fatal_without_wrapper():
    """Reproduce the bug: PIL's broken-PNG SyntaxError is a SYSTEM_EXCEPTION, so
    energon's handle_errors re-raises it as FatalSampleError (kills the worker)."""

    def raw(_self, _sample):
        raise SyntaxError(BROKEN_PNG_MSG)

    ctx = ErrorContext("MapDataset.raw", lambda *_a: None, tolerance=100)
    with pytest.raises(FatalSampleError):  # noqa: PT012 - exercise the context manager
        with ctx.handle_errors(_bare_sample()):
            raw(None, _bare_sample())


def test_wrapper_makes_syntaxerror_nonfatal():
    """With the wrapper the same SyntaxError becomes SkipSample, which
    handle_errors swallows: nothing escapes, the run continues."""
    handler_calls: list = []

    @skip_corrupt_samples
    def encode(_self, _sample):
        raise SyntaxError(BROKEN_PNG_MSG)

    ctx = ErrorContext("MapDataset.wrapped", lambda e, *_a: handler_calls.append(e), tolerance=100)
    with ctx.handle_errors(_bare_sample()):  # no exception escapes -> skipped
        encode(None, _bare_sample())

    # SkipSample neither invokes the error handler nor increments the
    # consecutive-failure counter, so corrupt samples can never fatalize.
    assert handler_calls == []
    assert ctx._consecutive_failures == 0  # pyright: ignore[reportPrivateUsage]


def test_isolated_skips_below_threshold_never_fatalize(monkeypatch):
    """Below the consecutive-skip threshold, every per-sample failure is an
    ordinary SkipSample, never a fatal error (energon's own tolerance, which
    SkipSample bypasses entirely, is covered by the crux test above)."""
    monkeypatch.setattr(ed, "_MAX_CONSECUTIVE_SKIPS", 5)

    @skip_corrupt_samples
    def encode(_self, _sample):
        raise ValueError("bad json")

    for _ in range(4):  # 1..4 consecutive — below the threshold of 5
        with pytest.raises(SkipSample):
            encode(None, _bare_sample())
    assert ed._consecutive_skipped == 4  # pyright: ignore[reportPrivateUsage]


def test_consecutive_skips_reaching_threshold_fatalize(monkeypatch):
    """A SYSTEMATIC failure (every sample fails) escalates to FatalSampleError
    once the consecutive-skip counter reaches VLM_MAX_CONSECUTIVE_SKIPS, so the
    run dies loud+fast instead of silently dropping every sample forever."""
    monkeypatch.setattr(ed, "_MAX_CONSECUTIVE_SKIPS", 5)

    @skip_corrupt_samples
    def encode(_self, _sample):
        raise ValueError("bad json")

    for _ in range(4):  # the first four are skipped
        with pytest.raises(SkipSample):
            encode(None, _bare_sample())
    # the 5th consecutive failure trips the threshold and fatalizes
    with pytest.raises(FatalSampleError):
        encode(None, _bare_sample())
    # FatalSampleError is a SYSTEM_EXCEPTION, so energon's handle_errors re-raises
    # it rather than swallowing it (unlike SkipSample) — the run actually dies.
    ctx = ErrorContext("MapDataset.wrapped", lambda *_a: None, tolerance=100)
    with pytest.raises(FatalSampleError):
        with ctx.handle_errors(_bare_sample()):
            encode(None, _bare_sample())


def test_successful_encode_resets_consecutive_skip_counter(monkeypatch):
    """A single successful encode between failures resets the consecutive
    counter, so interspersed/isolated corruption can never reach the threshold."""
    monkeypatch.setattr(ed, "_MAX_CONSECUTIVE_SKIPS", 3)
    state = {"fail": True}

    @skip_corrupt_samples
    def encode(_self, _sample):
        if state["fail"]:
            raise ValueError("bad json")
        return {"ok": True}

    for _ in range(2):  # two consecutive failures (below 3)
        with pytest.raises(SkipSample):
            encode(None, _bare_sample())
    assert ed._consecutive_skipped == 2  # pyright: ignore[reportPrivateUsage]

    state["fail"] = False  # one success resets the counter
    assert encode(None, _bare_sample()) == {"ok": True}
    assert ed._consecutive_skipped == 0  # pyright: ignore[reportPrivateUsage]

    state["fail"] = True  # two more failures — still below the threshold
    for _ in range(2):
        with pytest.raises(SkipSample):
            encode(None, _bare_sample())
    assert ed._consecutive_skipped == 2  # pyright: ignore[reportPrivateUsage]


def test_threshold_zero_disables_fatalize(monkeypatch):
    """The energon convention: a threshold of 0 means 'tolerate forever' — even a
    long systematic run of failures only ever produces SkipSample, never fatal."""
    monkeypatch.setattr(ed, "_MAX_CONSECUTIVE_SKIPS", 0)

    @skip_corrupt_samples
    def encode(_self, _sample):
        raise ValueError("bad json")

    for _ in range(50):
        with pytest.raises(SkipSample):
            encode(None, _bare_sample())
    assert ed._consecutive_skipped == 50  # pyright: ignore[reportPrivateUsage]


def test_deliberate_skipsample_passes_through_uncounted():
    """A SkipSample raised inside encode_sample is a deliberate skip — it must
    pass through unchanged and touch neither the drop counter nor the
    consecutive-skip counter (no double-wrap, no re-count, no reset)."""
    before = ed._skipped_samples  # pyright: ignore[reportPrivateUsage]
    ed._consecutive_skipped = 7  # pyright: ignore[reportPrivateUsage]

    @skip_corrupt_samples
    def encode(_self, _sample):
        raise SkipSample("intentional")

    with pytest.raises(SkipSample):
        encode(None, _bare_sample())
    assert ed._skipped_samples == before  # pyright: ignore[reportPrivateUsage]
    assert ed._consecutive_skipped == 7  # pyright: ignore[reportPrivateUsage]


def test_skip_logs_once_with_key_and_counts(caplog):
    @skip_corrupt_samples
    def encode(_self, _sample):
        raise SyntaxError(BROKEN_PNG_MSG)

    before = ed._skipped_samples  # pyright: ignore[reportPrivateUsage]
    with caplog.at_level(logging.WARNING, logger="vlm.data.energon_dataset"):
        with pytest.raises(SkipSample):
            encode(None, _bare_sample(key="kXYZ"))

    assert ed._skipped_samples == before + 1  # pyright: ignore[reportPrivateUsage]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "kXYZ" in msg and "skipping" in msg and "SyntaxError" in msg


# ---------------------------------------------------------------------------
# real task encoders: happy path unchanged, corrupt image skipped
# ---------------------------------------------------------------------------


@needs_tokenizer
def test_real_chat_encoder_keeps_good_skips_corrupt():
    enc = _raw_chat_encoder()

    # Happy path is unchanged: a good image encodes to the trainer-ready dict.
    out = enc.encode_sample(_chat_sample("good", [_good_png_bytes()]))
    assert {"input_ids", "labels", "image", "id"}.issubset(out.keys())
    assert len(out["image"]) == 1
    assert out["id"] == "good"

    # A corrupt/undecodable image is skipped (SkipSample), never fatal.
    before = ed._skipped_samples  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(SkipSample):
        enc.encode_sample(_chat_sample("corrupt-key", [b"this is not a png"]))
    assert ed._skipped_samples == before + 1  # pyright: ignore[reportPrivateUsage]


@needs_tokenizer
def test_corrupt_sample_is_skipped_through_energon_handler():
    """End-to-end through energon's real handle_errors: the good sample encodes,
    the corrupt one is dropped without any exception escaping (run survives)."""
    enc = _raw_chat_encoder()
    ctx = ErrorContext("MapDataset.encode_sample", lambda *_a: None, tolerance=100)

    encoded: list = []
    with ctx.handle_errors(_chat_sample("good", [_good_png_bytes()])):
        encoded.append(enc.encode_sample(_chat_sample("good", [_good_png_bytes()])))
    # corrupt: must be swallowed by the handler, nothing escapes
    with ctx.handle_errors(_chat_sample("bad", [b"<<corrupt>>"])):
        encoded.append(enc.encode_sample(_chat_sample("bad", [b"<<corrupt>>"])))

    assert len(encoded) == 1  # only the good sample produced output
    assert encoded[0]["id"] == "good"


def test_wds_encoders_inherit_the_wrapped_encode_sample():
    """The prepared-WebDataset path (tar shards) is exactly where the reported
    crash happened (shard 00059.tar). Its encoders subclass the jsonl ones and
    override only `cookers`, so they MUST inherit the @skip_corrupt_samples
    encode_sample — pin that identity so a future override can't silently
    re-introduce the fatal crash on a corrupt in-tar image."""
    assert WDSChatTaskEncoder.encode_sample is VLMChatTaskEncoder.encode_sample
    assert WDSGenTaskEncoder.encode_sample is VLMGenTaskEncoder.encode_sample


@needs_tokenizer
def test_wds_chat_encoder_skips_corrupt_image():
    # encode_sample consumes an already-cooked MMChatRawSample, shared verbatim
    # with the jsonl path — drive the WDS class directly on a corrupt image.
    conversation_lib.default_conversation = conversation_lib.conv_templates["qwen_2_5"]
    processor = SimpleNamespace(
        tokenizer=_TOKENIZER,
        image_processor=RawImageProcessor(
            patch_size=16, pooling_kernel_size=3, max_soft_tokens=280
        ),
    )
    enc = WDSChatTaskEncoder(processor, DataArguments(audio_enabled=False))
    with pytest.raises(SkipSample):
        enc.encode_sample(_chat_sample("wds-corrupt", [b"not an image at all"]))


@needs_tokenizer
def test_generation_encoder_skips_corrupt_image():
    processor = SimpleNamespace(
        tokenizer=_TOKENIZER,
        image_processor=RawImageProcessor(
            patch_size=16, pooling_kernel_size=3, max_soft_tokens=280
        ),
    )
    enc = VLMGenTaskEncoder(processor, DataArguments(audio_enabled=False), resolution=96)
    sample = MMChatRawSample(
        __key__="g0",
        __restore_key__=(),
        __subflavors__=None,
        messages=[{"role": "assistant", "content": "a caption"}],
        image_bytes=[b"not a png"],
        audio_bytes=[],
        source=None,
    )
    with pytest.raises(SkipSample):
        enc.encode_sample(sample)
