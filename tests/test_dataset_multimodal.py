"""Tests for the multimodal (image+audio) dataset pipeline.

Covers the shared per-sample core (placeholder injection, audio decoding,
multimodal tokenization), the plain/qwen template generalizations, and the
collator's two entry layouts:
  - encoder-free 4-tuples (patches, position_ids, size, modality)
  - classic CLIP 3-tuples (pixel_values, size, modality)  [regression]
"""

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")
PIL_Image = pytest.importorskip("PIL.Image")

try:
    from vlm.data.data_arguments import DataArguments
    from vlm.data.dataset import (
        DataCollatorForSupervisedDataset,
        LazySupervisedDataset,
        _load_local_records,  # pyright: ignore[reportPrivateUsage]
        inject_missing_media_tokens,
        load_audio_frames,
        make_supervised_data_module,
        tokenizer_image_token,
        tokenizer_multimodal_token,
    )
    from vlm.models.image_processing_raw import RawImageProcessor
    from vlm.utils import conversation as conversation_lib
except ModuleNotFoundError as e:  # pragma: no cover - slim envs
    pytest.skip(f"vlm package not importable here: {e}", allow_module_level=True)

try:
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
except Exception as e:  # pragma: no cover - no HF cache
    pytest.skip(f"Qwen3 tokenizer unavailable: {e}", allow_module_level=True)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def media_dir(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("media")
    (root / "imgs").mkdir()
    (root / "wavs").mkdir()
    PIL_Image.new("RGB", (200, 150), (255, 0, 0)).save(root / "imgs" / "a.jpg")
    PIL_Image.new("RGB", (640, 480), (0, 255, 0)).save(root / "imgs" / "b.jpg")
    rng = np.random.default_rng(0)
    # 2s stereo @ 22.05k (resample+downmix), 1s mono @ 16k, 40s (truncation)
    sf.write(root / "wavs" / "x.flac", rng.normal(size=(44100, 2)).astype(np.float32) * 0.1, 22050)
    sf.write(root / "wavs" / "y.wav", rng.normal(size=16000).astype(np.float32) * 0.1, 16000)
    sf.write(
        root / "wavs" / "long.wav", rng.normal(size=16000 * 40).astype(np.float32) * 0.1, 16000
    )
    return root


SAMPLES = [
    {
        "id": "img-only-no-ph",  # no placeholder -> injection
        "image": "a.jpg",
        "conversations": [
            {"from": "human", "value": "describe the picture"},
            {"from": "gpt", "value": "a red square"},
        ],
    },
    {
        "id": "audio-only-asr",  # ASR-style: no placeholder -> injection
        "audio": "y.wav",
        "conversations": [
            {"from": "human", "value": "transcribe"},
            {"from": "gpt", "value": "hello world"},
        ],
    },
    {
        "id": "both-with-ph",  # explicit placeholders are respected
        "image": "b.jpg",
        "audio": "x.flac",
        "conversations": [
            {"from": "human", "value": "<image>\n<audio>\nwhat do you see and hear?"},
            {"from": "gpt", "value": "green and noise"},
        ],
    },
    {
        "id": "long-audio",  # > max_audio_tokens -> truncated, head kept
        "audio": "long.wav",
        "conversations": [
            {"from": "human", "value": "<audio>\ntranscribe"},
            {"from": "gpt", "value": "..."},
        ],
    },
    {
        "id": "text-only",  # dummies for both modalities (qwen template only)
        "conversations": [
            {"from": "human", "value": "hi"},
            {"from": "gpt", "value": "hey"},
        ],
    },
]


def _make_dataset(media_dir: Path, samples: Any, audio_enabled: bool = True):
    data_path = media_dir / "data.json"
    data_path.write_text(json.dumps(samples))
    processor = RawImageProcessor(patch_size=16, pooling_kernel_size=3, max_soft_tokens=280)
    data_args = DataArguments(
        data_path=str(data_path),
        image_folder=str(media_dir / "imgs"),
        audio_folder=str(media_dir / "wavs"),
        audio_enabled=audio_enabled,
    )

    class _Proc:  # duck-typed: the dataset only reads these two attributes
        tokenizer: Any = _TOKENIZER
        image_processor: Any = processor

    return LazySupervisedDataset(str(data_path), _Proc(), data_args), data_args


# ---------------------------------------------------------------------------
# shared core units
# ---------------------------------------------------------------------------


def test_multimodal_tokenizer_matches_image_tokenizer_on_image_only():
    data_args = DataArguments()
    for prompt in ["<image>\nhello", "a <image> b <image> c", "no media at all"]:
        legacy = tokenizer_image_token(prompt, _TOKENIZER, data_args)
        new = tokenizer_multimodal_token(prompt, _TOKENIZER, data_args)
        assert new == legacy, prompt


def test_multimodal_tokenizer_audio_sentinel():
    data_args = DataArguments()
    ids = tokenizer_multimodal_token("<image>\n<audio>\nhi", _TOKENIZER, data_args)
    assert ids.count(-200) == 1 and ids.count(-201) == 1
    assert ids.index(-200) < ids.index(-201)


def test_injection_rules():
    data_args = DataArguments()
    convs = [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo"}]
    inject_missing_media_tokens(convs, n_images=2, n_audios=1, data_args=data_args)
    assert convs[0]["value"] == "<image>\n<image>\n<audio>\nhi"
    # matching counts: untouched
    before = [dict(t) for t in convs]
    inject_missing_media_tokens(convs, n_images=2, n_audios=1, data_args=data_args)
    assert convs == before
    # surplus literals (found > n): neutralized from the END, NOT raised — a
    # raise in energon's buffer-restore path would be a deterministic resume
    # crash-loop (the last <image> is kept aligned with the one real image).
    inject_missing_media_tokens(convs, n_images=1, n_audios=1, data_args=data_args)
    assert convs[0]["value"] == "<image>\n[image]\n<audio>\nhi"
    # too few placeholders (found < n): loud failure (ambiguous; surface it).
    with pytest.raises(ValueError, match="placeholder"):
        inject_missing_media_tokens(convs, n_images=3, n_audios=1, data_args=data_args)


def test_load_audio_frames(media_dir: Path):
    data_args = DataArguments(audio_sampling_rate=16000, audio_samples_per_token=640)
    # stereo 22.05k 2s -> mono 16k -> 50 frames
    frames = load_audio_frames(str(media_dir / "wavs" / "x.flac"), data_args)
    assert frames.shape == (50, 640) and frames.dtype == torch.float32
    # 40s -> capped at 750 frames (head kept)
    frames = load_audio_frames(str(media_dir / "wavs" / "long.wav"), data_args)
    assert frames.shape == (750, 640)
    # partial tail frame is zero-padded
    data_args_nocap = DataArguments(max_audio_tokens=None)
    sf_path = media_dir / "wavs" / "odd.wav"
    sf.write(sf_path, np.ones(1000, dtype=np.float32), 16000)
    frames = load_audio_frames(str(sf_path), data_args_nocap)
    assert frames.shape == (2, 640)
    assert torch.all(frames[1, 1000 - 640 :] == 0)


# ---------------------------------------------------------------------------
# plain template (stage-1 pretrain)
# ---------------------------------------------------------------------------


def test_plain_dataset_and_collator(media_dir: Path):
    conversation_lib.default_conversation = conversation_lib.conv_templates["plain"]
    samples = [s for s in SAMPLES if s["id"] != "text-only"]  # plain requires media
    ds, _ = _make_dataset(media_dir, samples)
    items = [ds[i] for i in range(len(ds))]

    by_id = {d["id"]: d for d in items}
    # injection produced exactly one sentinel each
    assert int((by_id["img-only-no-ph"]["input_ids"] == -200).sum()) == 1
    assert int((by_id["audio-only-asr"]["input_ids"] == -201).sum()) == 1
    # explicit placeholders respected, in order
    ids = by_id["both-with-ph"]["input_ids"]
    assert int((ids == -200).sum()) == 1 and int((ids == -201).sum()) == 1
    # plain masks exactly the media prefix
    labels = by_id["both-with-ph"]["labels"]
    assert int((labels == -100).sum()) == 2
    # variable resolution: 200x150 -> 266 patches @ budget 280
    patches, positions, _size, modality = by_id["img-only-no-ph"]["image"][0]
    assert patches.shape == (266, 6912) and positions.shape == (266, 2) and modality == "image"
    # dummies for absent modalities
    assert by_id["audio-only-asr"]["image"][0][3] == "text"
    assert by_id["audio-only-asr"]["image"][0][0].shape == (1, 6912)
    assert by_id["img-only-no-ph"]["audio"][0].shape == (1, 640)
    assert by_id["long-audio"]["audio"][0].shape == (750, 640)

    batch = DataCollatorForSupervisedDataset(tokenizer=_TOKENIZER)(items)
    assert sorted(batch.keys()) == [
        "attention_mask",
        "audios",
        "image_position_ids",
        "image_sizes",
        "images",
        "input_ids",
        "labels",
    ]
    # queue layout: one entry per sample per modality (real or dummy)
    assert len(batch["images"]) == len(items)
    assert len(batch["image_position_ids"]) == len(items)
    assert len(batch["audios"]) == len(items)


def test_plain_text_only_rejected(media_dir: Path):
    # legacy semantics preserved: plain pretrain requires media in every sample
    conversation_lib.default_conversation = conversation_lib.conv_templates["plain"]
    ds, _ = _make_dataset(media_dir, [s for s in SAMPLES if s["id"] == "text-only"])
    with pytest.raises(AssertionError, match="media placeholder"):
        ds._get_item(0)  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# qwen template (SFT)
# ---------------------------------------------------------------------------


def test_qwen_dataset_and_collator(media_dir: Path):
    conversation_lib.default_conversation = conversation_lib.conv_templates["qwen_2_5"]
    ds, _ = _make_dataset(media_dir, SAMPLES)
    items = [ds[i] for i in range(len(ds))]
    by_id = {d["id"]: d for d in items}

    assert int((by_id["both-with-ph"]["input_ids"] == -200).sum()) == 1
    assert int((by_id["both-with-ph"]["input_ids"] == -201).sum()) == 1
    # text-only works under qwen and still ships both dummies
    ids = by_id["text-only"]["input_ids"]
    assert int((ids == -200).sum()) == 0 and int((ids == -201).sum()) == 0
    assert by_id["text-only"]["image"][0][0].shape == (1, 6912)
    assert by_id["text-only"]["audio"][0].shape == (1, 640)
    # assistant turns are supervised
    assert int((by_id["text-only"]["labels"] != -100).sum()) > 0

    batch = DataCollatorForSupervisedDataset(tokenizer=_TOKENIZER)(items)
    assert len(batch["audios"]) == len(items)


def test_v1_template_with_audio_rejected(media_dir: Path):
    conversation_lib.default_conversation = conversation_lib.conv_templates["v1"]
    ds, _ = _make_dataset(media_dir, [s for s in SAMPLES if s["id"] == "audio-only-asr"])
    with pytest.raises(NotImplementedError, match="audio"):
        ds._get_item(0)  # pyright: ignore[reportPrivateUsage]


def test_audio_disabled_rejected(media_dir: Path):
    conversation_lib.default_conversation = conversation_lib.conv_templates["plain"]
    ds, _ = _make_dataset(
        media_dir, [s for s in SAMPLES if s["id"] == "audio-only-asr"], audio_enabled=False
    )
    with pytest.raises(ValueError, match="audio.enabled"):
        ds._get_item(0)  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# legacy CLIP path regression
# ---------------------------------------------------------------------------


def test_legacy_clip_path_unchanged(media_dir: Path):
    conversation_lib.default_conversation = conversation_lib.conv_templates["plain"]

    class MockClipProcessor:
        crop_size: dict[str, int] = {"height": 224, "width": 224}
        image_mean: list[float] = [0.5, 0.5, 0.5]

        def preprocess(self, image: Any, return_tensors: Any = None) -> dict:
            return {"pixel_values": [torch.zeros(3, 224, 224)]}

    data_path = media_dir / "legacy.json"
    data_path.write_text(json.dumps([s for s in SAMPLES if s["id"] == "img-only-no-ph"]))
    data_args = DataArguments(data_path=str(data_path), image_folder=str(media_dir / "imgs"))

    class _Proc:
        tokenizer: Any = _TOKENIZER
        image_processor: Any = MockClipProcessor()

    ds = LazySupervisedDataset(str(data_path), _Proc(), data_args)
    items = [ds[i] for i in range(len(ds))]
    # classic 3-tuple entries, no audio key at all
    assert len(items[0]["image"][0]) == 3
    assert "audio" not in items[0]

    batch = DataCollatorForSupervisedDataset(tokenizer=_TOKENIZER)(items)
    assert sorted(batch.keys()) == [
        "attention_mask",
        "image_sizes",
        "images",
        "input_ids",
        "labels",
    ]
    assert batch["images"][0].shape == (3, 224, 224)
    assert batch["image_sizes"] == [(200, 150)]


# ---------------------------------------------------------------------------
# #3 multi-image local JSON: one <image> sentinel per loaded image
# ---------------------------------------------------------------------------


def test_multi_image_local_json_preserves_placeholder_count(media_dir: Path):
    """A multi-image local-JSON sample must emit exactly one <image> sentinel
    per loaded image. Collapsing N->1 (the old preprocess_multimodal) makes the
    splice consume one feature here and leak the rest into the next sample."""
    conversation_lib.default_conversation = conversation_lib.conv_templates["qwen_2_5"]
    samples = [
        {
            "id": "multi-image",
            "image": ["a.jpg", "b.jpg"],
            "conversations": [
                {"from": "human", "value": "<image>\n<image>\nCompare the two images."},
                {"from": "gpt", "value": "ok"},
            ],
        }
    ]
    ds, _ = _make_dataset(media_dir, samples)
    item = ds[0]
    n_sentinels = int((item["input_ids"] == -200).sum())
    n_features = len(item["image"])
    assert n_features == 2
    assert n_sentinels == 2  # was 1 before the fix (collapsed)
    assert n_sentinels == n_features  # splice invariant

    # the collator queues both image features and the batch-wide counts agree
    batch = DataCollatorForSupervisedDataset(tokenizer=_TOKENIZER)([item])
    assert len(batch["images"]) == 2
    assert int((batch["input_ids"] == -200).sum()) == len(batch["images"])


def test_multi_image_injected_when_no_placeholder(media_dir: Path):
    """When the text carries no placeholder, injection prepends N <image>
    tokens; preprocess_multimodal must keep all N (not collapse to one)."""
    conversation_lib.default_conversation = conversation_lib.conv_templates["qwen_2_5"]
    samples = [
        {
            "id": "multi-image-noph",
            "image": ["a.jpg", "b.jpg"],
            "conversations": [
                {"from": "human", "value": "describe both"},
                {"from": "gpt", "value": "ok"},
            ],
        }
    ]
    ds, _ = _make_dataset(media_dir, samples)
    item = ds[0]
    assert int((item["input_ids"] == -200).sum()) == len(item["image"]) == 2


# ---------------------------------------------------------------------------
# #12 collator truncation must not silently drop a media sentinel
# ---------------------------------------------------------------------------


class _TinyTok:
    """Minimal tokenizer surface the collator reads — lets us force a tiny
    model_max_length without mutating the shared module tokenizer."""

    model_max_length: int = 8
    padding_side: str = "right"
    pad_token_id: int = 0


def test_collator_neutralizes_when_truncation_drops_media_sentinel():
    # Two image sentinels with two queued image features; the second sentinel
    # sits past model_max_length=8. Right-truncation drops it, so the collator
    # realigns the feature list to the one surviving sentinel and WARNs, rather
    # than raising — a raise crash-loops energon's buffer-restore on resume.
    collator = DataCollatorForSupervisedDataset(
        tokenizer=_TinyTok(),
        media_token_ids=(-200, -201),
        media_feature_token_ids={"image": -200},
    )
    ids = torch.tensor([1, -200, 3, 4, 5, 6, 7, 8, -200, 11])  # idx 1 kept, idx 8 dropped
    instance = {
        "input_ids": ids,
        "labels": torch.full_like(ids, -100),
        "image": ["feat-a", "feat-b"],
        "id": "trunc-victim",
    }
    collator._neutralize_truncated_media([instance])  # must not raise
    assert instance["image"] == ["feat-a"]  # realigned to the surviving sentinel


def test_collator_ok_when_sentinel_survives_truncation():
    collator = DataCollatorForSupervisedDataset(tokenizer=_TinyTok(), media_token_ids=(-200, -201))
    ids = torch.tensor([1, 2, -200, 4, 5, 6, 7, 8, 9, 10, 11])  # sentinel at idx 2 < 8
    instance = {"input_ids": ids, "labels": torch.full_like(ids, -100), "id": "ok"}
    batch = collator([instance])
    assert batch["input_ids"].shape[1] == 8
    assert int((batch["input_ids"] == -200).sum()) == 1


def test_collator_truncation_check_is_noop_without_media_token_ids():
    """Guard-off default: a collator built without media_token_ids leaves
    truncation a plain slice (no realign, no raise)."""
    collator = DataCollatorForSupervisedDataset(tokenizer=_TinyTok())  # media_token_ids=()
    ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, -200, 11])
    instance = {"input_ids": ids, "labels": torch.full_like(ids, -100), "id": "x"}
    batch = collator([instance])  # must not raise
    assert batch["input_ids"].shape[1] == 8


def test_make_supervised_data_module_wires_media_token_ids(media_dir: Path):
    conversation_lib.default_conversation = conversation_lib.conv_templates["qwen_2_5"]
    data_path = media_dir / "wire.json"
    data_path.write_text(json.dumps([s for s in SAMPLES if s["id"] == "img-only-no-ph"]))
    processor = RawImageProcessor(patch_size=16, pooling_kernel_size=3, max_soft_tokens=280)
    data_args = DataArguments(
        data_path=str(data_path),
        image_folder=str(media_dir / "imgs"),
        audio_folder=str(media_dir / "wavs"),
        audio_enabled=True,
    )

    class _Proc:
        tokenizer: Any = _TOKENIZER
        image_processor: Any = processor

    module = make_supervised_data_module(processor=_Proc(), data_args=data_args)
    collator = module["data_collator"]
    # audio_enabled=True, learnable_query off: image + audio sentinels, plus the
    # feature-list map the neutralize guard uses to realign truncated samples.
    assert collator.media_token_ids == [-200, -201]
    assert collator.media_feature_token_ids == {"image": -200, "audio": -201}


# ---------------------------------------------------------------------------
# #26 direct local .jsonl path (documented but previously unsupported)
# ---------------------------------------------------------------------------


def test_load_local_records_json_jsonl_and_rejection(tmp_path: Path):
    j = tmp_path / "a.json"
    j.write_text(json.dumps([{"x": 1}, {"x": 2}]))
    assert _load_local_records(str(j)) == [{"x": 1}, {"x": 2}]

    jl = tmp_path / "a.jsonl"
    jl.write_text('{"x": 1}\n\n{"x": 2}\n')  # blank line tolerated
    assert _load_local_records(str(jl)) == [{"x": 1}, {"x": 2}]

    bad = tmp_path / "a.txt"
    bad.write_text("nope")
    with pytest.raises(ValueError, match="unsupported local dataset file"):
        _load_local_records(str(bad))


def test_direct_jsonl_dataset_path_loads(media_dir: Path):
    """A direct dataset.path=*.jsonl now parses line-by-line; previously it fell
    through to json.load() and crashed on the second object."""
    conversation_lib.default_conversation = conversation_lib.conv_templates["qwen_2_5"]
    samples = [s for s in SAMPLES if s["id"] in ("img-only-no-ph", "text-only")]
    data_path = media_dir / "data.jsonl"
    data_path.write_text("\n".join(json.dumps(s) for s in samples) + "\n")
    processor = RawImageProcessor(patch_size=16, pooling_kernel_size=3, max_soft_tokens=280)
    data_args = DataArguments(
        data_path=str(data_path),
        image_folder=str(media_dir / "imgs"),
        audio_folder=str(media_dir / "wavs"),
        audio_enabled=True,
    )

    class _Proc:
        tokenizer: Any = _TOKENIZER
        image_processor: Any = processor

    ds = LazySupervisedDataset(str(data_path), _Proc(), data_args)
    assert len(ds) == 2  # both lines parsed (json.load would have raised)
    # the loaded sample still tokenizes end-to-end
    assert int((ds[0]["input_ids"] == -200).sum()) == 1


def test_direct_unsupported_extension_dataset_path_rejected(media_dir: Path):
    data_path = media_dir / "data.txt"
    data_path.write_text("not json")
    data_args = DataArguments(data_path=str(data_path), image_folder=str(media_dir / "imgs"))

    class _Proc:
        tokenizer: Any = _TOKENIZER
        image_processor: Any = RawImageProcessor(
            patch_size=16, pooling_kernel_size=3, max_soft_tokens=280
        )

    with pytest.raises(ValueError, match="unsupported local dataset file"):
        LazySupervisedDataset(str(data_path), _Proc(), data_args)
