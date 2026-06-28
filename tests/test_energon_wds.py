"""WDS (prepared CrudeWebdataset) cooker + loader-branch tests.

The headline feature reads image bytes from the IN-tar sample fields (not
media_root.get(path)). These tests exercise the cooker on a synthetic crude
sample (hermetic) and, when a tokenizer is cached, the full encode_sample path
(cook -> decode -> tokenize) so the WDS branch is shown to yield decoded
image+text samples.
"""

import io
import json

import pytest

torch = pytest.importorskip("torch")
PIL_Image = pytest.importorskip("PIL.Image")

try:
    from vlm.data.data_arguments import DataArguments
    from vlm.data.energon_dataset import MMChatRawSample
    from vlm.data.energon_wds import (
        WDSChatTaskEncoder,
        cook_mm_chat_wds,
        resolve_wds_path,
    )
    from vlm.models.image_processing_raw import RawImageProcessor
except ModuleNotFoundError as e:  # pragma: no cover - slim envs
    pytest.skip(f"vlm package not importable here: {e}", allow_module_level=True)


def _jpeg_bytes(size=(64, 48), color=(200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    PIL_Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def _crude_sample(rec: dict, fields: dict) -> dict:
    """A raw CrudeWebdataset sample dict as energon hands a cooker: the basic
    Sample keys + the in-tar member fields (json + image bytes)."""
    base = {
        "__key__": "shard0/sample0",
        "__restore_key__": (),
        "__subflavors__": {},
    }
    base["json"] = json.dumps(rec).encode()
    base.update(fields)
    return base


# ---------------------------------------------------------------------------
# Cooker (hermetic): in-tar bytes -> MMChatRawSample with decodable images.
# ---------------------------------------------------------------------------


def test_cooker_reads_in_tar_image_by_basename():
    img = _jpeg_bytes()
    rec = {
        "id": "x",
        "source": "bee_stage2",
        "messages": [
            {"role": "user", "content": [{"type": "image", "path": "imgs/abc_img0.jpg"}]},
            {"role": "assistant", "content": "a red rectangle"},
        ],
    }
    sample = _crude_sample(rec, {"abc_img0.jpg": img})
    out = cook_mm_chat_wds(sample)
    assert isinstance(out, MMChatRawSample)
    assert out.image_bytes == [img]
    assert out.source == "bee_stage2"
    assert out.messages == rec["messages"]
    # the bytes are a real, decodable image
    im = PIL_Image.open(io.BytesIO(out.image_bytes[0]))
    im.load()
    assert im.size == (64, 48)


def test_cooker_json_can_be_str_or_bytes():
    img = _jpeg_bytes()
    rec = {"messages": [{"role": "user", "content": [{"type": "image", "path": "a.jpg"}]}]}
    s = _crude_sample(rec, {"a.jpg": img})
    s["json"] = json.dumps(rec)  # str, not bytes
    out = cook_mm_chat_wds(s)
    assert out.image_bytes == [img]


def test_cooker_positional_fallback_for_mismatched_field_names():
    a, b = _jpeg_bytes(color=(10, 10, 200)), _jpeg_bytes(color=(10, 200, 10))
    rec = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": "first.jpg"},
                    {"type": "image", "path": "second.jpg"},
                ],
            }
        ]
    }
    # in-tar field names do NOT match the json path basenames -> positional
    sample = _crude_sample(rec, {"0001.aaa_img0.jpg": a, "0001.bbb_img1.jpg": b})
    out = cook_mm_chat_wds(sample)
    assert out.image_bytes == [a, b]  # assigned in field-sorted order


def test_cooker_text_only_no_images():
    rec = {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]}
    out = cook_mm_chat_wds(_crude_sample(rec, {}))
    assert out.image_bytes == []


def test_resolve_wds_path():
    assert resolve_wds_path("msc://azure/data/x/y") == "msc://azure/data/x/y"
    # container-relative -> resolved through the MSC default profile/container
    assert resolve_wds_path("yiming/bee_stage2/train-wds").startswith("msc://")
    assert resolve_wds_path("yiming/bee_stage2/train-wds").endswith("yiming/bee_stage2/train-wds")


# ---------------------------------------------------------------------------
# Full encode path (needs the cached tokenizer) — the WDS branch yields a
# decoded image entry + tokenized text with an image sentinel.
# ---------------------------------------------------------------------------


def _encoder():
    try:
        from transformers import AutoTokenizer

        from vlm.models import VLMProcessor
        from vlm.utils import conversation as conversation_lib
    except Exception as e:  # pragma: no cover
        pytest.skip(f"transformers/VLMProcessor unavailable: {e}")
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except Exception as e:  # pragma: no cover - no HF cache
        pytest.skip(f"Qwen3 tokenizer unavailable: {e}")
    # bee_stage2 caption pretraining runs the 2-turn `plain` template (the WDS
    # path always selects a template version, exactly as real training does).
    conversation_lib.default_conversation = conversation_lib.conv_templates["plain"]
    ip = RawImageProcessor(patch_size=16, pooling_kernel_size=3, max_soft_tokens=280)
    proc = VLMProcessor(image_processor=ip, tokenizer=tok)
    data_args = DataArguments(is_multimodal=True)
    return WDSChatTaskEncoder(proc, data_args)


def test_wds_encode_yields_decoded_image_and_sentinel():
    te = _encoder()
    rec = {
        "messages": [
            {"role": "user", "content": [{"type": "image", "path": "a.jpg"}]},
            {"role": "assistant", "content": "a red rectangle"},
        ]
    }
    cooked = cook_mm_chat_wds(_crude_sample(rec, {"a.jpg": _jpeg_bytes()}))
    dd = te.encode_sample(cooked)
    # exactly one image sentinel in the tokenized ids
    assert int((dd["input_ids"] == te.data_args.image_token_index).sum()) == 1
    # one decoded encoder-free raw-patch entry (4-tuple, modality "image")
    assert len(dd["image"]) == 1
    patches, positions, _size, modality = dd["image"][0]
    assert modality == "image"
    assert patches.ndim == 2 and patches.shape[1] == 6912  # (48px patch)^2 * 3
    assert positions.shape[0] == patches.shape[0]
