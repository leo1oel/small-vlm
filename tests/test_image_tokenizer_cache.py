import pytest

transformers = pytest.importorskip("transformers")

from vlm.data.dataset import _get_preprocess_tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except OSError:
        pytest.skip("no network / model not cached")


def test_cache_returns_same_object(tokenizer):
    a = _get_preprocess_tokenizer(tokenizer, has_image=True)
    b = _get_preprocess_tokenizer(tokenizer, has_image=True)
    assert a is b


def test_image_variant_has_image_token(tokenizer):
    tok = _get_preprocess_tokenizer(tokenizer, has_image=True)
    assert tok.convert_tokens_to_ids("<image>") is not None
    assert tok.convert_tokens_to_ids("<image>") != tokenizer.unk_token_id


def test_original_tokenizer_not_mutated(tokenizer):
    before = len(tokenizer)
    _get_preprocess_tokenizer(tokenizer, has_image=True)
    assert len(tokenizer) == before


def test_no_image_variant_is_distinct(tokenizer):
    a = _get_preprocess_tokenizer(tokenizer, has_image=True)
    b = _get_preprocess_tokenizer(tokenizer, has_image=False)
    assert a is not b
