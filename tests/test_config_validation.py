"""Config-validation regression tests for the silently-mis-trained combos that
``validate_dataset_config`` rejects:

  * #27 — ``batch_token_budget`` set without ``length_buckets``.
  * #14 — generation data/model patch-size mismatch with an independent embedder.
  * #5  — the 2-turn ``plain`` caption template composed with instruct data.
"""

import pytest

try:
    from vlm.config import (
        DatasetConfig,
        ModelConfig,
        TrainerConfig,
        validate_dataset_config,
    )
except ModuleNotFoundError as e:  # pragma: no cover - slim envs
    pytest.skip(f"vlm package not importable here: {e}", allow_module_level=True)


def _gen_model(*, enabled=True, independent=True, embed_patch_size=16) -> ModelConfig:
    m = ModelConfig()
    m.generation.enabled = enabled
    m.generation.independent_embed = independent
    m.generation.embed_patch_size = embed_patch_size
    return m


# ---------------------------------------------------------------------------
# #27 — batch_token_budget requires length_buckets
# ---------------------------------------------------------------------------


def test_budget_without_buckets_rejected():
    ds = DatasetConfig(type="energon", batch_token_budget=32000, length_buckets=None)
    with pytest.raises(ValueError, match="length_buckets"):
        validate_dataset_config(ds)


def test_budget_with_empty_buckets_rejected():
    ds = DatasetConfig(type="energon", batch_token_budget=32000, length_buckets=[])
    with pytest.raises(ValueError, match="length_buckets"):
        validate_dataset_config(ds)


def test_budget_with_buckets_ok():
    ds = DatasetConfig(type="energon", batch_token_budget=32000, length_buckets=[512, 1024])
    validate_dataset_config(ds)  # no raise


def test_buckets_without_budget_ok():
    ds = DatasetConfig(type="energon", length_buckets=[512, 1024])
    validate_dataset_config(ds)  # bucketing without a token budget is valid


def test_neither_ok():
    validate_dataset_config(DatasetConfig(type="energon"))  # no raise


# ---------------------------------------------------------------------------
# #14 — generation data/model patch-size must match when independent_embed
# ---------------------------------------------------------------------------


def test_gen_patch_mismatch_rejected():
    ds = DatasetConfig(type="energon", task="generation", gen_patch_size=48)
    with pytest.raises(ValueError, match="embed_patch_size"):
        validate_dataset_config(ds, _gen_model(embed_patch_size=16))


def test_gen_patch_none_with_independent_rejected():
    ds = DatasetConfig(type="energon", task="generation", gen_patch_size=None)
    with pytest.raises(ValueError, match="embed_patch_size"):
        validate_dataset_config(ds, _gen_model(embed_patch_size=16))


def test_gen_patch_match_ok():
    ds = DatasetConfig(type="energon", task="generation", gen_patch_size=16)
    validate_dataset_config(ds, _gen_model(embed_patch_size=16))  # no raise


def test_gen_patch_none_ok_when_not_independent():
    # Legacy reuse-the-connector path: gen_patch_size None is fine.
    ds = DatasetConfig(type="energon", task="generation", gen_patch_size=None)
    validate_dataset_config(ds, _gen_model(independent=False))  # no raise


def test_gen_patch_ignored_when_generation_disabled():
    ds = DatasetConfig(type="energon", task="generation", gen_patch_size=48)
    validate_dataset_config(ds, _gen_model(enabled=False, independent=True, embed_patch_size=16))


def test_gen_patch_ignored_for_understanding_task():
    ds = DatasetConfig(type="energon", task="understanding", gen_patch_size=48)
    validate_dataset_config(ds, _gen_model(embed_patch_size=16))  # no raise


# ---------------------------------------------------------------------------
# #5 — plain (2-turn caption) template vs instruct (multi-turn) data
# ---------------------------------------------------------------------------


def test_plain_with_instruct_rejected():
    ds = DatasetConfig(type="energon", conversation_kind="instruct")
    with pytest.raises(ValueError, match="plain"):
        validate_dataset_config(ds, None, TrainerConfig(version="plain"))


def test_v0_plain_with_instruct_rejected():
    ds = DatasetConfig(type="energon", conversation_kind="instruct")
    with pytest.raises(ValueError, match="v0_plain"):
        validate_dataset_config(ds, None, TrainerConfig(version="v0_plain"))


def test_plain_with_caption_ok():
    ds = DatasetConfig(type="energon", conversation_kind="caption")
    validate_dataset_config(ds, None, TrainerConfig(version="plain"))


def test_plain_with_auto_skips():
    ds = DatasetConfig(type="energon", conversation_kind="auto")
    validate_dataset_config(ds, None, TrainerConfig(version="plain"))


def test_instruct_with_chat_template_ok():
    ds = DatasetConfig(type="energon", conversation_kind="instruct")
    validate_dataset_config(ds, None, TrainerConfig(version="qwen_2_5"))


def test_no_trainer_skips_plain_check():
    ds = DatasetConfig(type="energon", conversation_kind="instruct")
    validate_dataset_config(ds)  # no trainer -> #5 not evaluated


# ---------------------------------------------------------------------------
# The pinned configs round-trip the check.
# ---------------------------------------------------------------------------


def test_energon_mix_marked_instruct():
    # energon-mix carries instruct data; declared instruct so plain rejects it.
    ds = DatasetConfig(type="energon", conversation_kind="instruct", name="energon-mix")
    with pytest.raises(ValueError):
        validate_dataset_config(ds, None, TrainerConfig(version="plain"))


def test_bee_stage2_wds_caption_ok_with_plain():
    ds = DatasetConfig(
        type="energon",
        conversation_kind="caption",
        wds_path="yiming/bee_stage2/train-wds",
        length_buckets=[512, 1024],
        batch_token_budget=32000,
        name="energon-bee-stage2-wds",
    )
    validate_dataset_config(ds, None, TrainerConfig(version="plain"))  # no raise
