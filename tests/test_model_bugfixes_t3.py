"""Regression tests for the model-forward / generation / preprocessing /
checkpoint bugs fixed on fm/fix-model-t3 (codereview-2026-06-28):

  #1  BREEN S2 learnable_query_placement persists/refreshes across reload
  #6  cross_modal_prefix_skip_ids covers every blanket-delimiter template
  #8  generation-module load guard (require_generation_modules)
  #10 Qwen local-JSON first-turn role/content schema (preprocess_qwen)
  #15 floating_point_ops on a stacked tensor image_position_ids
  #16 floating_point_ops counts BREEN <query> expansion
  #17 auto-resume skips optimizer-less checkpoints without losing weights
  #18 per-layer xmodal mask state cleared between generations
  #19 default all-ones attention mask in generate() so the xmodal install runs

The model tests build a tiny randomly-initialized encoder-free unified model
via the live dynamic classes (same recipe as tests/test_inference.py).
"""

import copy

import pytest

torch = pytest.importorskip("torch")

try:
    from vlm.data.data_arguments import DataArguments
    from vlm.data.dataset import preprocess_qwen, tokenizer_multimodal_token
    from vlm.models import get_dynamic_vlm
    from vlm.models.image_processing_raw import RawImageProcessor
    from vlm.vlm import _cross_modal_prefix_skip_ids, require_generation_modules
except ModuleNotFoundError as e:  # pragma: no cover - slim envs
    pytest.skip(f"vlm package not importable here: {e}", allow_module_level=True)

try:
    from transformers import AutoConfig, AutoTokenizer

    BASE_LM = "Qwen/Qwen3-0.6B"
    _TOKENIZER = AutoTokenizer.from_pretrained(BASE_LM)
    _BASE_CONFIG = AutoConfig.from_pretrained(BASE_LM)
except Exception as e:  # pragma: no cover - no HF cache
    pytest.skip(f"Qwen3 tokenizer/config unavailable: {e}", allow_module_level=True)

VISION_DIALS = dict(patch_size=4, pooling_kernel_size=1, max_soft_tokens=16)
PATCH_DIM = (4 * 1) ** 2 * 3  # 48


def _tiny_language_config_dict() -> dict:
    cfg = copy.deepcopy(_BASE_CONFIG)
    cfg.hidden_size = 32
    cfg.intermediate_size = 64
    cfg.num_hidden_layers = 2
    cfg.layer_types = ["full_attention"] * 2
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.head_dim = 8
    cfg.max_position_embeddings = 512
    return cfg.to_dict()


def _tiny_vlm_config_kwargs() -> dict:
    return dict(
        hf_name=BASE_LM,
        vision_config={**VISION_DIALS, "hf_name": None, "hidden_size": PATCH_DIM},
        connector_config={
            "name": "raw_patch",
            "type": "raw_patch",
            "mm_embed_dim": 32,
            "mm_posemb_size": VISION_DIALS["max_soft_tokens"],
        },
        audio_config=None,
        **_tiny_language_config_dict(),
        image_token="<image>",
        image_token_index=-200,
        audio_token="<audio>",
        audio_token_index=-201,
        ignore_index=-100,
        max_seq_length=512,
        padding_side="left",
        use_start_end_tokens=False,
        image_start_token="<im_start>",
        image_end_token="<im_end>",
        conversation_version="qwen_2_5",
    )


def _breen_config_kwargs(placement: str = "after_image") -> dict:
    kwargs = _tiny_vlm_config_kwargs()
    kwargs.update(
        learnable_query=True,
        learnable_query_num_query=8,
        learnable_query_placement=placement,
        query_token="<query>",
        query_token_index=-202,
    )
    return kwargs


def _build_tiny_model(breen: bool = False, **config_overrides):
    VLMForCausalLM, VLMConfig = get_dynamic_vlm(BASE_LM)
    kwargs = _breen_config_kwargs() if breen else _tiny_vlm_config_kwargs()
    kwargs.update(config_overrides)
    config = VLMConfig(**kwargs)
    torch.manual_seed(0)
    model = VLMForCausalLM(config)
    model.eval()
    return model


def _data_args() -> DataArguments:
    return DataArguments(audio_enabled=False)


# ---------------------------------------------------------------------------
# #1 — BREEN S2 learnable_query_placement persistence / refresh across reload
# ---------------------------------------------------------------------------


def test_learnable_query_placement_persists_and_refreshes(tmp_path):
    """An S1 checkpoint saved with after_image; S2 trains with after_text while
    loading from S1. The placement must round-trip through save_pretrained AND
    be overwritable so the saved S2 config (which inference reads) agrees with
    how S2 actually trained. The bug left the stale S1 value on reload."""
    VLMForCausalLM, VLMConfig = get_dynamic_vlm(BASE_LM)
    s1 = VLMForCausalLM(VLMConfig(**_breen_config_kwargs(placement="after_image")))
    s1.save_pretrained(tmp_path / "s1")

    # S2 from_pretrained: the checkpoint config still says after_image.
    reloaded = VLMForCausalLM.from_pretrained(tmp_path / "s1")
    assert reloaded.config.learnable_query_placement == "after_image"

    # vlm()'s branch-agnostic refresh sets it from data_args.query_placement.
    reloaded.config.learnable_query_placement = "after_text"
    reloaded.save_pretrained(tmp_path / "s2")

    s2 = VLMForCausalLM.from_pretrained(tmp_path / "s2")
    assert s2.config.learnable_query_placement == "after_text"  # inference agrees


# ---------------------------------------------------------------------------
# #10 — Qwen first-turn role/content schema must not KeyError
# ---------------------------------------------------------------------------


def test_preprocess_qwen_accepts_role_content_first_turn():
    src = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    out = preprocess_qwen([copy.deepcopy(src)], _TOKENIZER, _data_args(), has_image=False)
    assert out["input_ids"].shape == out["labels"].shape
    assert out["input_ids"].shape[0] == 1


def test_preprocess_qwen_role_content_with_leading_system():
    src = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    out = preprocess_qwen([copy.deepcopy(src)], _TOKENIZER, _data_args(), has_image=False)
    assert out["input_ids"].shape == out["labels"].shape


def test_preprocess_qwen_legacy_from_value_still_works():
    src = [
        {"from": "human", "value": "hello"},
        {"from": "gpt", "value": "hi"},
    ]
    out = preprocess_qwen([copy.deepcopy(src)], _TOKENIZER, _data_args(), has_image=False)
    assert out["input_ids"].shape == out["labels"].shape


# ---------------------------------------------------------------------------
# #8 — generation-module load guard
# ---------------------------------------------------------------------------


def test_require_generation_modules_guard():
    import types

    understanding = types.SimpleNamespace(gen_x_head=None)
    with pytest.raises(ValueError, match="generation"):
        require_generation_modules(understanding, requested=True)
    require_generation_modules(understanding, requested=False)  # not requested: ok
    require_generation_modules(types.SimpleNamespace(gen_x_head=object()), requested=True)


def test_tiny_understanding_model_has_no_generation_modules():
    """The guard's premise: an understanding-only build carries gen_x_head=None,
    so a target_patches batch would route into forward_generation and crash."""
    assert getattr(_build_tiny_model(), "gen_x_head", None) is None


# ---------------------------------------------------------------------------
# #15 / #16 — floating_point_ops
# ---------------------------------------------------------------------------


def test_floating_point_ops_tensor_image_position_ids():
    model = _build_tiny_model()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    # generation batching stacks image_position_ids into a (B, N, 2) tensor;
    # `if image_position_ids:` used to raise the ambiguous-truth RuntimeError.
    tensor_pos = torch.zeros(1, 4, 2, dtype=torch.long)
    flops = model.floating_point_ops(
        {
            "input_ids": input_ids,
            "image_position_ids": tensor_pos,
            "target_patches": torch.zeros(1, 4, PATCH_DIM),
        }
    )
    assert isinstance(flops, int) and flops > 0


def test_floating_point_ops_list_image_position_ids_unchanged():
    model = _build_tiny_model()
    input_ids = torch.tensor([[-200, 2, 3]])  # one <image> sentinel
    list_pos = [torch.zeros(4, 2, dtype=torch.long)]
    flops = model.floating_point_ops({"input_ids": input_ids, "image_position_ids": list_pos})
    assert isinstance(flops, int) and flops > 0


def test_floating_point_ops_none_image_position_ids():
    model = _build_tiny_model()
    flops = model.floating_point_ops({"input_ids": torch.tensor([[1, 2, 3]])})
    assert isinstance(flops, int) and flops > 0


def test_floating_point_ops_counts_breen_query_expansion():
    """Each <query> sentinel expands into num_query learnable-query rows (8 here);
    the FLOPs estimate must reflect that, not count it as 1."""
    model = _build_tiny_model(breen=True)
    rows = 8  # num_query
    base = model.floating_point_ops({"input_ids": torch.tensor([[1, 2, 3, 4]])})
    with_q = model.floating_point_ops({"input_ids": torch.tensor([[1, 2, 3, -202]])})
    per_token = base // 4  # base counts 4 tokens
    # the <query> sentinel replaces 1 counted token with `rows` rows.
    assert with_q == base + (rows - 1) * per_token


# ---------------------------------------------------------------------------
# #18 — per-layer xmodal mask state cleared after generate()
# ---------------------------------------------------------------------------


def test_generate_clears_stale_xmodal_layer_state():
    model = _build_tiny_model()
    for layer in model.model.layers:
        layer.self_attn._xmodal_mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    model._xmodal_gen_mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    model._ve_gen_mask = torch.ones(1, 4, 1)

    ids = _TOKENIZER("hello", return_tensors="pt").input_ids
    with torch.inference_mode():
        model.generate(
            inputs=ids,
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=_TOKENIZER.pad_token_id or _TOKENIZER.eos_token_id,
        )
    for layer in model.model.layers:
        assert getattr(layer.self_attn, "_xmodal_mask", None) is None
    assert model._xmodal_gen_mask is None
    assert model._ve_gen_mask is None


# ---------------------------------------------------------------------------
# #19 — generate() without attention_mask must still install the xmodal mask
# ---------------------------------------------------------------------------


def test_generate_without_attention_mask_runs_xmodal_install():
    model = _build_tiny_model(cross_modal_mask_mode="img2q_window")
    proc = RawImageProcessor(**VISION_DIALS)
    try:
        from PIL import Image
    except ModuleNotFoundError:  # pragma: no cover
        pytest.skip("PIL unavailable")
    out = proc.preprocess(Image.new("RGB", (16, 8), (10, 20, 30)))
    prompt_ids = tokenizer_multimodal_token(
        "<image>\ndescribe", _TOKENIZER, _data_args(), return_tensors="pt"
    ).unsqueeze(0)

    calls = []
    orig = model.install_xmodal_masks

    def spy(attn2d, image_block_ids, labels, query_block_ids=None):
        calls.append(attn2d is not None)
        return None  # don't apply a 4D mask; keep generation on the plain path

    model.install_xmodal_masks = spy
    try:
        with torch.inference_mode():
            model.generate(
                inputs=prompt_ids,
                images=out["pixel_values"],
                image_position_ids=out["image_position_ids"],
                max_new_tokens=2,
                do_sample=False,
                pad_token_id=_TOKENIZER.pad_token_id or _TOKENIZER.eos_token_id,
            )
    finally:
        model.install_xmodal_masks = orig

    # Without the default-mask fix, attention_mask stays None and the install is
    # skipped entirely. With the fix it runs with a real 2D mask.
    assert calls and calls[0] is True


# ---------------------------------------------------------------------------
# #6 — cross_modal_prefix_skip_ids must cover EVERY template whose preprocessor
#      blanket-unmasks delimiters (qwen AND llama3), not just qwen
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer stub mapping the delimiter tokens
    _cross_modal_prefix_skip_ids needs to fixed ids, plus newline encoding."""

    unk_token_id = 0

    _VOCAB = {
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
        "<|begin_of_text|>": 128000,
        "<|start_header_id|>": 128006,
        "<|end_header_id|>": 128007,
        "<|eot_id|>": 128009,
    }

    def convert_tokens_to_ids(self, tok):
        return self._VOCAB.get(tok, self.unk_token_id)

    def encode(self, text, add_special_tokens=False):
        return {"\n": [198], "\n\n": [271]}[text]


def test_prefix_skip_ids_qwen_uses_real_chatml_delimiters():
    # real Qwen3 tokenizer: <|im_start|>, <|im_end|>, newline
    assert _cross_modal_prefix_skip_ids("qwen_2_5", _TOKENIZER) == [151644, 151645, 198]


def test_prefix_skip_ids_llama3_uses_llama_delimiters():
    """preprocess_llama3 blanket-unmasks the SAME way qwen does, so a llama3 run
    must get a llama3 skip set — not None — else the cross-modal arm silently
    collapses to plain causal (#6 prefix-collapse, llama3 gap)."""
    assert _cross_modal_prefix_skip_ids("llava_llama_3", _FakeTokenizer()) == [
        128000,
        128006,
        128007,
        128009,
        271,
    ]


def test_prefix_skip_ids_none_for_non_delimiter_templates():
    """gemma (round-based masking) and unknown versions supervise answer tokens
    only, so no delimiter skip set is needed (prefix-from-labels is correct)."""
    assert _cross_modal_prefix_skip_ids("gemma_instruct", _FakeTokenizer()) is None
    assert _cross_modal_prefix_skip_ids("does_not_exist", _FakeTokenizer()) is None


def test_prefix_skip_ids_drops_unk_and_invalid():
    """unk / negative / non-int ids are dropped; an all-invalid set -> None."""

    class _BadTokenizer:
        unk_token_id = 151644  # pretend <|im_start|> resolves to unk

        def convert_tokens_to_ids(self, tok):
            return {"<|im_start|>": 151644, "<|im_end|>": -1}.get(tok, 151644)

        def encode(self, text, add_special_tokens=False):
            return []

    assert _cross_modal_prefix_skip_ids("qwen_2_5", _BadTokenizer()) is None


# ---------------------------------------------------------------------------
# #17 — auto-resume skips optimizer-less (save_only_model) checkpoints, but never
#       silently restarts from base when ONLY weights-only snapshots exist
# ---------------------------------------------------------------------------


def _make_ckpt(root, step, resumable):
    import os

    d = os.path.join(str(root), f"checkpoint-{step}")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "model.safetensors"), "w").close()  # every ckpt has weights
    if resumable:
        open(os.path.join(d, "optimizer.pt"), "w").close()  # full ckpt has optimizer
    return d


def test_auto_resume_prefers_newest_older_resumable(tmp_path):
    """Mixed history: latest is weights-only, older ones are full -> resume the
    newest OLDER resumable checkpoint (optimizer state preserved). Unchanged for
    every backend, so it holds even where weights-only resume is unsupported."""
    from vlm.train.train import _resolve_auto_resume_checkpoint

    _make_ckpt(tmp_path, 100, resumable=True)
    _make_ckpt(tmp_path, 150, resumable=True)
    latest = _make_ckpt(tmp_path, 200, resumable=False)
    for supported in (True, False):
        chosen = _resolve_auto_resume_checkpoint(
            str(tmp_path), weights_only_resume_supported=supported
        )
        assert chosen.endswith("checkpoint-150")
        assert chosen != latest


def test_auto_resume_weights_only_falls_back_to_latest(tmp_path, caplog):
    """All-weights-only history (e.g. config/trainer/pretrain.yaml
    save_only_model=True) on a backend that CAN load a weights-only snapshot
    (plain DDP/single process): no resumable checkpoint exists, so resume WEIGHTS
    ONLY from the latest snapshot (loud warning) instead of silently restarting
    from base and discarding all prior training."""
    import logging

    from vlm.train.train import _resolve_auto_resume_checkpoint

    _make_ckpt(tmp_path, 100, resumable=False)
    latest = _make_ckpt(tmp_path, 200, resumable=False)
    with caplog.at_level(logging.WARNING):
        chosen = _resolve_auto_resume_checkpoint(str(tmp_path), weights_only_resume_supported=True)
    assert chosen == latest
    assert "WEIGHTS ONLY" in caplog.text


def test_auto_resume_weights_only_safe_skips_on_deepspeed_fsdp(tmp_path, caplog):
    """All-weights-only history on DeepSpeed/FSDP (weights-only resume
    unsupported): feeding the snapshot to trainer.train(resume_from_checkpoint=...)
    would crash deepspeed_load_checkpoint with 'Can't find a valid checkpoint', so
    SAFE-SKIP (None, train from the loaded base) with a loud warning — never a
    crash loop."""
    import logging

    from vlm.train.train import _resolve_auto_resume_checkpoint

    _make_ckpt(tmp_path, 100, resumable=False)
    _make_ckpt(tmp_path, 200, resumable=False)
    with caplog.at_level(logging.WARNING):
        chosen = _resolve_auto_resume_checkpoint(str(tmp_path), weights_only_resume_supported=False)
    assert chosen is None
    assert "SKIPPING auto-resume" in caplog.text


def test_auto_resume_latest_already_resumable(tmp_path):
    """Latest checkpoint already carries optimizer state -> resume it directly,
    regardless of backend."""
    from vlm.train.train import _resolve_auto_resume_checkpoint

    _make_ckpt(tmp_path, 100, resumable=True)
    latest = _make_ckpt(tmp_path, 200, resumable=True)
    for supported in (True, False):
        assert (
            _resolve_auto_resume_checkpoint(str(tmp_path), weights_only_resume_supported=supported)
            == latest
        )


def test_auto_resume_no_checkpoints_is_scratch(tmp_path):
    """Empty output_dir (or no dir) -> train from scratch (None)."""
    import os

    from vlm.train.train import _resolve_auto_resume_checkpoint

    assert (
        _resolve_auto_resume_checkpoint(str(tmp_path), weights_only_resume_supported=True) is None
    )
    assert (
        _resolve_auto_resume_checkpoint(
            os.path.join(str(tmp_path), "missing"), weights_only_resume_supported=True
        )
        is None
    )
