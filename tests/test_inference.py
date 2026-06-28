"""Tests for the inference path (vlm.inference) and the hub-export templates.

Pins:
  - prompt construction parity with the training-side preprocess functions
    (plain and qwen templates);
  - conv-mode resolution (explicit > checkpoint-recorded > path heuristic);
  - placeholder injection mirror of inject_missing_media_tokens;
  - end-to-end CPU generate on a tiny randomly-initialized encoder-free
    unified model (image + audio), via the live dynamic classes;
  - the same end-to-end generate through the RENDERED hub-export templates
    (templates/*.j2), so the exported artifact stays in sync with the live
    model code.
"""

import copy
import importlib
import shutil
import sys
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
        preprocess_plain,
        preprocess_qwen,
        tokenizer_multimodal_token,
    )
    from vlm.inference.eval import (
        QWEN_SYSTEM_MESSAGE,
        build_prompt,
        ensure_placeholders,
        generate_response,
        resolve_conv_mode,
    )
    from vlm.inference.generator import process_images
    from vlm.models import VLMProcessor, get_dynamic_vlm
    from vlm.models.image_processing_raw import RawImageProcessor
except ModuleNotFoundError as e:  # pragma: no cover - slim envs
    pytest.skip(f"vlm package not importable here: {e}", allow_module_level=True)

try:
    from transformers import AutoConfig, AutoTokenizer

    BASE_LM = "Qwen/Qwen3-0.6B"
    _TOKENIZER = AutoTokenizer.from_pretrained(BASE_LM)
    _BASE_CONFIG = AutoConfig.from_pretrained(BASE_LM)
except Exception as e:  # pragma: no cover - no HF cache
    pytest.skip(f"Qwen3 tokenizer/config unavailable: {e}", allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "src" / "vlm" / "models"
TEMPLATES_DIR = REPO_ROOT / "templates"


# ---------------------------------------------------------------------------
# prompt parity with training preprocessing
# ---------------------------------------------------------------------------


def _data_args() -> DataArguments:
    return DataArguments(audio_enabled=True)


def test_plain_prompt_matches_training_masked_prefix():
    """The plain inference prompt must tokenize to exactly the span that
    preprocess_plain masks out (the media-placeholder prefix)."""
    data_args = _data_args()
    sources = [
        [
            {"from": "human", "value": "<image>\n<audio>\nignored text"},
            {"from": "gpt", "value": "a red square"},
        ]
    ]
    out = preprocess_plain(copy.deepcopy(sources), _TOKENIZER, data_args)
    train_ids = out["input_ids"][0]
    labels = out["labels"][0]
    masked_len = int((labels == data_args.ignore_index).sum())

    prompt = build_prompt("plain", "<image>\n<audio>\nignored text", data_args)
    assert prompt == "<image><audio>"
    prompt_ids = tokenizer_multimodal_token(prompt, _TOKENIZER, data_args, return_tensors="pt")
    assert prompt_ids.tolist() == train_ids[:masked_len].tolist()


def test_qwen_prompt_matches_training_prefix():
    """The qwen inference prompt must be a strict prefix of what
    preprocess_qwen produced for the same user turn.

    Caveat (tokenizer-merge boundary): training tokenizes
    "<|im_start|>assistant\\n" TOGETHER with the answer, so an answer starting
    with "\\n" merges into "\\n\\n" (token 271) and the strict-prefix property
    would not hold for that sample. This pins the prompt-side construction
    only; answers here are chosen not to merge across the boundary (the
    standard ChatML inference convention shares this boundary behavior).
    """
    data_args = _data_args()
    user_text = "<image>\nWhat is in the picture?"
    sources = [
        [
            {"from": "human", "value": user_text},
            {"from": "gpt", "value": "A cat."},
        ]
    ]
    out = preprocess_qwen(copy.deepcopy(sources), _TOKENIZER, data_args, has_image=True)
    train_ids = out["input_ids"][0].tolist()

    prompt = build_prompt("qwen_2_5", user_text, data_args)
    assert prompt.startswith(f"<|im_start|>system\n{QWEN_SYSTEM_MESSAGE}<|im_end|>\n")
    assert prompt.endswith("<|im_start|>assistant\n")
    prompt_ids = tokenizer_multimodal_token(prompt, _TOKENIZER, data_args)
    assert train_ids[: len(prompt_ids)] == prompt_ids


def test_qwen_prompt_uses_training_system_message():
    # preprocess_qwen hardcodes "You are a helpful assistant." for every
    # sample; the conv template's own system string is dead at train time.
    assert QWEN_SYSTEM_MESSAGE == "You are a helpful assistant."
    prompt = build_prompt("qwen_2_5", "hi", _data_args())
    assert "You are Qwen" not in prompt


def test_legacy_v1_prompt_moves_image_token_to_front():
    data_args = _data_args()
    prompt = build_prompt("v1", "What is this? <image>", data_args)
    # preprocess_multimodal parity: image token stripped and moved to front
    # of the user turn.
    assert "<image>\nWhat is this?" in prompt
    assert prompt.index("<image>") < prompt.index("What is this?")


# ---------------------------------------------------------------------------
# conv-mode resolution + placeholder injection
# ---------------------------------------------------------------------------


class _FakeConfig:
    def __init__(self, **kwargs: Any):
        self.__dict__.update(kwargs)


def test_resolve_conv_mode_priority():
    cfg = _FakeConfig(conversation_version="plain")
    # explicit wins over recorded
    assert resolve_conv_mode(cfg, conv_mode="qwen_2_5") == "qwen_2_5"
    # recorded wins over path heuristic
    assert resolve_conv_mode(cfg, pretrained="x/llava-v1.5") == "plain"
    # path heuristic as last resort (warns)
    with pytest.warns(UserWarning, match="conversation_version"):
        assert resolve_conv_mode(_FakeConfig(), pretrained="ckpt/qwen3-1.7b-sft") == "qwen_2_5"
    # nothing to go on -> explicit error
    with pytest.raises(ValueError, match="conversation_version"):
        resolve_conv_mode(_FakeConfig())
    # unknown explicit mode -> error
    with pytest.raises(ValueError, match="unknown conv_mode"):
        resolve_conv_mode(cfg, conv_mode="nope")


def test_ensure_placeholders_mirror_of_injection():
    data_args = _data_args()
    # media present, no placeholder -> prepended, images before audios
    assert ensure_placeholders("describe", 2, 1, data_args) == "<image>\n<image>\n<audio>\ndescribe"
    # counts match -> untouched
    assert ensure_placeholders("<image> hi", 1, 0, data_args) == "<image> hi"
    # mismatch -> loud error (same rule as inject_missing_media_tokens)
    with pytest.raises(ValueError, match="placeholder"):
        ensure_placeholders("<image> <image> hi", 1, 0, data_args)


def test_process_images_rejects_raw_image_processor():
    with pytest.raises(TypeError, match="encoder-free"):
        process_images(
            [PIL_Image.new("RGB", (32, 32))],
            RawImageProcessor(patch_size=4, pooling_kernel_size=1, max_soft_tokens=16),
            type("Cfg", (), {"image_aspect_ratio": "pad"})(),
        )


# ---------------------------------------------------------------------------
# tiny encoder-free unified model (CPU, random init)
# ---------------------------------------------------------------------------

VISION_DIALS = dict(patch_size=4, pooling_kernel_size=1, max_soft_tokens=16)
PATCH_DIM = (4 * 1) ** 2 * 3  # 48
AUDIO_FRAME = 64


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
        # get_dynamic_vlm's checkpoint-reload fallback reads this from
        # config.json (model_type "vlm" is not registered with AutoConfig).
        hf_name=BASE_LM,
        vision_config={**VISION_DIALS, "hf_name": None, "hidden_size": PATCH_DIM},
        connector_config={
            "name": "raw_patch",
            "type": "raw_patch",
            "mm_embed_dim": 32,
            "mm_posemb_size": VISION_DIALS["max_soft_tokens"],
        },
        audio_config={
            "enabled": True,
            "name": "raw_waveform",
            "type": "raw_waveform",
            "samples_per_token": AUDIO_FRAME,
            "sampling_rate": 16000,
            "max_audio_tokens": 10,
        },
        **_tiny_language_config_dict(),
        # LanguageModelConfig fields normally flattened in by vlm.load_model
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


@pytest.fixture(scope="module")
def tiny_model():
    VLMForCausalLM, VLMConfig = get_dynamic_vlm(BASE_LM)
    config = VLMConfig(**_tiny_vlm_config_kwargs())
    torch.manual_seed(0)
    model = VLMForCausalLM(config)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tiny_processor():
    return VLMProcessor(
        image_processor=RawImageProcessor(**VISION_DIALS),
        tokenizer=_TOKENIZER,
    )


@pytest.fixture(scope="module")
def wav_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("audio")
    rng = np.random.default_rng(0)
    path = root / "x.wav"
    sf.write(path, rng.normal(size=16000).astype(np.float32) * 0.1, 16000)
    return path


def test_generate_response_unified_qwen(tiny_model: Any, tiny_processor: Any, wav_path: Path):
    image = PIL_Image.new("RGB", (20, 10), (255, 0, 0))
    text = generate_response(
        tiny_model,
        tiny_processor,
        query="What do you see and hear?",
        images=image,
        audios=str(wav_path),
        max_new_tokens=3,
    )
    assert isinstance(text, str)


def test_generate_response_unified_plain(tiny_model: Any, tiny_processor: Any):
    image = PIL_Image.new("RGB", (20, 10), (0, 255, 0))
    text = generate_response(
        tiny_model,
        tiny_processor,
        images=image,
        conv_mode="plain",
        max_new_tokens=3,
    )
    assert isinstance(text, str)


def test_generate_response_missing_position_ids_error(tiny_model: Any):
    # the model-level guard for hand-rolled callers
    with pytest.raises(ValueError, match="image_position_ids"):
        tiny_model.encode_raw_patches([torch.zeros(4, PATCH_DIM)], None)


def test_generate_response_audio_without_pathway(tiny_processor: Any, wav_path: Path):
    VLMForCausalLM, VLMConfig = get_dynamic_vlm(BASE_LM)
    kwargs = _tiny_vlm_config_kwargs()
    kwargs["audio_config"] = None
    config = VLMConfig(**kwargs)
    model = VLMForCausalLM(config)
    model.eval()
    with pytest.raises(ValueError, match="audio"):
        generate_response(model, tiny_processor, query="hi", audios=str(wav_path), max_new_tokens=2)


def test_eval_model_save_load_roundtrip(
    tiny_model: Any, tiny_processor: Any, tmp_path: Path
) -> None:
    """save_pretrained -> load_model -> eval_model end-to-end on CPU.

    Pins the full checkpoint reload path: get_dynamic_vlm's config.json
    hf_name fallback, VLMProcessor.from_pretrained's RawImageProcessor
    branch, conversation_version resolution, and the one-shot wrapper.
    NOTE: must run before the exported_pkg fixture is first used — that
    fixture's module registers model_type "vlm" with AutoConfig process-wide,
    which would change get_dynamic_vlm's resolution path.
    """
    from vlm.inference.eval import eval_model

    ckpt = tmp_path / "ckpt"
    tiny_model.save_pretrained(ckpt)
    tiny_processor.save_pretrained(ckpt)
    image_path = tmp_path / "img.png"
    PIL_Image.new("RGB", (20, 10), (10, 20, 30)).save(image_path)

    out = eval_model(
        str(ckpt),
        query="What is this?",
        image_path=str(image_path),
        max_new_tokens=3,
        bf16=False,
        device="cpu",
    )
    assert isinstance(out, str)


# ---------------------------------------------------------------------------
# rendered hub-export templates
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def exported_pkg(tmp_path_factory: pytest.TempPathFactory):
    jinja2 = pytest.importorskip("jinja2")
    pkg_root = tmp_path_factory.mktemp("export")
    pkg_dir = pkg_root / "vlm_export_under_test"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    # Sibling modules the rendered modeling_vlm.py imports (mirrors
    # push_to_hub._copy_from_models). Without the gen_*/xmodal_mask/visual_distill
    # modules the rendered file would not import.
    for mod in (
        "connectors.py",
        "image_processing_raw.py",
        "xmodal_mask.py",
        "gen_diffusion.py",
        "gen_image.py",
        "gen_rope.py",
        "visual_distill.py",
        "gen_perceptual.py",
    ):
        shutil.copy(MODELS_DIR / mod, pkg_dir / mod)

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATES_DIR))
    modeling = env.get_template("modeling_vlm.py.j2").render(
        parent_class="Qwen3Model", causal_parent_class="Qwen3ForCausalLM"
    )
    configuration = env.get_template("configuration_vlm.py.j2").render(parent_class="Qwen3Config")
    assert "torch.arrange" not in modeling  # the classic typo, fixed
    assert 'inputs.pop("cache_position", None)' in modeling  # v5 KeyError guard
    (pkg_dir / "modeling_vlm.py").write_text(modeling)
    (pkg_dir / "configuration_vlm.py").write_text(configuration)

    sys.path.insert(0, str(pkg_root))
    try:
        module = importlib.import_module("vlm_export_under_test.modeling_vlm")
    finally:
        sys.path.remove(str(pkg_root))
    return module


def test_exported_template_unified_generate(exported_pkg: Any, wav_path: Path):
    """The rendered hub artifact must drive the encoder-free unified model:
    raw patches + image_position_ids + audio frames through generate()."""
    config = exported_pkg.VLMConfig(**_tiny_vlm_config_kwargs())
    torch.manual_seed(0)
    model = exported_pkg.VLMForCausalLM(config)
    model.eval()

    processor = RawImageProcessor(**VISION_DIALS)
    out = processor.preprocess(PIL_Image.new("RGB", (20, 10), (0, 0, 255)))
    images = out["pixel_values"]
    image_position_ids = out["image_position_ids"]

    data, _ = sf.read(wav_path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data).mean(dim=1)[: 5 * AUDIO_FRAME]
    audios = [wav.view(-1, AUDIO_FRAME)]

    prompt_ids = tokenizer_multimodal_token(
        "<image>\n<audio>\ndescribe", _TOKENIZER, _data_args(), return_tensors="pt"
    ).unsqueeze(0)

    with torch.inference_mode():
        gen = model.generate(
            inputs=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            images=images,
            image_position_ids=image_position_ids,
            audios=audios,
            max_new_tokens=3,
            do_sample=False,
            pad_token_id=_TOKENIZER.pad_token_id or _TOKENIZER.eos_token_id,
        )
    assert gen.shape[0] == 1 and gen.shape[1] <= 3


def test_exported_template_text_only_generate(exported_pkg: Any):
    config = exported_pkg.VLMConfig(**_tiny_vlm_config_kwargs())
    model = exported_pkg.VLMForCausalLM(config)
    model.eval()
    ids = _TOKENIZER("hello", return_tensors="pt").input_ids
    with torch.inference_mode():
        gen = model.generate(
            inputs=ids,
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=_TOKENIZER.pad_token_id or _TOKENIZER.eos_token_id,
        )
    assert gen.shape[1] <= 2


def test_exported_config_roundtrip(exported_pkg: Any, tmp_path: Path):
    """audio_config and the encoder-free vision dials must survive a
    save_pretrained/from_pretrained round trip of the exported config."""
    config = exported_pkg.VLMConfig(**_tiny_vlm_config_kwargs())
    config.save_pretrained(tmp_path)
    cfg_module = importlib.import_module("vlm_export_under_test.configuration_vlm")
    reloaded = cfg_module.VLMConfig.from_pretrained(tmp_path)
    assert reloaded.audio_config is not None
    assert int(reloaded.audio_config.samples_per_token) == AUDIO_FRAME
    assert getattr(reloaded.vision_config, "hf_name", None) is None
    assert reloaded.connector_config.type == "raw_patch"
    assert reloaded.conversation_version == "qwen_2_5"


# ---------------------------------------------------------------------------
# export coverage for the live arms (#7): BREEN learnable queries, visual-FFN
# experts, visual-prefix, text->image generation. A stale hand-maintained
# template silently dropped these modules and exported broken models; these
# pin the rendered artifact to the live model per arm.
# ---------------------------------------------------------------------------

_EXPORT_ARMS: dict[str, dict] = {
    "visual_expert": {"visual_expert": True},
    "visual_prefix": {
        "visual_prefix": True,
        "visual_prefix_depth": 2,
        "visual_prefix_heads": 4,
        "visual_prefix_intermediate": 64,
    },
    "generation": {"generation": True},
    "breen": {
        "learnable_query": True,
        "visual_expert": True,
        "query_token": "<query>",
        "query_token_index": -202,
        "query_placement": "after_image",
        "learnable_query_num_fine": 5,
        "learnable_query_num_coarse": 3,
    },
}


def test_export_template_in_sync_with_live_model():
    """#7 anti-drift: the committed template must equal a fresh generation from
    the live modeling_vlm.py, so a model change that isn't re-exported fails CI
    instead of silently shipping a broken hub artifact. Refresh with
    `uv run python -m vlm.utils.export_template`."""
    from vlm.utils.export_template import TEMPLATE_PATH, build_modeling_template

    assert build_modeling_template() == TEMPLATE_PATH.read_text(), (
        "templates/modeling_vlm.py.j2 is stale relative to "
        "src/vlm/models/modeling_vlm.py — regenerate it with "
        "`uv run python -m vlm.utils.export_template`."
    )


@pytest.mark.parametrize("arm", sorted(_EXPORT_ARMS))
def test_exported_template_state_dict_parity(exported_pkg: Any, arm: str):
    """#7: the exported model must build the SAME module set (state_dict keys)
    as the live dynamic model for every arm, so checkpoints with experts /
    visual-prefix / generation / learnable queries load with no dropped or
    unexpected weights (a dropped module = a silently broken export)."""
    kw = {**_tiny_vlm_config_kwargs(), **_EXPORT_ARMS[arm]}
    live_cls, live_cfg_cls = get_dynamic_vlm(BASE_LM)
    torch.manual_seed(0)
    live = live_cls(live_cfg_cls(**kw))
    torch.manual_seed(0)
    exported = exported_pkg.VLMForCausalLM(exported_pkg.VLMConfig(**kw))
    assert set(live.state_dict()) == set(exported.state_dict())


@pytest.mark.parametrize("arm", ["visual_expert", "breen"])
def test_exported_template_numerical_parity(exported_pkg: Any, arm: str):
    """#7: with identical weights the exported model must produce bit-identical
    logits to the live model — covering the routing-sensitive arms (visual-FFN
    expert blending; BREEN <query> splice + expert routing)."""
    kw = {**_tiny_vlm_config_kwargs(), **_EXPORT_ARMS[arm]}
    live_cls, live_cfg_cls = get_dynamic_vlm(BASE_LM)
    torch.manual_seed(0)
    live = live_cls(live_cfg_cls(**kw)).eval()
    exported = exported_pkg.VLMForCausalLM(exported_pkg.VLMConfig(**kw)).eval()
    exported.load_state_dict(live.state_dict())

    processor = RawImageProcessor(**VISION_DIALS)
    out = processor.preprocess(PIL_Image.new("RGB", (20, 10), (7, 9, 11)))
    is_breen = bool(_EXPORT_ARMS[arm].get("learnable_query"))
    da = DataArguments(
        audio_enabled=True,
        learnable_query_enabled=is_breen,
        query_token="<query>",
        query_token_index=-202,
    )
    prompt = "<image>\n<query>\nhi" if is_breen else "<image>\nhi"
    ids = tokenizer_multimodal_token(prompt, _TOKENIZER, da, return_tensors="pt").unsqueeze(0)
    feats = dict(images=out["pixel_values"], image_position_ids=out["image_position_ids"])
    with torch.inference_mode():
        live_logits = live(input_ids=ids, attention_mask=torch.ones_like(ids), **feats).logits
        exp_logits = exported(input_ids=ids, attention_mask=torch.ones_like(ids), **feats).logits
    assert torch.equal(live_logits, exp_logits)


# ---------------------------------------------------------------------------
# prompt-construction regressions (#2 BREEN <query>, #9 image_position order)
# ---------------------------------------------------------------------------


def test_media_regex_mirrors_training_pattern():
    """#2: inference _media_regex must equal training _media_pattern for both
    plain and BREEN (the plain template extracts placeholders via this regex —
    a divergence drops the injected <query> tokens)."""
    from vlm.data.dataset import _media_pattern
    from vlm.inference.eval import _media_regex

    for learnable in (False, True):
        da = DataArguments(learnable_query_enabled=learnable, query_token="<query>")
        assert _media_regex(da) == _media_pattern(da)


def test_plain_breen_prompt_keeps_query_placeholder():
    """#2: BREEN + plain template must keep the injected <query> alongside
    <image> (training preprocess_plain keeps both via _media_pattern). The
    regression dropped <query>, producing a query-block-free prompt."""
    da_breen = DataArguments(learnable_query_enabled=True, query_token="<query>")
    assert build_prompt("plain", "<image>\n<query>\ncaption", da_breen) == "<image><query>"
    # control: with BREEN off, a literal "<query>" is plain text and is dropped
    da_plain = DataArguments(learnable_query_enabled=False)
    assert build_prompt("plain", "<image>\n<query>\ncaption", da_plain) == "<image>"


def test_image_position_applied_after_placeholder_insertion(
    tiny_model: Any, tiny_processor: Any, monkeypatch: pytest.MonkeyPatch
):
    """#9: training injects the missing <image> placeholder FIRST, then
    repositions. A query with no placeholder + question_first must yield
    'describe\\n<image>', not the bare-prepended '<image>\\ndescribe'."""
    import vlm.inference.eval as eval_mod

    captured: dict[str, str] = {}
    orig_build_prompt = eval_mod.build_prompt

    def spy(conv_mode: str, query: str, data_args: Any) -> str:
        captured["query"] = query
        return orig_build_prompt(conv_mode, query, data_args)

    monkeypatch.setattr(eval_mod, "build_prompt", spy)
    # image_position is read via getattr(..., "keep") and not set on the tiny
    # config, so allow creating it (raising=False).
    monkeypatch.setattr(tiny_model.config, "image_position", "question_first", raising=False)
    image = PIL_Image.new("RGB", (20, 10), (1, 2, 3))
    generate_response(tiny_model, tiny_processor, query="describe", images=image, max_new_tokens=1)
    assert captured["query"] == "describe\n<image>"


class _StopAfterPrompt(Exception):
    """Short-circuit generate_response right after the prompt is assembled, so
    the audio-protection assertion runs on the repositioned query without also
    driving the model forward (whose splice would itself crash on a duplicated
    <audio> — we want to pin the placeholder count directly, not the crash)."""


@pytest.mark.parametrize("image_position", ["sandwich", "random"])
def test_inference_protects_audio_under_image_reposition(
    tiny_model: Any,
    tiny_processor: Any,
    wav_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image_position: str,
):
    """#8: with image + audio at inference, the eval call site must pass
    protected_tokens=(audio,) into apply_image_position so the <audio>
    placeholder keeps its 1:1 feature count, mirroring the training data paths.
    The sharp case is `sandwich`, which repeats the question text — without the
    fix the repeated copy carries <audio> twice and the splice misaligns; the
    `random` case pins that repositioning still emits exactly one <audio>."""
    import vlm.inference.eval as eval_mod

    captured: dict[str, str] = {}

    def spy(conv_mode: str, query: str, data_args: Any) -> str:
        captured["query"] = query
        raise _StopAfterPrompt

    monkeypatch.setattr(eval_mod, "build_prompt", spy)
    monkeypatch.setattr(tiny_model.config, "image_position", image_position, raising=False)
    image = PIL_Image.new("RGB", (20, 10), (4, 5, 6))
    with pytest.raises(_StopAfterPrompt):
        generate_response(
            tiny_model,
            tiny_processor,
            query="What do you see and hear?",
            images=image,
            audios=str(wav_path),
            max_new_tokens=1,
        )
    # exactly one <image> and one <audio> survive the reposition
    assert captured["query"].count("<audio>") == 1
    assert captured["query"].count("<image>") == 1
    if image_position == "sandwich":
        # the image question is repeated around the image, the audio is not
        assert captured["query"].count("What do you see and hear?") == 2


# ---------------------------------------------------------------------------
# generation-kwargs threading (#23) and one-shot forwarding (#30)
# ---------------------------------------------------------------------------


def test_generate_response_do_sample_independent_of_temperature(
    tiny_model: Any, tiny_processor: Any, monkeypatch: pytest.MonkeyPatch
):
    """#23: do_sample is honoured independently of temperature."""
    captured: dict[str, Any] = {}

    def fake_generate(input_ids: Any, **kwargs: Any) -> Any:
        captured.clear()
        captured.update(kwargs)
        return torch.zeros((1, 1), dtype=torch.long)

    monkeypatch.setattr(tiny_model, "generate", fake_generate)
    image = PIL_Image.new("RGB", (20, 10), (0, 0, 0))

    # do_sample=True with temperature=0: sampling enabled, temperature unset
    # (temperature=0 + do_sample=True is the invalid combination we avoid).
    generate_response(
        tiny_model,
        tiny_processor,
        query="hi",
        images=image,
        temperature=0.0,
        do_sample=True,
        max_new_tokens=1,
    )
    assert captured["do_sample"] is True
    assert "temperature" not in captured

    # do_sample=False with temperature>0: forced greedy.
    generate_response(
        tiny_model,
        tiny_processor,
        query="hi",
        images=image,
        temperature=0.7,
        do_sample=False,
        max_new_tokens=1,
    )
    assert captured["do_sample"] is False

    # default (None): derived from temperature (greedy at 0).
    generate_response(
        tiny_model,
        tiny_processor,
        query="hi",
        images=image,
        temperature=0.0,
        max_new_tokens=1,
    )
    assert captured["do_sample"] is False


def test_eval_model_forwards_image_aspect_ratio(monkeypatch: pytest.MonkeyPatch):
    """#30: eval_model exposes and forwards image_aspect_ratio."""
    from types import SimpleNamespace

    import vlm.inference.eval as eval_mod

    captured: dict[str, Any] = {}
    fake_model = SimpleNamespace(config=SimpleNamespace(conversation_version="qwen_2_5"))
    monkeypatch.setattr(eval_mod, "load_model", lambda *a, **k: (fake_model, object(), {}))
    monkeypatch.setattr(eval_mod, "generate_response", lambda *a, **k: captured.update(k) or "ok")
    eval_mod.eval_model("ckpt", query="hi", image_aspect_ratio="square")
    assert captured.get("image_aspect_ratio") == "square"


# ---------------------------------------------------------------------------
# lmms-eval adapter cache identity (#22)
# ---------------------------------------------------------------------------


def test_lmms_eval_cache_key_includes_doc_identity(monkeypatch: pytest.MonkeyPatch):
    """#22: two docs with identical context text but different images/docs must
    get distinct cache keys, so a repeated prompt with a different image does
    not collide on a stale cached response."""
    pytest.importorskip("lmms_eval")
    from types import SimpleNamespace

    from vlm.inference import lmms_eval as lm

    keys: list[Any] = []

    class FakeCache:
        def add_partial(self, attr: str, key: Any, value: Any) -> None:
            keys.append(key)

    adapter = lm.SmallVLM.__new__(lm.SmallVLM)
    adapter._model = object()
    adapter.processor = object()
    adapter.conv_mode = "qwen_2_5"
    adapter.image_aspect_ratio = None
    adapter._max_new_tokens = 4
    adapter._image_token = "<image>"
    adapter._audio_token = "<audio>"
    adapter.rank = 0
    adapter.cache_hook = FakeCache()
    docs = {0: {"img": "A"}, 1: {"img": "B"}}
    adapter.task_dict = {"task": {"val": docs}}

    monkeypatch.setattr(lm, "ChatMessages", lambda messages: messages)
    monkeypatch.setattr(lm, "generate_response", lambda *a, **k: "resp")
    monkeypatch.setattr(
        adapter, "_messages_to_query_and_media", lambda chat: ("same question", [chat], [])
    )

    requests = [
        SimpleNamespace(args=("same question", lambda doc: doc, {}, doc_id, "task", "val"))
        for doc_id in (0, 1)
    ]
    adapter.generate_until(requests)
    assert keys[0] != keys[1]
    assert keys[0][-1] == 0 and keys[1][-1] == 1  # doc_id distinguishes them
