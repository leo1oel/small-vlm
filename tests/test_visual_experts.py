"""Tests for the three independently-toggleable per-decoder-layer visual experts
(EVEv2 "divide-and-conquer" / Mono-InternVL / BREEN; spec 2026-06-14): the FFN
expert (`mlp.mlp_visual`), the norm expert (`input_layernorm`/
`post_attention_layernorm` -> `.norm_visual`) and the attention expert
(`self_attn.{q,k,v,o}_proj.proj_visual`).

Two layers:
  * unit  — install/route/init/gate on a tiny raw Qwen3 backbone (fast, no VLM):
            independent toggling, routes ONLY vision(+query) tokens, init_from_text
            step-0 identity, near-identity gate, param classification.
  * model — a tiny CPU encoder-free VLM: experts wire from config at build, the
            text path stays bit-identical when init_from_text + gate off, and a
            save -> reload round-trip rebuilds the same experts and generates
            identically (train/infer parity; exercises the attention expert
            through KV-cache decode).
"""

from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3 import modeling_qwen3 as M

from vlm.models.modeling_vlm import (
    init_visual_expert_gates,
    init_visual_experts_from_text,
    install_visual_experts,
    is_visual_expert_param,
)

BASE_LM = "Qwen/Qwen3-0.6B"

# ---------------------------------------------------------------------------
# unit layer: tiny raw Qwen3 backbone
# ---------------------------------------------------------------------------

HID = 32


def _tiny_qwen3_config() -> Any:
    return M.Qwen3Config(
        hidden_size=HID,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,  # 4*8=32==hidden here; o_proj.in_features still read directly
        vocab_size=64,
        max_position_embeddings=128,
    )


def _build_tiny_lm(
    *,
    ffn: bool = False,
    norm: bool = False,
    attention: bool = False,
    gate: bool = False,
    layers: list[int] | None = None,
    enabled: bool = True,
) -> Any:
    """A 2-layer Qwen3 backbone with the requested experts installed + init."""
    cfg = _tiny_qwen3_config()
    torch.manual_seed(0)
    model = M.Qwen3Model(cfg)
    cfg.visual_expert = enabled
    cfg.visual_expert_ffn = ffn
    cfg.visual_expert_norm = norm
    cfg.visual_expert_attention = attention
    cfg.visual_expert_gate = gate
    cfg.visual_expert_layers = layers
    install_visual_experts(model, cfg)
    init_visual_experts_from_text(model)
    if gate:
        init_visual_expert_gates(model)
    return model


def _siblings_of(model: Any, attr: str) -> list[nn.Module]:
    return [
        m for m in getattr(model, "_visual_routed_modules", []) if m._visual_sibling_attr == attr
    ]


def test_disabled_installs_nothing():
    model = _build_tiny_lm(enabled=False, ffn=True, norm=True, attention=True)
    assert not getattr(model, "_visual_routed_modules", [])
    # no module gained the routing override
    assert not any(getattr(m, "_visual_expert_installed", False) for m in model.modules())


@pytest.mark.parametrize(
    ("ffn", "norm", "attention", "exp_ffn", "exp_norm", "exp_attn"),
    [
        (True, False, False, 2, 0, 0),  # 2 layers x 1 mlp
        (False, True, False, 0, 4, 0),  # 2 layers x 2 norms
        (False, False, True, 0, 0, 8),  # 2 layers x 4 projections
        (True, True, True, 2, 4, 8),  # all three, independent
    ],
)
def test_experts_toggle_independently(ffn, norm, attention, exp_ffn, exp_norm, exp_attn):
    model = _build_tiny_lm(ffn=ffn, norm=norm, attention=attention)
    assert len(_siblings_of(model, "mlp_visual")) == exp_ffn
    assert len(_siblings_of(model, "norm_visual")) == exp_norm
    assert len(_siblings_of(model, "proj_visual")) == exp_attn
    # exactly the requested sublayers carry the override, nothing else
    assert len(model._visual_routed_modules) == exp_ffn + exp_norm + exp_attn


def test_layers_selects_subset():
    model = _build_tiny_lm(ffn=True, norm=True, attention=True, layers=[1])
    # only layer 1 -> 1 mlp + 2 norms + 4 projections
    assert len(_siblings_of(model, "mlp_visual")) == 1
    assert len(_siblings_of(model, "norm_visual")) == 2
    assert len(_siblings_of(model, "proj_visual")) == 4
    assert not getattr(model.layers[0].mlp, "_visual_expert_installed", False)
    assert getattr(model.layers[1].mlp, "_visual_expert_installed", False)


@pytest.mark.parametrize("which", ["ffn", "norm", "attention"])
def test_init_from_text_is_step0_identity(which):
    """After init_from_text (gate off) the visual sibling == the text sublayer, so
    routing any token through it is a no-op vs the text path."""
    model = _build_tiny_lm(**{which: True})
    for module in model._visual_routed_modules:
        sibling = getattr(module, module._visual_sibling_attr)
        text_sd = {
            k: v
            for k, v in module.state_dict().items()
            if not k.startswith(f"{module._visual_sibling_attr}.")
            and not k.startswith("expert_gate")
        }
        sib_sd = sibling.state_dict()
        assert set(sib_sd) == set(text_sd)
        for k in sib_sd:
            assert torch.equal(sib_sd[k], text_sd[k]), f"{which}:{k} sibling != text after init"
        # forward identity: routing with an all-vision mask == the text path
        in_dim = module.in_features if isinstance(module, nn.Linear) else HID
        x = torch.randn(2, 5, in_dim)
        module._visual_mask = None
        text_out = module(x)
        module._visual_mask = torch.ones(2, 5, 1)
        vis_out = module(x)
        assert torch.allclose(text_out, vis_out, atol=1e-6)


@pytest.mark.parametrize("which", ["ffn", "norm", "attention"])
def test_routes_only_vision_tokens(which):
    """Per-token mask: vision tokens use the (perturbed) visual sibling, text
    tokens are bit-identical to the no-expert text path."""
    model = _build_tiny_lm(**{which: True})
    for module in model._visual_routed_modules:
        sibling = getattr(module, module._visual_sibling_attr)
        with torch.no_grad():  # make the visual sibling differ from text
            for p in sibling.parameters():
                p.add_(1.0)
        in_dim = module.in_features if isinstance(module, nn.Linear) else HID
        x = torch.randn(2, 6, in_dim)
        module._visual_mask = None
        text_out = module(x)
        # token 0 vision, tokens 1.. text
        mask = torch.zeros(2, 6, 1)
        mask[:, 0, :] = 1.0
        module._visual_mask = mask
        mixed = module(x)
        assert torch.allclose(mixed[:, 1:], text_out[:, 1:], atol=1e-6), "text tokens perturbed"
        assert not torch.allclose(mixed[:, 0], text_out[:, 0]), "vision token not routed"
        # mask=None and all-zero mask both collapse to the text path
        module._visual_mask = torch.zeros(2, 6, 1)
        assert torch.allclose(module(x), text_out, atol=1e-6)


def test_gate_near_identity_and_input_dims():
    model = _build_tiny_lm(ffn=True, norm=True, attention=True, gate=True)
    for module in model._visual_routed_modules:
        assert module._expert_gate
        for gname in ("expert_gate_text", "expert_gate_visual"):
            g = getattr(module, gname)
            # gate consumes the module input: in_features for a projection,
            # hidden for FFN/norm. o_proj differs in general (here head_dim*heads).
            expected_in = module.in_features if isinstance(module, nn.Linear) else HID
            assert g.in_features == expected_in and g.out_features == 1
            assert torch.allclose(g.bias, torch.full_like(g.bias, 4.0))
            assert torch.allclose(g.weight, torch.zeros_like(g.weight))


def test_gate_off_is_bit_identical_baseline():
    """Ungated + init_from_text routing must equal the unrouted backbone exactly."""
    plain = M.Qwen3Model(_tiny_qwen3_config())
    expert = _build_tiny_lm(ffn=True, norm=True, attention=True, gate=False)
    expert.load_state_dict(
        {k: v for k, v in plain.state_dict().items()}, strict=False
    )  # share text weights
    init_visual_experts_from_text(expert)  # re-copy text -> visual after the load
    ids = torch.randint(0, 64, (2, 7))
    with torch.no_grad():
        ref = plain(input_ids=ids).last_hidden_state
        # drive an all-vision mask through every expert: still identity (visual==text)
        for m in expert._visual_routed_modules:
            m._visual_mask = torch.ones(2, 7, 1)
        got = expert(input_ids=ids).last_hidden_state
    assert torch.allclose(ref, got, atol=1e-5)


def test_is_visual_expert_param_classification():
    model = _build_tiny_lm(ffn=True, norm=True, attention=True, gate=True)
    names = [n for n, _ in model.named_parameters()]
    flagged = [n for n in names if is_visual_expert_param(n)]
    assert any(".mlp_visual." in n for n in flagged)
    assert any(".norm_visual." in n for n in flagged)
    assert any(".proj_visual." in n for n in flagged)
    assert any("expert_gate" in n for n in flagged)
    # the shared (text) sublayers are NOT flagged
    assert not is_visual_expert_param("layers.0.mlp.gate_proj.weight")
    assert not is_visual_expert_param("layers.0.input_layernorm.weight")
    assert not is_visual_expert_param("layers.0.self_attn.q_proj.weight")


# ---------------------------------------------------------------------------
# model layer: tiny CPU encoder-free VLM (config -> build -> reload parity)
# ---------------------------------------------------------------------------

VISION_DIALS = dict(patch_size=4, pooling_kernel_size=1, max_soft_tokens=16)
PATCH_DIM = (4 * 1) ** 2 * 3  # 48


def _tiny_vlm_config_kwargs(**visual_expert: Any) -> dict:
    base_cfg = AutoConfig.from_pretrained(BASE_LM)
    base_cfg.hidden_size = 32
    base_cfg.intermediate_size = 64
    base_cfg.num_hidden_layers = 2
    base_cfg.layer_types = ["full_attention"] * 2
    base_cfg.num_attention_heads = 4
    base_cfg.num_key_value_heads = 2
    base_cfg.head_dim = 8
    base_cfg.max_position_embeddings = 512
    kwargs = dict(
        hf_name=BASE_LM,
        vision_config={**VISION_DIALS, "hf_name": None, "hidden_size": PATCH_DIM},
        connector_config={
            "name": "raw_patch",
            "type": "raw_patch",
            "mm_embed_dim": 32,
            "mm_posemb_size": VISION_DIALS["max_soft_tokens"],
        },
        audio_config=None,
        **base_cfg.to_dict(),
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
    # flat visual_expert.* fields (normally set by vlm.load_model)
    kwargs["visual_expert"] = visual_expert.get("enabled", False)
    kwargs["visual_expert_ffn"] = visual_expert.get("ffn", True)
    kwargs["visual_expert_norm"] = visual_expert.get("norm", False)
    kwargs["visual_expert_attention"] = visual_expert.get("attention", False)
    kwargs["visual_expert_gate"] = visual_expert.get("gate", False)
    kwargs["visual_expert_layers"] = visual_expert.get("layers", None)
    kwargs["visual_expert_init_from_text"] = visual_expert.get("init_from_text", True)
    return kwargs


def _build_vlm(**visual_expert: Any) -> Any:
    from vlm.models import get_dynamic_vlm

    VLMForCausalLM, VLMConfig = get_dynamic_vlm(BASE_LM)
    config = VLMConfig(**_tiny_vlm_config_kwargs(**visual_expert))
    torch.manual_seed(0)
    model = VLMForCausalLM(config)
    # mirror vlm.load_model's fresh-build init steps for the experts
    if config.visual_expert and config.visual_expert_init_from_text:
        init_visual_experts_from_text(model.model)
    if config.visual_expert and config.visual_expert_gate:
        init_visual_expert_gates(model.model)
    model.eval()
    return model


def test_full_model_installs_three_experts():
    model = _build_vlm(enabled=True, ffn=True, norm=True, attention=True, gate=True)
    routed = model.model._visual_routed_modules
    assert len([m for m in routed if m._visual_sibling_attr == "mlp_visual"]) == 2
    assert len([m for m in routed if m._visual_sibling_attr == "norm_visual"]) == 4
    assert len([m for m in routed if m._visual_sibling_attr == "proj_visual"]) == 8
    assert all(m._expert_gate for m in routed)
    # the expert/gate params are present in the serialized state_dict
    sd = model.state_dict()
    assert any(".mlp_visual." in k for k in sd)
    assert any(".norm_visual." in k for k in sd)
    assert any(".proj_visual." in k for k in sd)
    assert any("expert_gate" in k for k in sd)


def test_text_forward_bit_identical_when_initfromtext_gate_off():
    """A full VLM with all three experts on (init_from_text, gate off) produces
    the same text-only logits as the same backbone with the experts off."""
    expert = _build_vlm(enabled=True, ffn=True, norm=True, attention=True, gate=False)
    baseline = _build_vlm(enabled=False)
    # give the baseline the expert model's shared (non-expert) weights
    shared = {k: v for k, v in expert.state_dict().items() if not is_visual_expert_param(k)}
    missing, unexpected = baseline.load_state_dict(shared, strict=False)
    assert not unexpected, f"unexpected keys loading shared weights: {unexpected[:3]}"
    ids = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        lo_expert = expert(input_ids=ids).logits
        lo_base = baseline(input_ids=ids).logits
    assert torch.allclose(lo_expert, lo_base, atol=1e-5)


def test_generate_save_reload_parity(tmp_path: Path):
    """Experts serialize into config.json + weights; a reloaded checkpoint rebuilds
    the same experts and generates identically (train/infer parity). The greedy
    image generation also drives the attention expert through KV-cache decode."""
    from PIL import Image as PIL_Image

    from vlm.inference.eval import generate_response
    from vlm.models import VLMProcessor
    from vlm.models.image_processing_raw import RawImageProcessor

    model = _build_vlm(enabled=True, ffn=True, norm=True, attention=True, gate=True)
    processor = VLMProcessor(
        image_processor=RawImageProcessor(**VISION_DIALS),
        tokenizer=AutoTokenizer.from_pretrained(BASE_LM),
    )
    image = PIL_Image.new("RGB", (20, 10), (123, 50, 200))
    out1 = generate_response(
        model, processor, query="What is this?", images=image, max_new_tokens=4
    )

    ckpt = tmp_path / "ckpt"
    model.save_pretrained(ckpt)
    processor.save_pretrained(ckpt)

    from vlm.models import get_dynamic_vlm

    VLMForCausalLM, _ = get_dynamic_vlm(BASE_LM)
    reloaded = VLMForCausalLM.from_pretrained(ckpt)
    reloaded.eval()
    # config persisted the toggles, install rebuilt the same experts at load
    assert reloaded.config.visual_expert_norm and reloaded.config.visual_expert_attention
    routed = reloaded.model._visual_routed_modules
    assert len(routed) == 2 + 4 + 8 and all(m._expert_gate for m in routed)
    out2 = generate_response(
        reloaded, processor, query="What is this?", images=image, max_new_tokens=4
    )
    assert out1 == out2
