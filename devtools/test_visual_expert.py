"""CPU unit test for the visual-FFN expert routing (spec 2026-06-14).

Tests the custom code in modeling_vlm.py (install_visual_experts,
init_visual_experts_from_text, _routed_mlp_forward) on a REAL tiny Qwen3Model —
no download, no GPU. Validates: structure attach + param names, init-from-text
copy, routing math (mask 0 / 1 / blend), gradient flow to the expert under an
all-zero mask (DeepSpeed ZeRO uneven-participation guard), and that an
un-installed baseline mlp is byte-identical.

Run:  .venv/bin/python devtools/test_visual_expert.py
"""

import copy

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3Model

from vlm.models.modeling_vlm import (
    init_visual_experts_from_text,
    install_visual_experts,
)


def tiny_model():
    cfg = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=100,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return Qwen3Model(cfg), cfg


def main():
    torch.manual_seed(0)
    model, cfg = tiny_model()

    # Reference: the original text FFN output for layer 0, BEFORE install.
    x = torch.randn(2, 5, cfg.hidden_size)
    ref_text_out = model.layers[0].mlp(x.clone())

    # --- install ---
    cfg.visual_expert = True
    cfg.visual_expert_layers = None
    install_visual_experts(model, cfg)

    names = dict(model.named_parameters())
    assert "layers.0.mlp.gate_proj.weight" in names, "text FFN param name changed!"
    assert "layers.0.mlp.mlp_visual.gate_proj.weight" in names, "expert param missing!"
    assert len(model._visual_expert_mlps) == 2, "experts not installed on all layers"
    print("[ok] structure attached; text FFN param names preserved")

    # text path still byte-identical with mask=None (no routing requested)
    mlp0 = model.layers[0].mlp
    mlp0._visual_mask = None
    out_none = mlp0(x.clone())
    assert torch.allclose(out_none, ref_text_out, atol=0), "mask=None changed text output"
    print("[ok] mask=None reproduces the original text FFN exactly")

    # --- init-from-text copy ---
    # before copy: expert is randomly initialized -> differs from text
    g_text = mlp0.gate_proj.weight
    g_vis = mlp0.mlp_visual.gate_proj.weight
    assert not torch.allclose(g_text, g_vis), "expert unexpectedly equals text pre-copy"
    init_visual_experts_from_text(model)
    assert torch.allclose(g_text, g_vis), "init-from-text did not copy weights"
    for k in ("gate_proj", "up_proj", "down_proj"):
        a = getattr(mlp0, k).weight
        b = getattr(mlp0.mlp_visual, k).weight
        assert torch.allclose(a, b), f"{k} not copied"
    print("[ok] init-from-text copies gate/up/down into the expert")

    # after copy, expert == text, so ANY mask reproduces the text output
    mlp0._visual_mask = torch.ones(2, 5, 1)
    out_ones = mlp0(x.clone())
    assert torch.allclose(out_ones, ref_text_out, atol=1e-6), "post-copy expert != text"
    print("[ok] post-copy expert reproduces text output (mask=1)")

    # --- routing math: perturb the expert so text != visual, check the blend ---
    with torch.no_grad():
        mlp0.mlp_visual.gate_proj.weight.add_(1.0)
    mlp0._visual_mask = None
    pure_text = mlp0(x.clone())
    mlp0._visual_mask = torch.ones(2, 5, 1)
    pure_vis = mlp0(x.clone())
    assert not torch.allclose(pure_text, pure_vis), "perturb failed; text==visual"
    # per-token mask: token 0 -> text, all others -> visual
    m = torch.ones(2, 5, 1)
    m[:, 0, :] = 0.0
    mlp0._visual_mask = m
    blended = mlp0(x.clone())
    assert torch.allclose(blended[:, 0], pure_text[:, 0], atol=1e-6), "text-routed token wrong"
    assert torch.allclose(blended[:, 1:], pure_vis[:, 1:], atol=1e-6), "visual-routed token wrong"
    print("[ok] per-token blend routes text vs visual correctly")

    # --- gradient flow: all-zero mask must STILL give the expert a gradient ---
    model.zero_grad(set_to_none=True)
    mlp0._visual_mask = torch.zeros(2, 5, 1)  # all text
    out = mlp0(x.clone())
    out.sum().backward()
    gv = mlp0.mlp_visual.gate_proj.weight.grad
    assert gv is not None, "expert got NO gradient under all-zero mask (ZeRO would hang)"
    assert torch.allclose(gv, torch.zeros_like(gv)), "all-zero mask should give zero grad"
    print("[ok] all-zero mask keeps the expert in the graph with a (zero) gradient")

    # --- baseline untouched: a fresh model with no install is byte-identical ---
    base, bcfg = tiny_model()
    base_out = base.layers[0].mlp(x.clone())
    ref2, _ = tiny_model()
    assert torch.allclose(base_out, ref2.layers[0].mlp(x.clone()), atol=0), "baseline drift"
    assert not hasattr(base.layers[0].mlp, "mlp_visual"), "baseline got an expert!"
    print("[ok] un-installed baseline mlp is unchanged")

    print("\nALL VISUAL-EXPERT UNIT CHECKS PASSED")


if __name__ == "__main__":
    main()
