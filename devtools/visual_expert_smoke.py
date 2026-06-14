"""GPU smoke for the visual-FFN expert arm (spec 2026-06-14).

End-to-end on a small encoder-free VLM (Qwen3-0.6B, random raw-patch connector)
built through the REAL load_model, exercising every integration point that the
CPU unit test (devtools/test_visual_expert.py) cannot:

  1. fresh-build install + init-from-text: experts attached, visual==text at t0.
  2. training chunked-CE forward+backward: mlp_visual receives a NON-zero
     gradient on a batch with image tokens (the capacity actually trains).
  3. generate(): the prefill routes image tokens through mlp_visual (the fixed
     one-shot _ve_gen_mask) — verified with a call-counting hook; decode steps
     do not call the expert.
  4. save/reload: reloaded model has the experts, and they equal the SAVED
     (perturbed) weights — i.e. reload loads checkpoint experts and does NOT
     re-copy from the text FFN (no clobber of trained experts).

Run: sbatch devtools/visual_expert_smoke.slurm   (or --cpu for a tiny dry run).
Exits nonzero on any failure.
"""

import argparse
import sys
import tempfile

import torch
from omegaconf import OmegaConf

from vlm.config.config_schema import (
    ConnectorConfig,
    LanguageModelConfig,
    ModelConfig,
    TrainerConfig,
    VisualEncoderConfig,
    VisualExpertConfig,
)
from vlm.vlm import load_model

# reuse the encoder-free batch builders from the xmodal smoke
from xmodal_smoke import make_image, splice  # noqa: E402

OK = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global OK
    OK = OK and cond
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def build(device: str, base_lm: str, from_pretrained: str | None = None):
    bf16 = device == "cuda"
    model_cfg = OmegaConf.structured(
        ModelConfig(
            name="ve-smoke",
            visual_encoder=VisualEncoderConfig(
                hf_name=None, patch_size=16, pooling_kernel_size=3, max_soft_tokens=64
            ),
            language_model=LanguageModelConfig(hf_name=base_lm, max_seq_length=4096),
            connector=ConnectorConfig(name="raw_patch", type="raw_patch"),
            visual_expert=VisualExpertConfig(enabled=True, layers=None, init_from_text=True),
        )
    )
    trainer_cfg = OmegaConf.structured(
        TrainerConfig(
            name="smoke", bf16=bf16, fp16=False, attn_implementation="sdpa",
            from_pretrained=from_pretrained,
        )
    )
    model, processor = load_model(model_cfg, trainer_cfg)
    model = model.to(device=device, dtype=torch.bfloat16 if bf16 else torch.float32)
    return model, processor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--base-lm", default="Qwen/Qwen3-0.6B")
    args = ap.parse_args()
    device = "cpu" if args.cpu else "cuda"
    dtype = torch.float32 if args.cpu else torch.bfloat16

    model, processor = build(device, args.base_lm)
    experts = model.model._visual_expert_mlps
    n_layers = len(model.model.layers)
    check("install: experts on all layers", len(experts) == n_layers, f"{len(experts)}/{n_layers}")

    # (1) init-from-text: visual == text at t0
    same = all(
        torch.equal(m.gate_proj.weight, m.mlp_visual.gate_proj.weight)
        and torch.equal(m.up_proj.weight, m.mlp_visual.up_proj.weight)
        and torch.equal(m.down_proj.weight, m.mlp_visual.down_proj.weight)
        for m in experts
    )
    check("init-from-text: visual FFN == text FFN at t0", same)

    # (2) training chunked-CE fwd/bwd -> expert gets a NON-zero gradient
    model.config.loss_chunk_size = 64  # exercise the real training path
    model.train()
    input_ids, attn, labels, images, poss = splice(
        model, processor, device, dtype, questions=[[5, 6, 7], [8, 9]]
    )
    out = model(
        input_ids=input_ids, attention_mask=attn, labels=labels,
        images=images, image_position_ids=poss,
    )
    check("train: finite loss", torch.isfinite(out.loss).item(), f"loss={out.loss.item():.4f}")
    out.loss.backward()
    g = experts[0].mlp_visual.gate_proj.weight.grad
    check("train: expert has gradient", g is not None)
    check(
        "train: expert gradient is non-zero (image tokens trained it)",
        g is not None and g.abs().sum().item() > 0.0,
        f"|grad|={g.abs().sum().item():.3e}" if g is not None else "",
    )
    model.zero_grad(set_to_none=True)

    # (3) generate(): prefill must route image tokens through mlp_visual
    model.eval()
    calls = {"n": 0}
    hooks = [m.mlp_visual.register_forward_hook(lambda *_: calls.__setitem__("n", calls["n"] + 1))
             for m in experts]
    prompt_ids = torch.tensor([[model.config.image_token_index, 5, 6, 7]], device=device)
    pix, pos = make_image(processor, device, dtype)
    with torch.no_grad():
        gen = model.generate(
            inputs=prompt_ids, images=[pix], image_position_ids=[pos], max_new_tokens=3, do_sample=False,
        )
    for h in hooks:
        h.remove()
    check(
        "generate: experts run at prefill (one call per layer)",
        calls["n"] == n_layers,
        f"{calls['n']} expert calls vs {n_layers} layers",
    )
    # generate(inputs_embeds=...) returns ONLY the newly generated ids (there are
    # no input_ids to prepend), so the prompt length is not included.
    check("generate: produced new tokens", gen.shape[1] >= 1, f"{gen.shape[1]} new tokens")

    # (4) save/reload: perturb experts, save, reload -> reloaded == perturbed
    #     (proves reload loads checkpoint experts, does NOT re-copy from text)
    with torch.no_grad():
        for m in experts:
            m.mlp_visual.gate_proj.weight.add_(0.123)
    saved = experts[0].mlp_visual.gate_proj.weight.detach().clone()
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        processor.save_pretrained(d)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        rl, _ = build(device, args.base_lm, from_pretrained=d)
        check("reload: config carries visual_expert", bool(getattr(rl.config, "visual_expert", False)))
        rexperts = rl.model._visual_expert_mlps
        check("reload: experts rebuilt", len(rexperts) == n_layers)
        rl_w = rexperts[0].mlp_visual.gate_proj.weight.detach().to(saved.device)
        check(
            "reload: expert == SAVED (perturbed) weights, not re-copied from text",
            torch.equal(rl_w, saved),
        )
        # and it must NOT equal the text FFN (which we did not perturb)
        check(
            "reload: expert != text FFN (no clobber)",
            not torch.equal(rl_w, rexperts[0].gate_proj.weight.detach().to(saved.device)),
        )

    print("\n" + ("ALL VISUAL-EXPERT SMOKE CHECKS PASSED" if OK else "SMOKE FAILED"))
    sys.exit(0 if OK else 1)


if __name__ == "__main__":
    main()
