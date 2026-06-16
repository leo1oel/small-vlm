"""GPU smoke for the visual-prefix arm (spec 2026-06-14, early-capacity lever).

End-to-end via the real load_model on a small encoder-free VLM (Qwen3-0.6B):
  1. fresh-build: the prefix stack is attached (depth layers) with real weights.
  2. training chunked-CE fwd/bwd: prefix params get a NON-zero gradient (the
     internal encoder actually trains).
  3. generate(): runs (the prefix is applied in encode_raw_patches at prefill).
  4. save/reload: reloaded model rebuilds the prefix and loads its (perturbed)
     weights — config-driven structure + checkpoint weights round-trip.

Run: sbatch devtools/visual_prefix_smoke.slurm   (or --cpu for a tiny dry run).
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
    VisualPrefixConfig,
)
from vlm.vlm import load_model

from xmodal_smoke import make_image, splice  # noqa: E402

OK = True
DEPTH = 4


def check(name, cond, detail=""):
    global OK
    OK = OK and cond
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def build(device, base_lm, from_pretrained=None):
    bf16 = device == "cuda"
    model_cfg = OmegaConf.structured(
        ModelConfig(
            name="prefix-smoke",
            visual_encoder=VisualEncoderConfig(
                hf_name=None, patch_size=16, pooling_kernel_size=3, max_soft_tokens=64
            ),
            language_model=LanguageModelConfig(hf_name=base_lm, max_seq_length=4096),
            connector=ConnectorConfig(name="raw_patch", type="raw_patch"),
            visual_prefix=VisualPrefixConfig(enabled=True, depth=DEPTH),
        )
    )
    trainer_cfg = OmegaConf.structured(
        TrainerConfig(name="smoke", bf16=bf16, fp16=False, attn_implementation="sdpa",
                      from_pretrained=from_pretrained)
    )
    model, processor = load_model(model_cfg, trainer_cfg)
    return model.to(device=device, dtype=torch.bfloat16 if bf16 else torch.float32), processor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--base-lm", default="Qwen/Qwen3-0.6B")
    args = ap.parse_args()
    device = "cpu" if args.cpu else "cuda"
    dtype = torch.float32 if args.cpu else torch.bfloat16

    model, processor = build(device, args.base_lm)
    pf = getattr(model.model, "visual_prefix", None)
    check("build: prefix attached", pf is not None and len(pf.layers) == DEPTH,
          f"{None if pf is None else len(pf.layers)} layers")

    # training fwd/bwd -> prefix gets a non-zero gradient
    model.config.loss_chunk_size = 64
    model.train()
    input_ids, attn, labels, images, poss = splice(
        model, processor, device, dtype, questions=[[5, 6, 7], [8, 9]]
    )
    out = model(input_ids=input_ids, attention_mask=attn, labels=labels,
                images=images, image_position_ids=poss)
    check("train: finite loss", torch.isfinite(out.loss).item(), f"loss={out.loss.item():.4f}")
    out.loss.backward()
    g = pf.layers[0].q_proj.weight.grad
    check("train: prefix has non-zero gradient", g is not None and g.abs().sum().item() > 0,
          f"|grad|={g.abs().sum().item():.3e}" if g is not None else "None")
    model.zero_grad(set_to_none=True)

    # generate
    model.eval()
    prompt = torch.tensor([[model.config.image_token_index, 5, 6, 7]], device=device)
    pix, pos = make_image(processor, device, dtype)
    with torch.no_grad():
        gen = model.generate(inputs=prompt, images=[pix], image_position_ids=[pos],
                             max_new_tokens=3, do_sample=False)
    check("generate: produced new tokens", gen.shape[1] >= 1, f"{gen.shape[1]} new tokens")

    # save / reload
    with torch.no_grad():
        pf.layers[0].q_proj.weight.add_(0.123)
    saved = pf.layers[0].q_proj.weight.detach().clone()
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d); processor.save_pretrained(d)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        rl, _ = build(device, args.base_lm, from_pretrained=d)
        rpf = getattr(rl.model, "visual_prefix", None)
        check("reload: config carries visual_prefix", bool(getattr(rl.config, "visual_prefix", False)))
        check("reload: prefix rebuilt", rpf is not None and len(rpf.layers) == DEPTH)
        rw = rpf.layers[0].q_proj.weight.detach().to(saved.device)
        check("reload: prefix == saved (perturbed) weights", torch.equal(rw, saved))

    print("\n" + ("ALL VISUAL-PREFIX SMOKE CHECKS PASSED" if OK else "SMOKE FAILED"))
    sys.exit(0 if OK else 1)


if __name__ == "__main__":
    main()
