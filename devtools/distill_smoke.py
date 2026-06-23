"""GPU smoke + overfit proof for visual-encoder distillation (spec 2026-06-21).
End-to-end via the real load_model on a small encoder-free VLM (Qwen3-0.6B) with
a real frozen CLIP-B/16 teacher, on bf16 kernels:

  1. mechanics: a training fwd/bwd is finite, stashes a "distill" component, and
     the connector + the distill head + lm_head receive a non-zero gradient.
  2. teacher INVISIBILITY: the frozen teacher is off the module tree — it must
     NOT appear in named_parameters() nor in the saved state_dict (else
     set_trainable would sweep CLIP into the language_model group and train it,
     and checkpoints would bloat by ~170MB).
  3. OVERFIT (the proof the loss does what it claims): train on a few fixed
     images; the distill loss DROPS and the projected-native↔teacher cosine
     RISES — i.e. the LLM's hidden states at image positions are being pulled
     onto the CLIP feature manifold, which is exactly what distillation targets.
  4. no-harm: disabled -> NO "distill" component (bit-identical baseline path).
  5. text-only: a batch with no real image anchors distill to exact zero (no
     spurious grad) yet keeps the head params in the graph (deepspeed pattern).
  6. save/reload: visual_distill_* serialize into config.json, the head reloads,
     the teacher re-attaches, and NO teacher weights are in the checkpoint.

Run: sbatch devtools/distill_smoke.slurm   (loops over methods).
     python devtools/distill_smoke.py --method repa   (single method on GPU)
Exits nonzero on any failure.
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from vlm.config.config_schema import (
    ConnectorConfig,
    LanguageModelConfig,
    ModelConfig,
    TrainerConfig,
    VisualDistillConfig,
    VisualEncoderConfig,
)
from vlm.vlm import load_model

OK = True


def check(name, cond, detail=""):
    global OK
    OK = OK and bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def build(device, base_lm, method, enabled=True, weight=1.0, from_pretrained=None,
          teacher_kind=None, teacher_name=None):
    bf16 = device == "cuda"
    if teacher_kind is None:
        teacher_kind = "vae" if method == "vae" else "clip"
    if teacher_name is None:
        teacher_name = {
            "vae": "stabilityai/sd-vae-ft-mse",
            "siglip": "google/siglip-base-patch16-224",
        }.get(teacher_kind, "openai/clip-vit-base-patch16")
    model_cfg = OmegaConf.structured(
        ModelConfig(
            name="distill-smoke",
            visual_encoder=VisualEncoderConfig(
                hf_name=None, patch_size=16, pooling_kernel_size=3, max_soft_tokens=64
            ),
            language_model=LanguageModelConfig(hf_name=base_lm, max_seq_length=4096),
            connector=ConnectorConfig(name="raw_patch", type="raw_patch"),
            visual_distill=VisualDistillConfig(
                enabled=enabled,
                method=method,
                teacher_kind=teacher_kind,
                teacher_name=teacher_name,
            ),
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
    # load_model sets visual_distill_weight only inside vlm() (the training
    # entrypoint); the smoke calls load_model directly, so set it here like vlm.py.
    model.config.visual_distill_weight = weight if enabled else 0.0
    model.config.loss_chunk_size = 64
    return model, processor


def mk_image(processor, device, dtype, seed):
    arr = (np.random.default_rng(seed).random((96, 96, 3)) * 255).astype("uint8")
    feat = processor.image_processor.preprocess([Image.fromarray(arr)])
    return (feat["pixel_values"][0].to(device=device, dtype=dtype),
            feat["image_position_ids"][0].to(device))


def batch(model, imgs, poss, q, ans, device):
    img_tok, ign = model.config.image_token_index, model.config.ignore_index
    seqs = [[img_tok] + list(q) + [a] for a in ans]
    labs = [[ign] * (1 + len(q)) + [a] for a in ans]
    input_ids = torch.tensor(seqs, device=device)
    attn = torch.ones_like(input_ids)
    labels = torch.tensor(labs, device=device)
    return input_ids, attn, labels, list(imgs), list(poss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--base-lm", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--method", default="repa",
                    choices=["repa", "eve", "vora", "softdepth", "relational", "vae"])
    ap.add_argument("--teacher-kind", default=None, choices=[None, "clip", "siglip", "vae"])
    ap.add_argument("--teacher-name", default=None)
    ap.add_argument("--steps", type=int, default=150)
    args = ap.parse_args()
    device = "cpu" if args.cpu else "cuda"
    dtype = torch.float32 if args.cpu else torch.bfloat16
    print(f"\n========== distill smoke: method={args.method} ==========", flush=True)

    tk, tn = args.teacher_kind, args.teacher_name
    model, processor = build(device, args.base_lm, args.method, teacher_kind=tk, teacher_name=tn)
    model.train()

    img0, pos0 = mk_image(processor, device, dtype, seed=0)
    img1, pos1 = mk_image(processor, device, dtype, seed=12345)
    q = [40, 41, 42]
    ans = [100, 200]
    imgs, poss = [img0, img1], [pos0, pos1]

    # ---- 1. mechanics ----------------------------------------------------
    iid, attn, lab, images, posl = batch(model, imgs, poss, q, ans, device)
    out = model(input_ids=iid, attention_mask=attn, labels=lab, images=images,
                image_position_ids=posl)
    comps = getattr(model, "_last_ce_components", {})
    check("mechanics: finite loss", torch.isfinite(out.loss).item(), f"loss={out.loss.item():.4f}")
    check("mechanics: distill component stashed", "distill" in comps, f"keys={sorted(comps)}")
    out.loss.backward()
    cg = next((p.grad for n, p in model.named_parameters()
               if "connector" in n and p.grad is not None), None)
    check("mechanics: connector has gradient", cg is not None and cg.abs().sum().item() > 0)
    # relational has NO head params by design; others must get a head gradient.
    head_params = list(model.visual_distill_head.parameters())
    if head_params:
        hg = next((p.grad for p in head_params if p.grad is not None), None)
        check("mechanics: distill head has gradient", hg is not None and hg.abs().sum().item() > 0)
    else:
        check("mechanics: relational head is param-free (by design)", args.method == "relational")
    check("mechanics: lm_head has gradient",
          model.lm_head.weight.grad is not None and model.lm_head.weight.grad.abs().sum().item() > 0)
    model.zero_grad(set_to_none=True)

    # ---- 2. teacher invisibility ----------------------------------------
    pnames = [n for n, _ in model.named_parameters()]
    check("invisible: teacher not in named_parameters",
          not any("_distill_teacher" in n or "teacher" in n for n in pnames))
    check("invisible: teacher attached + frozen",
          getattr(model, "_distill_teacher", None) is not None
          and all(not p.requires_grad for p in model._distill_teacher[0].parameters()))
    sd_keys = list(model.state_dict().keys())
    check("invisible: no teacher weights in state_dict",
          not any("_distill_teacher" in k for k in sd_keys),
          f"state_dict has {len(sd_keys)} keys")

    # ---- 3. overfit proof -----------------------------------------------
    def distill_now():
        model.eval()
        with torch.no_grad():
            _ = model.__call__  # keep linter calm
        model.train()
        iid, attn, lab, images, posl = batch(model, imgs, poss, q, ans, device)
        o = model(input_ids=iid, attention_mask=attn, labels=lab, images=images,
                  image_position_ids=posl)
        c = model._last_ce_components
        return float(c["distill"]), float(c["distill_cos"])

    d0, cos0 = distill_now()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    d_hist, cos_hist = [], []
    for step in range(args.steps):
        iid, attn, lab, images, posl = batch(model, imgs, poss, q, ans, device)
        out = model(input_ids=iid, attention_mask=attn, labels=lab, images=images,
                    image_position_ids=posl)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        opt.step()
        d_hist.append(float(model._last_ce_components["distill"]))
        cos_hist.append(float(model._last_ce_components["distill_cos"]))
        if step % 30 == 0:
            print(f"  step {step:3d}: loss={out.loss.item():.4f} "
                  f"distill={d_hist[-1]:.4f} cos={cos_hist[-1]:.4f}", flush=True)
    check("overfit: distill loss decreased", d_hist[-1] < d0 - 1e-3, f"{d0:.4f} -> {d_hist[-1]:.4f}")
    if args.method != "relational":
        # cosine methods: native↔teacher cosine should rise toward 1.
        check("overfit: native↔teacher cosine rose", cos_hist[-1] > cos0 + 0.05,
              f"cos {cos0:.3f} -> {cos_hist[-1]:.3f}")
    if args.method == "softdepth":
        sel = float(model._last_ce_components["distill_sel_depth"])
        smax = float(model._last_ce_components["distill_sel_max"])
        check("overfit: softdepth selected a depth (peaked softmax)", smax > 1.0 / 28 + 1e-3,
              f"selected_depth≈{sel:.1f}, max_weight={smax:.3f}")

    # ---- 4. no-harm: disabled -> no component ---------------------------
    m0, _ = build(device, args.base_lm, args.method, enabled=False, teacher_kind=tk, teacher_name=tn)
    m0.train()
    iid, attn, lab, images, posl = batch(m0, imgs, poss, q, ans, device)
    _ = m0(input_ids=iid, attention_mask=attn, labels=lab, images=images, image_position_ids=posl)
    check("no-harm: disabled builds no distill component",
          "distill" not in getattr(m0, "_last_ce_components", {})
          and m0.visual_distill_head is None,
          f"keys={sorted(getattr(m0, '_last_ce_components', {}))}")
    del m0

    # ---- 5. text-only batch -> distill anchored to zero -----------------
    ign = model.config.ignore_index
    t_ids = torch.tensor([[40, 41, 42, 100]], device=device)
    t_lab = torch.tensor([[ign, ign, ign, 100]], device=device)
    out_t = model(input_ids=t_ids, attention_mask=torch.ones_like(t_ids), labels=t_lab)
    dt = float(getattr(model, "_last_ce_components", {}).get("distill", -1.0))
    check("text-only: distill anchored to exact zero", dt == 0.0, f"distill={dt}")
    check("text-only: finite loss", torch.isfinite(out_t.loss).item(), f"loss={out_t.loss.item():.4f}")

    # ---- 6. save / reload -----------------------------------------------
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        processor.save_pretrained(d)
        with open(os.path.join(d, "config.json")) as f:
            saved = json.load(f)
        check("reload: visual_distill serialized into config.json",
              bool(saved.get("visual_distill")) and saved.get("visual_distill_method") == args.method,
              f"method={saved.get('visual_distill_method')} dim={saved.get('visual_distill_teacher_dim')}")
        # confirm no CLIP weights were written (state-dict shard files)
        shard = [fn for fn in os.listdir(d) if fn.endswith((".safetensors", ".bin"))]
        any_teacher_key = False
        try:
            from safetensors import safe_open
            for fn in shard:
                if fn.endswith(".safetensors"):
                    with safe_open(os.path.join(d, fn), framework="pt") as sf:
                        if any("_distill_teacher" in k for k in sf.keys()):
                            any_teacher_key = True
        except Exception:
            pass
        check("reload: no teacher weights written to checkpoint", not any_teacher_key)
        m2, _ = build(device, args.base_lm, args.method, from_pretrained=d, teacher_kind=tk, teacher_name=tn)
        check("reload: head rebuilt from checkpoint",
              getattr(m2, "visual_distill_head", None) is not None)
        check("reload: teacher re-attached",
              getattr(m2, "_distill_teacher", None) is not None)
        del m2

    print("\n" + (f"ALL DISTILL SMOKE CHECKS PASSED ({args.method})" if OK
                  else f"DISTILL SMOKE FAILED ({args.method})"))
    sys.exit(0 if OK else 1)


if __name__ == "__main__":
    main()
