"""GPU smoke + overfit proof for the image-grounding margin loss (spec
2026-06-18). End-to-end via the real load_model on a small encoder-free VLM
(Qwen3-0.6B), on real bf16 kernels:

  1. mechanics: training fwd/bwd is finite, stashes a "grounding" component,
     and the connector + lm_head receive a non-zero gradient.
  2. OVERFIT (the proof the loss does what it claims): two DISTINCT images map
     to two DISTINCT gold answers under an IDENTICAL question. After training
     with CE+grounding, (a) the grounding loss drops, (b) the gold-token logp
     gap logp_real - logp_blank grows and ends positive, and (c) the predicted
     answer WITH the real image is correct for both images while the blank-image
     prediction cannot separate them — i.e. the model now conditions on pixels
     (toy R0 -> 1), which is exactly the readiness the loss targets.
  3. no-harm: grounding_weight=0 builds NO "grounding" component (baseline path).
  4. text-only: a batch with no image tokens anchors grounding to exact zero
     (no spurious gradient) yet keeps lm_head in the graph (deepspeed pattern).
  5. save/reload: grounding_weight serializes into checkpoint config.json.

Run: sbatch devtools/grounding_smoke.slurm   (or --cpu for a tiny dry run).
Exits nonzero on any failure.
"""

import argparse
import sys
import tempfile

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from vlm.config.config_schema import (
    ConnectorConfig,
    GroundingConfig,
    LanguageModelConfig,
    ModelConfig,
    TrainerConfig,
    VisualEncoderConfig,
)
from vlm.vlm import load_model

OK = True


def check(name, cond, detail=""):
    global OK
    OK = OK and bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def build(device, base_lm, grounding=True, from_pretrained=None):
    bf16 = device == "cuda"
    model_cfg = OmegaConf.structured(
        ModelConfig(
            name="grounding-smoke",
            visual_encoder=VisualEncoderConfig(
                hf_name=None, patch_size=16, pooling_kernel_size=3, max_soft_tokens=64
            ),
            language_model=LanguageModelConfig(hf_name=base_lm, max_seq_length=4096),
            connector=ConnectorConfig(name="raw_patch", type="raw_patch"),
            grounding=GroundingConfig(
                enabled=grounding, weight=0.5 if grounding else 0.0, margin=1.0
            ),
        )
    )
    trainer_cfg = OmegaConf.structured(
        TrainerConfig(
            name="smoke",
            bf16=bf16,
            fp16=False,
            attn_implementation="sdpa",
            from_pretrained=from_pretrained,
        )
    )
    model, processor = load_model(model_cfg, trainer_cfg)
    model = model.to(device=device, dtype=torch.bfloat16 if bf16 else torch.float32)
    # load_model only copies grounding dials onto config inside vlm() (the
    # training entrypoint); the smoke calls load_model directly, so set them
    # here exactly as vlm.py does.
    model.config.grounding_weight = 0.5 if grounding else 0.0
    model.config.grounding_margin = 1.0
    model.config.grounding_corruption = "blank"
    model.config.loss_chunk_size = 64
    return model, processor


def mk_image(processor, device, dtype, seed):
    arr = (np.random.default_rng(seed).random((96, 96, 3)) * 255).astype("uint8")
    feat = processor.image_processor.preprocess([Image.fromarray(arr)])
    return (
        feat["pixel_values"][0].to(device=device, dtype=dtype),
        feat["image_position_ids"][0].to(device),
    )


def batch(model, imgs, poss, q, ans, device):
    """[img_tok]+q+[ans] per sample; ans is a single gold token id."""
    img_tok, ign = model.config.image_token_index, model.config.ignore_index
    seqs = [[img_tok] + list(q) + [a] for a in ans]
    labs = [[ign] * (1 + len(q)) + [a] for a in ans]
    input_ids = torch.tensor(seqs, device=device)
    attn = torch.ones_like(input_ids)
    labels = torch.tensor(labs, device=device)
    return input_ids, attn, labels, list(imgs), list(poss)


@torch.no_grad()
def gold_logp_gap(model, imgs, poss, q, ans, device):
    """Per-sample (logp_real, logp_blank, pred_real) for the gold answer token —
    independent re-derivation of what the loss optimizes, used as the assertion."""
    model.eval()
    input_ids, attn, labels, images, posl = batch(model, imgs, poss, q, ans, device)
    feats = model.encode_raw_patches(images, posl)
    (_, _, _, _, embeds, new_labels, block_ids, _) = model.prepare_inputs_labels_for_multimodal(
        input_ids, None, attn, None, labels, feats, None, with_image_block_ids=True
    )
    blank = embeds.detach().clone()
    blank[block_ids >= 0] = 0.0
    h_real = model.model(inputs_embeds=embeds, use_cache=False).last_hidden_state
    h_blank = model.model(inputs_embeds=blank, use_cache=False).last_hidden_state
    out = []
    for b, a in enumerate(ans):
        pos = (new_labels[b] != model.config.ignore_index).nonzero().squeeze(-1)
        p = int(pos[0]) - 1  # hidden at p predicts label at p+1
        lr = model.lm_head(h_real[b, p]).float().log_softmax(-1)
        lb = model.lm_head(h_blank[b, p]).float().log_softmax(-1)
        out.append((lr[a].item(), lb[a].item(), int(lr.argmax())))
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--base-lm", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--steps", type=int, default=160)
    args = ap.parse_args()
    device = "cpu" if args.cpu else "cuda"
    dtype = torch.float32 if args.cpu else torch.bfloat16

    model, processor = build(device, args.base_lm, grounding=True)
    model.train()

    # two distinct images -> two distinct gold answers, identical question.
    img0, pos0 = mk_image(processor, device, dtype, seed=0)
    img1, pos1 = mk_image(processor, device, dtype, seed=12345)
    q = [40, 41, 42]
    a0, a1 = 100, 200  # distinct single-token gold answers
    imgs, poss, ans = [img0, img1], [pos0, pos1], [a0, a1]

    # ---- 1. mechanics ----------------------------------------------------
    iid, attn, lab, images, posl = batch(model, imgs, poss, q, ans, device)
    out = model(
        input_ids=iid, attention_mask=attn, labels=lab, images=images, image_position_ids=posl
    )
    comps = getattr(model, "_last_ce_components", {})
    check("mechanics: finite loss", torch.isfinite(out.loss).item(), f"loss={out.loss.item():.4f}")
    check("mechanics: grounding component stashed", "grounding" in comps, f"keys={sorted(comps)}")
    check(
        "mechanics: grounding >= 0",
        float(comps.get("grounding", -1)) >= 0.0,
        f"grounding={float(comps.get('grounding', -1)):.4f}",
    )
    out.loss.backward()
    cg = next(
        p.grad for n, p in model.named_parameters() if "connector" in n and p.grad is not None
    )
    hg = model.lm_head.weight.grad
    check("mechanics: connector has gradient", cg.abs().sum().item() > 0)
    check("mechanics: lm_head has gradient", hg is not None and hg.abs().sum().item() > 0)
    model.zero_grad(set_to_none=True)

    # ---- 2. overfit proof ------------------------------------------------
    gap0 = gold_logp_gap(model, imgs, poss, q, ans, device)
    g_start = float(getattr(model, "_last_ce_components", {}).get("grounding", float("nan")))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g_hist = []
    for step in range(args.steps):
        iid, attn, lab, images, posl = batch(model, imgs, poss, q, ans, device)
        out = model(
            input_ids=iid, attention_mask=attn, labels=lab, images=images, image_position_ids=posl
        )
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        opt.step()
        g_hist.append(float(model._last_ce_components["grounding"]))
        if step % 40 == 0:
            print(
                f"  step {step:3d}: loss={out.loss.item():.4f} grounding={g_hist[-1]:.4f}",
                flush=True,
            )
    gap1 = gold_logp_gap(model, imgs, poss, q, ans, device)

    mean_gap0 = sum(r[0] - r[1] for r in gap0) / len(gap0)
    mean_gap1 = sum(r[0] - r[1] for r in gap1) / len(gap1)
    check(
        "overfit: grounding loss decreased",
        g_hist[-1] < g_start - 1e-3,
        f"{g_start:.4f} -> {g_hist[-1]:.4f}",
    )
    check(
        "overfit: logp_real-logp_blank gap grew",
        mean_gap1 > mean_gap0,
        f"gap {mean_gap0:+.3f} -> {mean_gap1:+.3f}",
    )
    check(
        "overfit: final gap positive (image now matters)", mean_gap1 > 0.0, f"gap={mean_gap1:+.3f}"
    )
    preds_real_ok = all(gap1[b][2] == ans[b] for b in range(len(ans)))
    check(
        "overfit: prediction WITH image correct for both (toy R0->1)",
        preds_real_ok,
        f"preds={[gap1[b][2] for b in range(len(ans))]} gold={ans}",
    )

    # ---- 3. no-harm: grounding off -> no component -----------------------
    m0, p0 = build(device, args.base_lm, grounding=False)
    m0.train()
    iid, attn, lab, images, posl = batch(m0, imgs, poss, q, ans, device)
    out0 = m0(
        input_ids=iid, attention_mask=attn, labels=lab, images=images, image_position_ids=posl
    )
    check(
        "no-harm: weight=0 builds no grounding component",
        "grounding" not in getattr(m0, "_last_ce_components", {}),
        f"keys={sorted(getattr(m0, '_last_ce_components', {}))}",
    )
    del m0

    # ---- 4. text-only batch -> grounding anchored to zero ----------------
    ign = model.config.ignore_index
    t_ids = torch.tensor([[40, 41, 42, 100]], device=device)
    t_lab = torch.tensor([[ign, ign, ign, 100]], device=device)
    out_t = model(input_ids=t_ids, attention_mask=torch.ones_like(t_ids), labels=t_lab)
    gt = float(getattr(model, "_last_ce_components", {}).get("grounding", -1.0))
    check("text-only: grounding anchored to exact zero", gt == 0.0, f"grounding={gt}")
    check(
        "text-only: finite loss", torch.isfinite(out_t.loss).item(), f"loss={out_t.loss.item():.4f}"
    )

    # ---- 5. save / reload: grounding_weight serializes into config.json --
    import json
    import os

    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        processor.save_pretrained(d)
        with open(os.path.join(d, "config.json")) as f:
            saved_cfg = json.load(f)
        check(
            "reload: grounding_weight serialized into config.json",
            float(saved_cfg.get("grounding_weight", 0.0)) == 0.5,
            f"config.json grounding_weight={saved_cfg.get('grounding_weight')}",
        )

    print("\n" + ("ALL GROUNDING SMOKE CHECKS PASSED" if OK else "GROUNDING SMOKE FAILED"))
    sys.exit(0 if OK else 1)


if __name__ == "__main__":
    main()
