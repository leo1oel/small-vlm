"""Phase-0 port-correctness smoke for the BREEN port (arXiv:2503.12446, spec
2026-06-24). End-to-end via the real load_model on a small encoder-free VLM
(Qwen3-0.6B) with the real frozen CLIP-L/14-336 teacher, the visual FFN expert
(+ gate), and the 100 learnable queries. Verifies the four acceptance checks:

  (a) the distill_cos component RISES from ~0 over a short overfit (the queries
      align to the dual-pooled CLIP grid — the breen mechanism works);
  (b) query positions are LABEL-MASKED (CE ignores them) AND routed to the
      visual FFN expert (query_block_ids tag image+query for the mask);
  (c) the teacher is ABSENT from named_parameters / state_dict, and a checkpoint
      loads + forwards with NO CLIP teacher attached (the inference path);
  (d) in a frozen-LLM config the query / norm_layer / mlp_visual / gate params
      DO receive gradients (the build-item-8 param-grouping trap), while the LM
      trunk stays frozen.

Plus a light data-path check that inject_query_placeholders + the multimodal
tokenizer emit the query sentinel (build item 7).

Run: sbatch devtools/breen_smoke.slurm   (or --cpu for a tiny dry run).
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
    LanguageModelConfig,
    LearnableQueryConfig,
    ModelConfig,
    TrainerConfig,
    UnfreezeConfig,
    VisualDistillConfig,
    VisualEncoderConfig,
    VisualExpertConfig,
)
from vlm.data.data_arguments import DataArguments
from vlm.data.dataset import inject_query_placeholders, tokenizer_multimodal_token
from vlm.models import get_dynamic_vlm
from vlm.train.set_trainable import set_trainable_params
from vlm.vlm import load_model

OK = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global OK
    OK = OK and bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def build(device, base_lm, enabled=True, gate=True, from_pretrained=None, weight=1.0):
    bf16 = device == "cuda"
    model_cfg = OmegaConf.structured(
        ModelConfig(
            name="breen-smoke",
            visual_encoder=VisualEncoderConfig(
                hf_name=None, patch_size=16, pooling_kernel_size=3, max_soft_tokens=64
            ),
            language_model=LanguageModelConfig(hf_name=base_lm, max_seq_length=4096),
            connector=ConnectorConfig(name="raw_patch", type="raw_patch"),
            visual_expert=VisualExpertConfig(
                enabled=enabled, layers=None, init_from_text=True, gate=gate
            ),
            learnable_query=LearnableQueryConfig(
                enabled=enabled, num_query=100, placement="after_image"
            ),
            visual_distill=VisualDistillConfig(
                enabled=enabled,
                method="breen",
                teacher_kind="clip",
                teacher_name="openai/clip-vit-large-patch14-336",
                teacher_out_size=336,
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
    # load_model only sets visual_distill_weight inside vlm() (training entry);
    # the smoke calls load_model directly, so set it here like vlm.py does.
    model.config.visual_distill_weight = weight if enabled else 0.0
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
    img_tok = model.config.image_token_index
    q_tok = model.config.query_token_index
    ign = model.config.ignore_index
    # [<image>, <query>] + question + answer (one <query> per image; the splice
    # expands the single sentinel into the 100-row query block).
    seqs = [[img_tok, q_tok] + list(q) + [a] for a in ans]
    labs = [[ign, ign] + [ign] * len(q) + [a] for a in ans]
    input_ids = torch.tensor(seqs, device=device)
    attn = torch.ones_like(input_ids)
    labels = torch.tensor(labs, device=device)
    return input_ids, attn, labels, list(imgs), list(poss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--base-lm", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--steps", type=int, default=150)
    args = ap.parse_args()
    device = "cpu" if args.cpu else "cuda"
    dtype = torch.float32 if args.cpu else torch.bfloat16
    print("\n========== BREEN port smoke ==========", flush=True)

    model, processor = build(device, args.base_lm)
    nq = model.config.learnable_query_num_query
    check(
        "build: learnable_query Parameter present",
        getattr(model, "learnable_query", None) is not None
        and tuple(model.learnable_query.shape) == (nq, model.config.hidden_size),
        f"shape={None if model.learnable_query is None else tuple(model.learnable_query.shape)}",
    )
    check(
        "build: breen distill head + norm_layer present",
        getattr(model, "visual_distill_head", None) is not None
        and hasattr(model.visual_distill_head, "norm_layer"),
    )
    check(
        "build: visual experts + gates installed",
        len(getattr(model.model, "_visual_expert_mlps", [])) == len(model.model.layers)
        and all(getattr(m, "_expert_gate", False) for m in model.model._visual_expert_mlps),
    )

    img0, pos0 = mk_image(processor, device, dtype, seed=0)
    img1, pos1 = mk_image(processor, device, dtype, seed=12345)
    q, ans = [40, 41, 42], [100, 200]
    imgs, poss = [img0, img1], [pos0, pos1]

    # ---- DIAG: isolate the NaN on a FRESH (untrained) model -------------
    di, da_, dl, dim, dps = batch(model, imgs, poss, q, ans, device)
    fe = model.encode_raw_patches(dim, dps)
    print("DIAG image_features finite:", bool(all(torch.isfinite(f).all() for f in fe)), flush=True)
    print(
        "DIAG learnable_query finite:",
        bool(torch.isfinite(model.learnable_query).all()),
        "absmax=",
        float(model.learnable_query.abs().max()),
        flush=True,
    )
    model.eval()
    bad = []
    hs = [
        layer.register_forward_hook(
            (
                lambda li: (
                    lambda m, i, o: (
                        bad.append(li)
                        if not torch.isfinite(o[0] if isinstance(o, tuple) else o).all()
                        else None
                    )
                )
            )(li)
        )
        for li, layer in enumerate(model.model.layers)
    ]
    with torch.no_grad():
        oev = model(input_ids=di, attention_mask=da_, images=dim, image_position_ids=dps)
    for h in hs:
        h.remove()
    print(
        "DIAG first non-finite decoder layers:",
        sorted(set(bad))[:5],
        "| eval logits finite:",
        bool(torch.isfinite(oev.logits).all()),
        flush=True,
    )
    model.train()
    model.config.visual_distill_weight = 0.0
    with torch.no_grad():
        oce = model(input_ids=di, attention_mask=da_, labels=dl, images=dim, image_position_ids=dps)
    print(
        "DIAG CE-only(train,no-distill) finite:",
        bool(torch.isfinite(oce.loss)),
        "val=",
        float(oce.loss),
        flush=True,
    )
    model.config.visual_distill_weight = 1.0
    model.train()

    # ---- (b) label-mask + expert routing via the splice -----------------
    iid, attn, lab, images, posl = batch(model, imgs, poss, q, ans, device)
    img_feats = model.encode_raw_patches(images, posl)
    (_, _, _, _, _, new_labels, image_block_ids, query_block_ids) = (
        model.prepare_inputs_labels_for_multimodal(
            iid, None, attn, None, lab, img_feats, None, with_image_block_ids=True
        )
    )
    ign = model.config.ignore_index
    q_pos = query_block_ids >= 0
    img_pos = image_block_ids >= 0
    n_q_rows = int(q_pos.sum())
    check(
        "(b) query rows == num_query per image (100 x 2)",
        n_q_rows == nq * len(images),
        f"{n_q_rows} vs {nq * len(images)}",
    )
    check(
        "(b) query positions are label-masked (CE ignores them)",
        bool((new_labels[q_pos] == ign).all()),
    )
    check("(b) image positions are label-masked", bool((new_labels[img_pos] == ign).all()))
    # The answer tokens (the only non-ignored labels) must survive.
    check(
        "(b) answer tokens still supervised",
        int((new_labels != ign).sum()) == len(ans),
        f"{int((new_labels != ign).sum())} vs {len(ans)}",
    )
    # The visual-expert routing mask = image OR query positions.
    routed = img_pos | q_pos
    check(
        "(b) routing mask covers image AND query positions",
        int(routed.sum()) == int(img_pos.sum()) + n_q_rows,
    )

    # ---- (a) mechanics + grads ------------------------------------------
    out = model(
        input_ids=iid, attention_mask=attn, labels=lab, images=images, image_position_ids=posl
    )
    comps = getattr(model, "_last_ce_components", {})
    check("(a) finite loss", torch.isfinite(out.loss).item(), f"loss={out.loss.item():.4f}")
    check(
        "(a) distill + distill_cos stashed",
        "distill" in comps and "distill_cos" in comps,
        f"keys={sorted(comps)}",
    )
    out.loss.backward()

    def grad_nonzero(p):
        return p is not None and p.grad is not None and p.grad.abs().sum().item() > 0.0

    check("(a) learnable_query has gradient", grad_nonzero(model.learnable_query))
    norm_g = next((p for p in model.visual_distill_head.parameters() if p.grad is not None), None)
    check("(a) distill head (norm_layer) has gradient", grad_nonzero(norm_g))
    mv = model.model._visual_expert_mlps[0].mlp_visual.gate_proj.weight
    check("(a) mlp_visual has gradient (image+query routed)", grad_nonzero(mv))
    eg = model.model._visual_expert_mlps[0].expert_gate_visual.weight
    check("(a) expert gate has gradient", grad_nonzero(eg))
    check("(a) lm_head has gradient", grad_nonzero(model.lm_head.weight))
    model.zero_grad(set_to_none=True)

    # ---- (a) overfit: distill_cos RISES ---------------------------------
    def distill_now():
        i, a, lb, im, ps = batch(model, imgs, poss, q, ans, device)
        o = model(input_ids=i, attention_mask=a, labels=lb, images=im, image_position_ids=ps)
        c = model._last_ce_components
        return float(c["distill"]), float(c["distill_cos"]), float(o.loss)

    d0, cos0, _ = distill_now()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    d_last = cos_last = None
    for step in range(args.steps):
        i, a, lb, im, ps = batch(model, imgs, poss, q, ans, device)
        o = model(input_ids=i, attention_mask=a, labels=lb, images=im, image_position_ids=ps)
        opt.zero_grad(set_to_none=True)
        o.loss.backward()
        opt.step()
        d_last = float(model._last_ce_components["distill"])
        cos_last = float(model._last_ce_components["distill_cos"])
        if step % 30 == 0:
            print(
                f"  step {step:3d}: loss={float(o.loss):.4f} distill={d_last:.4f} cos={cos_last:.4f}",
                flush=True,
            )
    check("(a) distill loss decreased", d_last < d0 - 1e-3, f"{d0:.4f} -> {d_last:.4f}")
    check(
        "(a) query<->CLIP cosine ROSE from ~0",
        cos_last > cos0 + 0.05,
        f"cos {cos0:.3f} -> {cos_last:.3f}",
    )

    # ---- (c) teacher invisibility + inference without CLIP --------------
    pnames = [n for n, _ in model.named_parameters()]
    check(
        "(c) teacher not in named_parameters",
        not any("_distill_teacher" in n or "teacher" in n for n in pnames),
    )
    sd_keys = list(model.state_dict().keys())
    check(
        "(c) no teacher weights in state_dict",
        not any("_distill_teacher" in k for k in sd_keys),
        f"{len(sd_keys)} keys",
    )
    check(
        "(c) learnable_query IS in state_dict (trained -> serialized)", "learnable_query" in sd_keys
    )
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        processor.save_pretrained(d)
        # Load the checkpoint WITHOUT attaching the CLIP teacher (the inference
        # path: no CLIP files needed). Forward in eval (no labels) must work.
        VLMForCausalLM, _ = get_dynamic_vlm(args.base_lm)
        reloaded = VLMForCausalLM.from_pretrained(d, dtype=dtype, attn_implementation="sdpa").to(
            device
        )
        reloaded.eval()
        check(
            "(c) reloaded model has NO teacher attached (inference path)",
            getattr(reloaded, "_distill_teacher", None) is None,
        )
        check(
            "(c) reloaded model rebuilt the queries + breen head",
            getattr(reloaded, "learnable_query", None) is not None
            and getattr(reloaded, "visual_distill_head", None) is not None,
        )
        ri, ra, _, rim, rps = batch(reloaded, imgs, poss, q, ans, device)
        with torch.no_grad():
            rout = reloaded(input_ids=ri, attention_mask=ra, images=rim, image_position_ids=rps)
        check(
            "(c) forward works with no CLIP teacher (finite logits)",
            rout.logits is not None and torch.isfinite(rout.logits).all().item(),
        )
        del reloaded

    # ---- (d) frozen-LLM param grouping ----------------------------------
    fm, _ = build(device, args.base_lm)
    frozen_cfg = OmegaConf.structured(
        UnfreezeConfig(train_vision_model=False, train_language_model=False, train_connector=True)
    )
    set_trainable_params(fm, frozen_cfg)

    def req(substr):
        # noqa justified: `fm` is bound above and `req` runs before the later
        # `del fm`; pyflakes flags the closure ref only because of that del.
        return [p for n, p in fm.named_parameters() if substr in n]  # noqa: F821

    check("(d) learnable_query trainable under frozen LM", fm.learnable_query.requires_grad)
    check(
        "(d) visual_distill_head trainable under frozen LM",
        all(p.requires_grad for p in fm.visual_distill_head.parameters()),
    )
    check(
        "(d) mlp_visual trainable under frozen LM",
        bool(req(".mlp_visual.")) and all(p.requires_grad for p in req(".mlp_visual.")),
    )
    check(
        "(d) expert gates trainable under frozen LM",
        bool(req("expert_gate")) and all(p.requires_grad for p in req("expert_gate")),
    )
    # The LM trunk must stay FROZEN: a text-FFN weight (NOT mlp_visual) and the
    # token embeddings are off.
    text_ffn = [
        p
        for n, p in fm.named_parameters()
        if ".mlp.gate_proj.weight" in n and ".mlp_visual." not in n
    ]
    check(
        "(d) LM text-FFN stays frozen",
        bool(text_ffn) and not any(p.requires_grad for p in text_ffn),
    )
    check(
        "(d) token embeddings stay frozen",
        not any(p.requires_grad for n, p in fm.named_parameters() if "embed_tokens" in n),
    )
    # And gradients actually flow to the visual modules on a train fwd/bwd.
    fm.train()
    fi, fa, fl, fim, fps = batch(fm, imgs, poss, q, ans, device)
    fout = fm(input_ids=fi, attention_mask=fa, labels=fl, images=fim, image_position_ids=fps)
    fout.loss.backward()
    check("(d) frozen-LM: learnable_query RECEIVES gradient", grad_nonzero(fm.learnable_query))
    nl = next((p for p in fm.visual_distill_head.parameters() if p.grad is not None), None)
    check("(d) frozen-LM: norm_layer RECEIVES gradient", grad_nonzero(nl))
    fmv = fm.model._visual_expert_mlps[0].mlp_visual.gate_proj.weight
    check("(d) frozen-LM: mlp_visual RECEIVES gradient", grad_nonzero(fmv))
    del fm

    # ---- (item 7) data-path query injection -----------------------------
    da = DataArguments(
        learnable_query_enabled=True,
        query_token="<query>",
        query_token_index=-202,
        query_placement="after_image",
    )
    turns = [{"from": "human", "value": "<image>\nWhat is this?"}]
    inject_query_placeholders(turns, n_images=1, data_args=da)
    check(
        "(item7) after_image injects <query> right after <image>",
        "<image>\n<query>" in turns[0]["value"],
        turns[0]["value"],
    )
    da_t = DataArguments(learnable_query_enabled=True, query_placement="after_text")
    turns2 = [{"from": "human", "value": "<image>\nDescribe."}]
    inject_query_placeholders(turns2, n_images=1, data_args=da_t)
    check(
        "(item7) after_text appends <query> after the question",
        turns2[0]["value"].rstrip().endswith("<query>") and "Describe" in turns2[0]["value"],
        turns2[0]["value"],
    )
    ids = tokenizer_multimodal_token(turns[0]["value"], processor.tokenizer, da)
    check("(item7) tokenizer maps <query> -> query_token_index (-202)", -202 in ids)

    print("\n" + ("ALL BREEN SMOKE CHECKS PASSED" if OK else "BREEN SMOKE FAILED"))
    sys.exit(0 if OK else 1)


if __name__ == "__main__":
    main()
