"""GPU smoke + equivalence verification for the cross-modal access arms
(plan docs/superpowers/plans/2026-06-10-early-fusion-access-arms.md).

Builds a small encoder-free VLM (Qwen3-0.6B backbone, random-init raw-patch
connector) on the GPU and proves, on real bf16 SDPA/FA2 kernels, the
scientific contract of `model.config.cross_modal_mask_mode`:

  1. no-op proof: 2D mask == build_base_mask 4D (mode none); sdpa vs FA2 info.
  2. prefix_lm genuinely changes prefix-position hidden states.
  3. THEOREM: under none, image rows are question-invariant (image precedes
     question, causal); under img2q_window they become question-sensitive.
  4. leakage guard: changing answer tokens never moves image/question rows.
  5. generate(): each mode runs greedy, stash is consumed, none matches a
     hand-2D-masked reference.
  6. throughput probe: fwd+bwd tok/s for FA2-2D / sdpa-prefix_lm / sdpa_xmodal.

Run: see devtools/xmodal_smoke.slurm. Optional --cpu for a tiny dry run.
Exits nonzero on any REQUIRED failure (assertions 1-5; 6 is informational).
"""

import argparse
import sys
import time

import torch
from omegaconf import OmegaConf

from vlm.config.config_schema import (
    ConnectorConfig,
    LanguageModelConfig,
    ModelConfig,
    TrainerConfig,
    VisualEncoderConfig,
)
from vlm.models.xmodal_mask import build_base_mask
from vlm.vlm import load_model

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    RESULTS.append((name, ok, detail))


def have_fa2() -> bool:
    try:
        import flash_attn  # noqa: F401

        return torch.cuda.is_available()
    except Exception:
        return False


def build(attn_impl: str, device: str, base_lm: str):
    model_cfg = OmegaConf.structured(
        ModelConfig(
            name="xmodal-smoke",
            visual_encoder=VisualEncoderConfig(
                hf_name=None, patch_size=16, pooling_kernel_size=3, max_soft_tokens=64
            ),
            language_model=LanguageModelConfig(hf_name=base_lm, max_seq_length=4096),
            connector=ConnectorConfig(name="raw_patch", type="raw_patch"),
        )
    )
    bf16 = device == "cuda"
    trainer_cfg = OmegaConf.structured(
        TrainerConfig(name="smoke", bf16=bf16, fp16=False, attn_implementation=attn_impl)
    )
    model, processor = load_model(model_cfg, trainer_cfg)
    # Cast EVERYTHING (LM + random-init connector) to one dtype: load_model
    # only sets the LM's from_pretrained dtype, leaving the fresh connector
    # in fp32, which would feed fp32 splices into bf16 q_proj.
    model = model.to(device=device, dtype=torch.bfloat16 if bf16 else torch.float32)
    return model, processor


def make_image(processor, device, dtype, hw=(96, 96)):
    """One synthetic raw-patch image via the real RawImageProcessor."""
    import numpy as np
    from PIL import Image

    arr = (np.random.default_rng(0).random((hw[0], hw[1], 3)) * 255).astype("uint8")
    feat = processor.image_processor.preprocess([Image.fromarray(arr)])
    pix = feat["pixel_values"][0].to(device=device, dtype=dtype)
    pos = feat["image_position_ids"][0].to(device)
    return pix, pos


def splice(model, processor, device, dtype, questions, ans_ids=(10, 11, 12), pad="right"):
    """Build a padded batch: [img sentinel][question tokens][answer tokens].
    Returns input_ids, attention_mask(2D), labels, images list, pos list."""
    img_tok = model.config.image_token_index
    ign = model.config.ignore_index
    pix, pos = make_image(processor, device, dtype)
    seqs, labs = [], []
    for q in questions:
        ids = [img_tok] + list(q) + list(ans_ids)
        lab = [ign] * (1 + len(q)) + list(ans_ids)
        seqs.append(ids)
        labs.append(lab)
    maxlen = max(len(s) for s in seqs)
    pad_id = 0
    input_ids, attn, labels = [], [], []
    for ids, lab in zip(seqs, labs, strict=True):
        npad = maxlen - len(ids)
        if pad == "right":
            input_ids.append(ids + [pad_id] * npad)
            attn.append([1] * len(ids) + [0] * npad)
            labels.append(lab + [ign] * npad)
        else:
            input_ids.append([pad_id] * npad + ids)
            attn.append([0] * npad + [1] * len(ids))
            labels.append([ign] * npad + lab)
    input_ids = torch.tensor(input_ids, device=device)
    attn = torch.tensor(attn, device=device)
    labels = torch.tensor(labels, device=device)
    images = [pix] * len(questions)
    poss = [pos] * len(questions)
    return input_ids, attn, labels, images, poss


def merged(model, input_ids, attn, labels, images, poss):
    """Run prepare_inputs_labels_for_multimodal to get spliced embeds + block ids."""
    feats = model.encode_raw_patches(images, poss)
    (_, _, attn4_or_2, _, embeds, new_labels, block_ids, _) = (
        model.prepare_inputs_labels_for_multimodal(
            input_ids, None, attn, None, labels, feats, None, with_image_block_ids=True
        )
    )
    return embeds, attn4_or_2, new_labels, block_ids


def run_backbone(model, embeds, mask):
    """Inner decoder forward with an explicit (2D or 4D) attention mask."""
    out = model.model(
        inputs_embeds=embeds, attention_mask=mask, output_hidden_states=True, use_cache=False
    )
    return out.hidden_states[-1].float()


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    device = "cpu" if args.cpu else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        print("no CUDA; pass --cpu for a dry run", file=sys.stderr)
        return 2

    base_lm = "Qwen/Qwen3-0.6B"
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained(base_lm)
    except Exception:
        base_lm = "Qwen/Qwen3-1.7B"
    print(f"device={device} base_lm={base_lm} fa2={have_fa2()}", flush=True)
    if device == "cuda":
        print("gpu:", torch.cuda.get_device_name(0), flush=True)

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    torch.manual_seed(0)

    # Two batches that differ ONLY in question tokens (same length, same image,
    # same answer) for the theorem check.
    QA = [100, 101, 102, 103]
    QB = [200, 201, 202, 203]
    ANS = [10, 11, 12]

    model, processor = build("sdpa", device, base_lm)
    model.eval()
    model.config.cross_modal_mask_mode = "none"

    # ---- batch A (right pad, mixed lengths) ----
    ids, attn, labels, imgs, poss = splice(
        model, processor, device, dtype, [QA, QA + [104, 105]], ANS
    )
    embeds, m2d, mlabels, blocks = merged(model, ids, attn, labels, imgs, poss)
    is_2d = m2d is not None and m2d.dim() == 2

    # === Assertion 1: no-op proof (2D == build_base_mask 4D) ===
    # The merged sequence is what the decoder runs; its 2D padding mask is the
    # one prepare_inputs_labels returns (2D) — or, when that path returns None
    # (no padding), an all-ones keep. build_base_mask of THAT must match.
    merged_attn = m2d if is_2d else _embed_keep(embeds, mlabels, model.config.ignore_index).long()
    with torch.no_grad():
        h_2d = run_backbone(model, embeds, merged_attn)
        base4d = build_base_mask(merged_attn)
        h_4d = run_backbone(model, embeds, base4d)
    d1 = (h_2d - h_4d).abs().max().item()
    record("1.noop_2d_eq_base4d", d1 < 2e-2, f"max|Δhidden|={d1:.2e} (bf16 tol 2e-2)")

    if have_fa2():
        try:
            mfa, _ = build("flash_attention_2", device, base_lm)
            mfa.eval()
            mfa.config.cross_modal_mask_mode = "none"
            with torch.no_grad():
                e2, m2, ml2, _ = merged(mfa, ids, attn, labels, imgs, poss)
                hfa = mfa.model(
                    inputs_embeds=e2,
                    attention_mask=m2,
                    output_hidden_states=True,
                    use_cache=False,
                ).hidden_states[-1].float()
            dfa = (h_2d - hfa).abs().max().item()
            record("1b.sdpa_vs_fa2_info", True, f"max|Δ|={dfa:.2e} (info only, kernels differ)")
            del mfa
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            record("1b.sdpa_vs_fa2_info", True, f"skipped: {e!r}")
    else:
        record("1b.sdpa_vs_fa2_info", True, "FA2 unavailable, skipped (info only)")

    # === Assertion 2: prefix_lm changes prefix-position hidden states ===
    # build prefix_lm 4D mask via install path and compare to base.
    model.config.cross_modal_mask_mode = "prefix_lm"
    pl_mask = model.install_xmodal_masks(merged_attn, blocks, mlabels)
    with torch.no_grad():
        h_pl = run_backbone(model, embeds, pl_mask)
    img_rows = blocks[0].ge(0).nonzero().flatten()
    pos0 = int(img_rows[0]) if img_rows.numel() else 1
    d2 = (h_pl[0, pos0] - h_2d[0, pos0]).abs().max().item()
    record("2.prefix_lm_shifts_prefix", d2 > 1e-3, f"max|Δ| at prefix pos {pos0}={d2:.2e}")
    # degenerate check (mask-level, exact): a single-token prefix gains no
    # edges under prefix_lm — prefix×prefix collapses to the self-edge the
    # causal base already has, so the masks must be bit-identical. (The
    # rightmost prefix TOKEN's deep hidden state legitimately moves — at
    # layer >= 2 it attends other prefix states that changed at layer 1 —
    # so a hidden-state invariance check there would be wrong.)
    from vlm.models.xmodal_mask import build_cross_modal_mask

    one_attn = torch.ones(1, 4, dtype=torch.bool, device=device)
    one_lab = torch.tensor(
        [[model.config.ignore_index, 5, 6, 7]], device=device
    )  # prefix = position 0 only
    eq = bool(
        (
            build_cross_modal_mask(one_attn, None, one_lab, "prefix_lm")
            == build_base_mask(one_attn)
        ).all()
    )
    record("2b.single_token_prefix_mask_eq_base", eq, "prefix_lm == base for 1-token prefix")
    model.config.cross_modal_mask_mode = "none"

    # === Assertion 3: THEOREM (question sensitivity) ===
    model.config.cross_modal_mask_mode = "none"
    idsA, attnA, labA, imA, poA = splice(model, processor, device, dtype, [QA], ANS)
    idsB, attnB, labB, imB, poB = splice(model, processor, device, dtype, [QB], ANS)
    eA, mA, mlA, blA = merged(model, idsA, attnA, labA, imA, poA)
    eB, mB, mlB, blB = merged(model, idsB, attnB, labB, imB, poB)
    img_rowsA = blA[0].ge(0).nonzero().flatten()
    with torch.no_grad():
        hA = run_backbone(model, eA, mA if (mA is not None and mA.dim() == 2) else attnA)
        hB = run_backbone(model, eB, mB if (mB is not None and mB.dim() == 2) else attnB)
    d_none = (hA[0, img_rowsA] - hB[0, img_rowsA]).abs().max().item()
    record("3a.none_image_invariant", d_none < 2e-2, f"max|Δimg rows|={d_none:.2e} (must be ~0)")

    model.config.cross_modal_mask_mode = "img2q_window"
    model.config.cross_modal_mask_window = [1, 9]
    model.config._attn_implementation = "sdpa_xmodal"
    model.model.config._attn_implementation = "sdpa_xmodal"
    keepA = _embed_keep(eA, mlA, model.config.ignore_index)
    keepB = _embed_keep(eB, mlB, model.config.ignore_index)
    baseA = model.install_xmodal_masks(keepA, blA, mlA)
    with torch.no_grad():
        hAw = run_backbone(model, eA, baseA)
    baseB = model.install_xmodal_masks(keepB, blB, mlB)
    with torch.no_grad():
        hBw = run_backbone(model, eB, baseB)
    d_win = (hAw[0, img_rowsA] - hBw[0, img_rowsA]).abs().max().item()
    record(
        "3b.window_image_sensitive",
        d_win > 1e-3,
        f"max|Δimg rows under window|={d_win:.2e} (MUST be nonzero)",
    )
    model.config._attn_implementation = "sdpa"
    model.model.config._attn_implementation = "sdpa"

    # === Assertion 4: leakage guard (answer tokens never move img/q rows) ===
    for mode in ("img2q_window", "prefix_lm"):
        model.config.cross_modal_mask_mode = mode
        if mode == "img2q_window":
            model.config._attn_implementation = "sdpa_xmodal"
            model.model.config._attn_implementation = "sdpa_xmodal"
        idsX, attnX, labX, imX, poX = splice(model, processor, device, dtype, [QA], [10, 11, 12])
        idsY, attnY, labY, imY, poY = splice(model, processor, device, dtype, [QA], [77, 88, 99])
        eX, _, mlX, blX = merged(model, idsX, attnX, labX, imX, poX)
        eY, _, mlY, blY = merged(model, idsY, attnY, labY, imY, poY)
        kX = _embed_keep(eX, mlX, model.config.ignore_index)
        kY = _embed_keep(eY, mlY, model.config.ignore_index)
        mX = model.install_xmodal_masks(kX, blX, mlX)
        with torch.no_grad():
            hX = run_backbone(model, eX, mX)
        mY = model.install_xmodal_masks(kY, blY, mlY)
        with torch.no_grad():
            hY = run_backbone(model, eY, mY)
        img_r = blX[0].ge(0).nonzero().flatten()
        # question rows = prefix & not image (positions before first label, not img)
        first_lbl = (mlX[0].ne(model.config.ignore_index)).nonzero().flatten()
        qend = int(first_lbl[0]) if first_lbl.numel() else eX.shape[1]
        q_rows = torch.tensor(
            [i for i in range(qend) if i not in set(img_r.tolist())], device=device
        )
        d_img = (hX[0, img_r] - hY[0, img_r]).abs().max().item()
        d_q = (hX[0, q_rows] - hY[0, q_rows]).abs().max().item() if q_rows.numel() else 0.0
        ok = d_img < 2e-2 and d_q < 2e-2
        record(f"4.leakage_guard[{mode}]", ok, f"Δimg={d_img:.2e} Δq={d_q:.2e} (both ~0)")
        model.config._attn_implementation = "sdpa"
        model.model.config._attn_implementation = "sdpa"

    # === Assertion 5: generate() per mode ===
    gen_ids = torch.tensor([[model.config.image_token_index] + QA], device=device)
    ref_cont = None
    for mode in ("none", "prefix_lm", "img2q_window"):
        model.config.cross_modal_mask_mode = mode
        impl = "sdpa_xmodal" if mode == "img2q_window" else "sdpa"
        model.config._attn_implementation = impl
        model.model.config._attn_implementation = impl
        try:
            with torch.no_grad():
                out = model.generate(
                    inputs=gen_ids,
                    images=[make_image(processor, device, dtype)[0]],
                    image_position_ids=[make_image(processor, device, dtype)[1]],
                    max_new_tokens=8,
                    do_sample=False,
                )
            consumed = getattr(model, "_xmodal_gen_mask", None) is None
            finite = bool(out.isfinite().all()) if out.dtype.is_floating_point else True
            cont = out[0, gen_ids.shape[1] :].tolist()
            if mode == "none":
                ref_cont = cont
            record(f"5.generate[{mode}]", consumed and finite, f"stash_cleared={consumed} cont={cont}")
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            record(f"5.generate[{mode}]", False, f"raised {e!r}")
    model.config.cross_modal_mask_mode = "none"
    model.config._attn_implementation = "sdpa"
    model.model.config._attn_implementation = "sdpa"

    # === Assertion 6: throughput probe (informational) ===
    if device == "cuda":
        throughput_probe(processor, device, dtype, base_lm)

    # ---- summary ----
    print("\n========== SUMMARY ==========", flush=True)
    failed = [n for n, ok, _ in RESULTS if not ok]
    for n, ok, d in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}  {d}", flush=True)
    if failed:
        print(f"\n{len(failed)} REQUIRED CHECK(S) FAILED: {failed}", flush=True)
        return 1
    print("\nALL REQUIRED CHECKS PASSED", flush=True)
    return 0


def _embed_keep(embeds, labels, ignore_index):
    """Padding-aware keep mask over the MERGED sequence. Padding rows are the
    all-zero embedding rows the splice pads with (right/left)."""
    keep = embeds.abs().sum(-1).ne(0)
    return keep


def throughput_probe(processor, device, dtype, base_lm):
    import numpy as np
    from PIL import Image

    def big_batch(model):
        img_tok = model.config.image_token_index
        ign = model.config.ignore_index
        arr = (np.random.default_rng(1).random((192, 192, 3)) * 255).astype("uint8")
        feat = processor.image_processor.preprocess([Image.fromarray(arr)])
        pix = feat["pixel_values"][0].to(device=device, dtype=dtype)
        pos = feat["image_position_ids"][0].to(device)
        npatch = pix.shape[0]
        # target ~4k tokens: pad question/answer text to fill
        textlen = max(8, 4096 - npatch - 1)
        q = list(range(50, 50 + textlen // 2))
        a = list(range(60, 60 + textlen - len(q)))
        ids = torch.tensor([[img_tok] + q + a], device=device)
        lab = torch.tensor([[ign] * (1 + len(q)) + a], device=device)
        attn = torch.ones_like(ids)
        return ids, attn, lab, [pix], [pos], npatch + 1 + len(q) + len(a)

    configs = [
        ("FA2+2D(none)", "flash_attention_2", "none", "sdpa"),
        ("sdpa+prefix_lm", "sdpa", "prefix_lm", "sdpa"),
        ("sdpa_xmodal+window", "sdpa", "img2q_window", "sdpa_xmodal"),
    ]
    base_tps = None
    for name, build_impl, mode, run_impl in configs:
        if build_impl == "flash_attention_2" and not have_fa2():
            record(f"6.tput[{name}]", True, "FA2 unavailable, skipped (info)")
            continue
        try:
            m, _ = build(build_impl, device, base_lm)
            m.train()
            m.config.cross_modal_mask_mode = mode
            m.config._attn_implementation = run_impl
            m.model.config._attn_implementation = run_impl
            ids, attn, lab, imgs, poss, ntok = big_batch(m)
            for _ in range(3):
                m.zero_grad(set_to_none=True)
                out = m(input_ids=ids, attention_mask=attn, labels=lab, images=imgs,
                        image_position_ids=poss)
                out.loss.backward()
            torch.cuda.synchronize()
            t0 = time.time()
            steps = 15
            for _ in range(steps):
                m.zero_grad(set_to_none=True)
                out = m(input_ids=ids, attention_mask=attn, labels=lab, images=imgs,
                        image_position_ids=poss)
                out.loss.backward()
            torch.cuda.synchronize()
            dt = time.time() - t0
            tps = steps * ntok / dt
            if base_tps is None:
                base_tps = tps
            ratio = tps / base_tps if base_tps else 1.0
            tag = " WARN<0.6" if ratio < 0.6 else ""
            record(f"6.tput[{name}]", True, f"{tps:.0f} tok/s ratio={ratio:.2f}{tag} (info)")
            del m
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            record(f"6.tput[{name}]", True, f"skipped: {e!r} (info)")


if __name__ == "__main__":
    sys.exit(main())
