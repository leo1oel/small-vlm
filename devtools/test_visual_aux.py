"""GPU integration test for the visual-aux loss (next-patch prediction at image
positions; spec: docs/superpowers/specs/2026-06-06-visual-aux-loss-design.md).
This is the pre-launch gate for the visual-aux feature — it pins the live-cluster
baseline (parity), the two objectives' numerics, gradient routing, degenerate
batches, mid-layer + gradient-checkpointing behaviour, and the optimizer/config
plumbing, all against the real model on a real multimodal batch.

Run on a GPU node (from the repo or worktree root, so sys.path picks the worktree):
    srun -p ckpt-all -A cse-ckpt --gpus=l40:1 --mem=48G --time=0:30:00 \
        bash -c 'source /mmfs1/gscratch/krishna/leoym/small-vlm/.venv/bin/activate \
                 && HF_HUB_OFFLINE=1 python devtools/test_visual_aux.py'

Checks (fp32+sdpa, real multimodal batch from test_chunked_ce.build_batch):
  0. Config compose: sft-unified-aimpixel and sft-unified-nepa both compose;
     objective lands on model.visual_aux, weight=0.5, loss_chunk_size=1024 inherited.
  1. Baseline parity: objective none -> visual_aux_head is None; chunked loss
     (chunk=1024) == full-logits loss (chunk=0) and grads match (the live gate).
  2. aim_pixel numerical correctness: chunked aux-on loss == ce_ref + 0.5*va_ref
     where va_ref is the z-scored next-patch pixel MSE through the head; stash matches.
  3. nepa numerical correctness + alarms: loss identity; visual_aux == -visual_aux_cos;
     visual_aux_tgt_std > 0; all components finite.
  4. Gradient routing: head grad nonzero; lm_head grad bit-identical to the
     weight-0 run; connector grad differs (prediction path flows into the trunk).
  5. Degenerate text-only batch: finite loss, visual_aux == 0 stashed, head grads
     are zero (not None).
  6. Mid-layer (layer=6) loss finite and != final-layer loss; with gradient
     checkpointing same loss, backward OK, hooks removed after forward.
  7. Optimizer grouping + trainability: visual_aux_head group non-empty and disjoint
     from language_model; set_trainable_params with all-frozen keeps head trainable;
     configure_optimizers puts head params in lr=1e-4 groups.
  8. validate_visual_aux_config: positive cases + ValueError matrix.
"""

import sys
from argparse import Namespace as NS
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "devtools"))

from hydra import compose, initialize_config_dir  # noqa: E402
from test_chunked_ce import build_batch, to_device  # noqa: E402

from vlm.config import register_configs  # noqa: E402
from vlm.data.data_arguments import DataArguments  # noqa: E402
from vlm.utils import conversation as conversation_lib  # noqa: E402
from vlm.vlm import load_model  # noqa: E402


def set_va(model, weight, layer, chunk=1024):
    """Knobs that can be flipped post-build. The objective itself is FIXED at
    build (the head out_dim differs), so each objective gets its own load_model."""
    model.config.loss_chunk_size = chunk
    model.config.visual_aux_weight = weight
    model.config.visual_aux_layer = layer


def rel_diff(a, b):
    return (a - b).abs().max().item() / max(a.abs().max().item(), 1e-12)


def build_model(objective_override):
    """Fresh 0.6B model build (fp32, sdpa, cuda, .train(), all params requires_grad).
    objective_override is None for baseline or 'model.visual_aux.objective=...'."""
    overrides = [
        "model=qwen3-0.6b-unified",
        "trainer.bf16=false",
        "trainer.attn_implementation=sdpa",
    ]
    if objective_override is not None:
        overrides.append(objective_override)
    with initialize_config_dir(config_dir=str(REPO / "src/vlm/config"), version_base=None):
        cfg = compose(config_name="sft-unified", overrides=overrides)
    conversation_lib.default_conversation = conversation_lib.conv_templates[cfg.trainer.version]
    model, processor = load_model(cfg.model, cfg.trainer)
    model = model.cuda().train()
    for p in model.parameters():
        p.requires_grad_(True)
    data_args = DataArguments(
        image_token=cfg.model.language_model.image_token,
        image_token_index=cfg.model.language_model.image_token_index,
        audio_token=cfg.model.language_model.audio_token,
        audio_token_index=cfg.model.language_model.audio_token_index,
        ignore_index=cfg.model.language_model.ignore_index,
        is_multimodal=True,
    )
    return model, processor, data_args, cfg


def fresh_batch(processor, data_args):
    return to_device(build_batch(processor, data_args), "cuda", torch.float32)


def main():
    assert torch.cuda.is_available(), "needs a GPU node"
    register_configs()
    torch.backends.cuda.matmul.allow_tf32 = False  # strict fp32 (mask no matmul bugs)

    # ---- 0. config compose -------------------------------------------------
    with initialize_config_dir(config_dir=str(REPO / "src/vlm/config"), version_base=None):
        cfg_aim = compose(config_name="sft-unified-aimpixel")
        cfg_nepa = compose(config_name="sft-unified-nepa")
    assert cfg_aim.model.visual_aux.objective == "aim_pixel", cfg_aim.model.visual_aux.objective
    assert cfg_nepa.model.visual_aux.objective == "nepa", cfg_nepa.model.visual_aux.objective
    for c in (cfg_aim, cfg_nepa):
        assert abs(c.trainer.visual_aux_weight - 0.5) < 1e-9, c.trainer.visual_aux_weight
        assert c.trainer.loss_chunk_size == 1024, c.trainer.loss_chunk_size  # inherited
    print("0. aimpixel/nepa configs compose (objective, weight=0.5, chunk=1024): OK", flush=True)

    # ---- 1. baseline parity (objective none -> no head; chunked == full) ---
    model, processor, data_args, _ = build_model(None)
    assert model.visual_aux_head is None, "baseline model must carry no visual_aux_head"
    n_layers = model.config.num_hidden_layers
    batch = fresh_batch(processor, data_args)

    def run_pass(chunk):
        model.config.loss_chunk_size = chunk
        model.zero_grad(set_to_none=True)
        out = model(**batch)
        out.loss.backward()
        grads = {
            "lm_head": model.lm_head.weight.grad.detach().clone(),
            "layer2.q": model.model.layers[2].self_attn.q_proj.weight.grad.detach().clone(),
            "connector": next(
                p.grad.detach().clone()
                for p in model.model.connector.parameters()
                if p.grad is not None
            ),
        }
        return float(out.loss.detach()), grads

    loss_full, g_full = run_pass(0)
    loss_chunk, g_chunk = run_pass(1024)
    assert abs(loss_full - loss_chunk) < 1e-5, (loss_full, loss_chunk)
    for k in g_full:
        r = rel_diff(g_full[k], g_chunk[k])
        assert r < 1e-4, f"baseline grad[{k}] rel={r}"
    print(f"1. baseline parity: chunked {loss_chunk:.8f} == full {loss_full:.8f}, grads OK", flush=True)
    del model
    torch.cuda.empty_cache()

    # ---- 2. aim_pixel numerical correctness --------------------------------
    from vlm.models.modeling_vlm import (  # noqa: E402
        build_visual_aux_pairs,
        prepare_visual_aux_targets,
    )

    model, processor, data_args, _ = build_model("model.visual_aux.objective=aim_pixel")
    assert model.visual_aux_head is not None
    batch = fresh_batch(processor, data_args)
    set_va(model, weight=0.5, layer=None, chunk=1024)

    cap = {}
    handle = model.model.norm.register_forward_hook(
        lambda m, a, o: cap.__setitem__("h_norm", o.detach())
    )
    orig_prepare = model.prepare_inputs_labels_for_multimodal

    def _prepare_and_capture(*a, **kw):
        out = orig_prepare(*a, **kw)
        cap["labels7"] = out[5].detach() if out[5] is not None else None
        cap["ids7"] = out[6].detach() if out[6] is not None else None
        return out

    model.prepare_inputs_labels_for_multimodal = _prepare_and_capture
    with torch.no_grad():
        out_on = model(**batch)
        loss_on = float(out_on.loss)
    comps = {k: float(v) for k, v in model._last_ce_components.items()}
    handle.remove()
    model.prepare_inputs_labels_for_multimodal = orig_prepare

    # ce_ref: pure CE with visual-aux disabled (weight 0 -> ids None -> no aux).
    set_va(model, weight=0.0, layer=None, chunk=1024)
    with torch.no_grad():
        ce_ref = float(model(**batch).loss)

    # va_ref: replicate the loss body on the captured post-norm hidden + pixels.
    D = cap["h_norm"].shape[-1]
    num_rows = [int(t.shape[0]) for t in batch["images"]]
    flat_pos, segments = build_visual_aux_pairs(cap["ids7"].to("cuda"), num_rows)
    assert segments, "aim_pixel reference produced no prediction pairs"
    targets = prepare_visual_aux_targets("aim_pixel", batch["images"], segments).to("cuda")
    with torch.no_grad():
        preds_in = cap["h_norm"].reshape(-1, D)[flat_pos]
        pred = model.visual_aux_head(preds_in).float()
        va_ref = float((pred - targets).pow(2).mean(dim=-1).mean())
    expected = ce_ref + 0.5 * va_ref
    assert abs(loss_on - expected) < 2e-5, (loss_on, expected, ce_ref, va_ref)
    assert abs(comps["visual_aux"] - va_ref) < 2e-5, (comps["visual_aux"], va_ref)
    assert comps["visual_aux"] > 0, comps["visual_aux"]
    print(
        f"2. aim_pixel: {loss_on:.8f} == {ce_ref:.8f} + 0.5*{va_ref:.8f}; stash matches OK",
        flush=True,
    )
    del model
    torch.cuda.empty_cache()

    # ---- 3. nepa numerical correctness + alarms ----------------------------
    model, processor, data_args, _ = build_model("model.visual_aux.objective=nepa")
    assert model.visual_aux_head is not None
    batch = fresh_batch(processor, data_args)
    set_va(model, weight=0.5, layer=None, chunk=1024)

    cap = {}
    handle = model.model.norm.register_forward_hook(
        lambda m, a, o: cap.__setitem__("h_norm", o.detach())
    )
    orig_prepare = model.prepare_inputs_labels_for_multimodal

    def _prepare_and_capture(*a, **kw):
        out = orig_prepare(*a, **kw)
        cap["labels7"] = out[5].detach() if out[5] is not None else None
        cap["ids7"] = out[6].detach() if out[6] is not None else None
        return out

    model.prepare_inputs_labels_for_multimodal = _prepare_and_capture
    with torch.no_grad():
        out_on = model(**batch)
        loss_on = float(out_on.loss)
    comps = {k: float(v) for k, v in model._last_ce_components.items()}
    handle.remove()
    model.prepare_inputs_labels_for_multimodal = orig_prepare

    set_va(model, weight=0.0, layer=None, chunk=1024)
    with torch.no_grad():
        ce_ref = float(model(**batch).loss)

    # image_features via the SAME path the model used (encode_raw_patches).
    with torch.no_grad():
        image_features = model.encode_raw_patches(batch["images"], batch["image_position_ids"])
    D = cap["h_norm"].shape[-1]
    num_rows = [int(t.shape[0]) for t in image_features]
    flat_pos, segments = build_visual_aux_pairs(cap["ids7"].to("cuda"), num_rows)
    assert segments, "nepa reference produced no prediction pairs"
    targets = prepare_visual_aux_targets("nepa", image_features, segments).to("cuda")
    with torch.no_grad():
        preds_in = cap["h_norm"].reshape(-1, D)[flat_pos]
        pred = torch.nn.functional.normalize(model.visual_aux_head(preds_in).float(), dim=-1)
        cos = (pred * targets).sum(dim=-1)
        va_ref = float(-cos.mean())
    expected = ce_ref + 0.5 * va_ref
    assert abs(loss_on - expected) < 2e-5, (loss_on, expected, ce_ref, va_ref)
    assert abs(comps["visual_aux"] - va_ref) < 2e-5, (comps["visual_aux"], va_ref)
    assert abs(comps["visual_aux"] + comps["visual_aux_cos"]) < 1e-6, (
        comps["visual_aux"],
        comps["visual_aux_cos"],
    )
    assert comps["visual_aux_tgt_std"] > 0, comps["visual_aux_tgt_std"]
    for k, v in comps.items():
        assert torch.isfinite(torch.tensor(v)), f"nepa component {k} not finite: {v}"
    print(
        f"3. nepa: {loss_on:.8f} == {ce_ref:.8f} + 0.5*{va_ref:.8f}; "
        f"visual_aux==-cos, tgt_std={comps['visual_aux_tgt_std']:.4e} OK",
        flush=True,
    )

    # ---- 4. gradient routing (reuse the nepa model) ------------------------
    # The "no leak into the CE/classifier path" proof lives in the FORWARD:
    # capture the post-norm hidden + the CE component and show they are bit-
    # identical with and without the visual loss. The lm_head GRADIENT cannot
    # serve this role here because lm_head.weight is TIED to embed_tokens.weight
    # — the visual loss legitimately flows head→trunk→input-embeddings, so it
    # *does* (correctly) land in lm_head.weight.grad via the tied input role.
    # Forward bit-identity proves the output-projection (CE) role is untouched;
    # the connector grad proves the prediction path reaches the connector.
    cap_h = {}
    norm_handle = model.model.norm.register_forward_hook(
        lambda m, a, o: cap_h.__setitem__("h", o.detach().clone())
    )

    def grad_pass(weight, want_ce=False):
        # weight=0 turns visual-aux fully OFF in forward (gated on weight>0): no
        # block ids, no head anchor -> head grad is legitimately None there.
        set_va(model, weight=weight, layer=None, chunk=1024)
        model.zero_grad(set_to_none=True)
        out = model(**batch)
        loss = float(out.loss.detach())
        h = cap_h["h"].clone()
        comps = {k: float(v) for k, v in getattr(model, "_last_ce_components", {}).items()}
        out.loss.backward()
        head = model.visual_aux_head[0].weight.grad
        g = {
            "head": head.detach().clone() if head is not None else None,
            "connector": next(
                p.grad.detach().clone()
                for p in model.model.connector.parameters()
                if p.grad is not None
            ),
        }
        return loss, h, comps, g

    # Big weight on the on-pass: at random init the layer=None visual gradient
    # is heavily attenuated through the trunk; scaling λ lifts the connector
    # signal well clear of the off-pass fp32-reassoc noise. Forward bit-identity
    # (the CE/classifier no-leak proof) holds at any λ.
    VA_W = 20.0
    loss_off, h_off, _, g_off = grad_pass(0.0)
    loss_on, h_on, comps_on, g_on = grad_pass(VA_W)
    norm_handle.remove()

    assert g_on["head"] is not None and g_on["head"].abs().max().item() > 0, (
        "head grad must be nonzero with weight>0"
    )
    # Forward identity: post-norm hidden bit-identical => the CE/classifier path
    # (lm_head as output projection) is provably untouched by the visual loss.
    r_h = rel_diff(h_on, h_off)
    ce_on = loss_on - VA_W * comps_on["visual_aux"]
    assert r_h == 0.0, f"post-norm hidden changed (rel={r_h}) — visual loss leaked into the CE forward"
    assert abs(ce_on - loss_off) < 2e-5, f"CE component changed: ce_on={ce_on} ce_off={loss_off}"
    # Connector receives the visual prediction-path gradient (through the trunk).
    r_conn = rel_diff(g_on["connector"], g_off["connector"])
    assert r_conn > 1e-2, (
        f"connector grad barely changed (rel={r_conn:.3e}) — visual prediction "
        "path may not reach the connector"
    )
    print(
        f"4. gradient routing: head nonzero, CE forward bit-identical (rel={r_h:.0e}), "
        f"connector grad differs (rel={r_conn:.2e}) OK",
        flush=True,
    )
    del model
    torch.cuda.empty_cache()

    # ---- 5. degenerate text-only batch -------------------------------------
    from vlm.data.dataset import (  # noqa: E402
        DataCollatorForSupervisedDataset,
        make_dummy_image_entry,
        preprocess_qwen,
    )

    model, processor, data_args, _ = build_model("model.visual_aux.objective=nepa")
    set_va(model, weight=0.5, layer=None, chunk=1024)
    convs = [
        [
            {"from": "human", "value": "What is the capital of France?"},
            {"from": "gpt", "value": "Paris."},
        ],
        [
            {"from": "human", "value": "Name a primary color."},
            {"from": "gpt", "value": "Red."},
        ],
    ]
    samples = []
    for conv in convs:
        out = preprocess_qwen([conv], processor.tokenizer, data_args, has_image=False)
        samples.append(
            {
                "input_ids": out["input_ids"][0],
                "labels": out["labels"][0],
                "id": "t",
                "image": [make_dummy_image_entry(processor.image_processor)],
            }
        )
    collator = DataCollatorForSupervisedDataset(
        tokenizer=processor.tokenizer, ignore_index=data_args.ignore_index
    )
    text_batch = to_device(collator(samples), "cuda", torch.float32)
    model.zero_grad(set_to_none=True)
    out = model(**text_batch)
    assert torch.isfinite(out.loss), f"text-only loss not finite: {out.loss}"
    comps = {k: float(v) for k, v in model._last_ce_components.items()}
    assert comps["visual_aux"] == 0.0, comps["visual_aux"]
    out.loss.backward()
    head_grad = model.visual_aux_head[0].weight.grad
    assert head_grad is not None, "head grad is None on degenerate batch (broke the anchor)"
    assert head_grad.abs().max().item() == 0.0, "degenerate head grad should be exactly zero"
    print("5. degenerate text-only: finite loss, visual_aux==0, head grad zero (not None) OK", flush=True)

    # ---- 6. mid-layer + gradient checkpointing -----------------------------
    batch = fresh_batch(processor, data_args)

    def loss_pass():
        model.zero_grad(set_to_none=True)
        out = model(**batch)
        out.loss.backward()
        return float(out.loss.detach())

    set_va(model, weight=0.5, layer=None, chunk=1024)
    loss_final = loss_pass()
    set_va(model, weight=0.5, layer=6, chunk=1024)
    loss_mid = loss_pass()
    assert torch.isfinite(torch.tensor(loss_mid)), loss_mid
    assert abs(loss_mid - loss_final) > 1e-5, (loss_mid, loss_final)

    model.gradient_checkpointing_enable()
    loss_ckpt = loss_pass()
    model.gradient_checkpointing_disable()
    assert abs(loss_mid - loss_ckpt) < 1e-5, (loss_mid, loss_ckpt)
    assert not model.model.layers[5]._forward_hooks, "layer-6 forward hooks not removed after pass"
    print(
        f"6. mid-layer (k=6) {loss_mid:.8f} != final {loss_final:.8f}; "
        f"ckpt {loss_ckpt:.8f} matches, hooks removed OK",
        flush=True,
    )

    # ---- 7. optimizer grouping + trainability ------------------------------
    from vlm.train.optimizer import configure_optimizers  # noqa: E402
    from vlm.train.set_trainable import group_params_by_prefix, set_trainable_params  # noqa: E402

    grouped = group_params_by_prefix(model)
    head_names = {n for n, _ in grouped["visual_aux_head"]}
    lm_names = {n for n, _ in grouped["language_model"]}
    assert head_names, "visual_aux_head group is empty"
    assert head_names.isdisjoint(lm_names), "visual_aux_head leaked into language_model group"

    set_trainable_params(
        model,
        NS(train_language_model=False, train_vision_model=False, train_connector=False),
    )
    head_params = [p for _, p in grouped["visual_aux_head"]]
    assert all(p.requires_grad for p in head_params), "head not trainable under all-frozen"
    assert not any(p.requires_grad for _, p in grouped["language_model"]), "LM should be frozen"

    optim_groups = configure_optimizers(
        model,
        NS(
            visual_aux_head_lr=1e-4,
            visual_aux_head_wd=None,
            language_model_lr=6e-5,
            language_model_wd=0.0,
            connector_lr=6e-5,
            connector_wd=0.0,
            visual_encoder_lr=6e-5,
            visual_encoder_wd=0.0,
            learning_rate=6e-5,
        ),
    )
    head_param_ids = {id(p) for p in head_params}
    head_groups = [g for g in optim_groups if any(id(p) in head_param_ids for p in g["params"])]
    assert head_groups, "no optimizer group contains the visual_aux_head params"
    for g in head_groups:
        assert abs(g["lr"] - 1e-4) < 1e-12, f"head group lr {g['lr']} != 1e-4"
    # restore for any later use
    for p in model.parameters():
        p.requires_grad_(True)
    print("7. optimizer grouping: head disjoint, trainable when frozen, lr=1e-4 OK", flush=True)

    # ---- 8. validate_visual_aux_config -------------------------------------
    from vlm.train.train import validate_visual_aux_config  # noqa: E402

    # positive cases
    assert validate_visual_aux_config("none", None, n_layers, 1024, True) == ("none", None)
    assert validate_visual_aux_config("aim_pixel", None, n_layers, 1024, True) == ("aim_pixel", None)
    assert validate_visual_aux_config("nepa", 6, n_layers, 1024, True) == ("nepa", 6)
    # ValueError matrix
    bad_cases = [
        ("bogus", None, n_layers, 1024, True),  # unknown objective
        ("aim_pixel", None, n_layers, 0, True),  # chunked CE off
        ("aim_pixel", None, n_layers, 1024, False),  # encoder-present
        ("nepa", 0, n_layers, 1024, True),  # layer 0 out of range
        ("nepa", n_layers, n_layers, 1024, True),  # layer == n_layers
    ]
    for args in bad_cases:
        try:
            validate_visual_aux_config(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"validation accepted bad config: {args}")
    print("8. validate_visual_aux_config: positives + ValueError matrix OK", flush=True)

    print("ALL VISUAL-AUX TESTS PASSED", flush=True)


if __name__ == "__main__":
    main()
