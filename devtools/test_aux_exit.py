"""Tests for the auxiliary intermediate-layer CE loss (aux-exit, early-fusion
ablation). Spec: docs/superpowers/specs/2026-06-05-aux-exit-loss-design.md.

Run on a GPU node (from the repo or worktree root):
    srun -p ckpt-all -A cse-ckpt --gpus=l40:1 --mem=48G --time=0:30:00 \
        bash -c 'source /mmfs1/gscratch/krishna/leoym/small-vlm/.venv/bin/activate \
                 && HF_HUB_OFFLINE=1 python devtools/test_aux_exit.py'

Checks (fp32+sdpa, real multimodal batch from test_chunked_ce.build_batch):
  0. sft-unified-earlyfusion composes; aux fields land in trainer config.
  1. Baseline parity: aux disabled -> chunked loss/grads still match the
     full-logits reference (protects the live sft-unified baseline).
  2. Aux numerical correctness: chunked aux-on loss == L_final_ref +
     lambda * naive full-vocab CE at layer k (norm module + lm_head on the
     captured spliced hidden states), and the logged components match.
  3. _rms_norm functional replica == Qwen3RMSNorm module (fp32 and bf16).
  4. Gradient routing: layers above k and (under detach) the shared norm
     get bit-identical grads vs aux-off; layers <= k get different grads.
  5. Gradient checkpointing (non-reentrant default): same aux-on loss,
     backward works, captured tensors are graph-connected.
  6. validate_aux_exit_config rejects k<1, k>=num_layers, chunked CE off.
"""

import sys
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

AUX_K = 6  # exit after the 6th decoder layer (1-indexed), matches the run config


def set_aux(model, layers, weight=0.0, detach=False, chunk=1024):
    model.config.loss_chunk_size = chunk
    model.config.aux_exit_layers = layers
    model.config.aux_exit_weight = weight
    model.config.aux_exit_detach = detach


def run_pass(model, batch):
    model.zero_grad(set_to_none=True)
    out = model(**batch)
    out.loss.backward()
    grads = {
        "upper.q": model.model.layers[20].self_attn.q_proj.weight.grad.detach().clone(),
        "lower.q": model.model.layers[2].self_attn.q_proj.weight.grad.detach().clone(),
        "norm.w": model.model.norm.weight.grad.detach().clone(),
    }
    return float(out.loss.detach()), grads


def rel_diff(a, b):
    return (a - b).abs().max().item() / max(a.abs().max().item(), 1e-12)


def main():
    assert torch.cuda.is_available(), "needs a GPU node"
    register_configs()

    # ---- 0. config compose -------------------------------------------------
    with initialize_config_dir(config_dir=str(REPO / "src/vlm/config"), version_base=None):
        cfg_ef = compose(config_name="sft-unified-earlyfusion")
    assert list(cfg_ef.trainer.aux_exit_layers) == [6], cfg_ef.trainer.aux_exit_layers
    assert abs(cfg_ef.trainer.aux_exit_weight - 0.25) < 1e-9
    assert cfg_ef.trainer.aux_exit_detach is False
    assert cfg_ef.trainer.loss_chunk_size == 1024  # inherited from sft-unified
    print("0. earlyfusion config composes with aux fields: OK", flush=True)

    # ---- model + batch (fp32 + sdpa, strict) -------------------------------
    with initialize_config_dir(config_dir=str(REPO / "src/vlm/config"), version_base=None):
        cfg = compose(
            config_name="sft-unified",
            overrides=[
                "model=qwen3-0.6b-unified",
                "trainer.bf16=false",
                "trainer.attn_implementation=sdpa",
            ],
        )
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
    batch = to_device(build_batch(processor, data_args), "cuda", torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    n_layers = model.config.num_hidden_layers

    # ---- 1. baseline parity (aux disabled == full-logits reference) --------
    set_aux(model, layers=[], chunk=0)
    loss_full, g_full = run_pass(model, batch)
    set_aux(model, layers=[], chunk=1024)
    loss_off, g_off = run_pass(model, batch)
    assert abs(loss_full - loss_off) < 1e-5, (loss_full, loss_off)
    for k in g_full:
        r = rel_diff(g_full[k], g_off[k])
        assert r < 1e-4, f"baseline grad[{k}] rel={r}"
    print(
        f"1. baseline parity (aux off): loss {loss_off:.8f} == full {loss_full:.8f} OK", flush=True
    )

    # ---- 2. aux numerical correctness vs naive reference -------------------
    # Capture the spliced layer-k hidden states AND the spliced labels from a
    # plain full-logits pass, then compute the reference aux CE with the real
    # norm module + lm_head over the full vocab.
    cap = {}
    handle = model.model.layers[AUX_K - 1].register_forward_hook(
        lambda m, a, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach())
    )
    orig_prepare = model.prepare_inputs_labels_for_multimodal

    def _prepare_and_capture(*a, **kw):
        out = orig_prepare(*a, **kw)
        cap["labels"] = out[5].detach()
        return out

    model.prepare_inputs_labels_for_multimodal = _prepare_and_capture
    set_aux(model, layers=[], chunk=0)
    with torch.no_grad():
        loss_final_ref = float(model(**batch).loss)
    handle.remove()
    model.prepare_inputs_labels_for_multimodal = orig_prepare

    shift = torch.nn.functional.pad(cap["labels"], (0, 1), value=-100)[..., 1:]
    flat_t = shift.reshape(-1)
    valid = flat_t != -100
    h_valid = cap["h"].reshape(-1, cap["h"].shape[-1])[valid]
    with torch.no_grad():
        logits_ref = model.lm_head(model.model.norm(h_valid)).float()
        aux_ref = float(
            torch.nn.functional.cross_entropy(logits_ref, flat_t[valid], reduction="mean")
        )
    lam = 0.3
    set_aux(model, layers=[AUX_K], weight=lam, chunk=1024)
    with torch.no_grad():
        loss_aux_on = float(model(**batch).loss)
    expected = loss_final_ref + lam * aux_ref
    assert abs(loss_aux_on - expected) < 2e-5, (loss_aux_on, expected)
    comps = model._last_ce_components
    assert abs(float(comps["ce_final"]) - loss_final_ref) < 2e-5
    assert abs(float(comps["ce_aux"]) - aux_ref) < 2e-5
    assert aux_ref > loss_final_ref, "layer-k CE should exceed final CE at init"
    print(
        f"2. aux correctness: {loss_aux_on:.8f} == {loss_final_ref:.8f} + "
        f"{lam}*{aux_ref:.8f} OK (components logged)",
        flush=True,
    )

    # ---- 3. functional RMSNorm replica -------------------------------------
    from vlm.models.modeling_vlm import _rms_norm  # noqa: E402

    norm = model.model.norm
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(64, model.config.hidden_size, device="cuda", dtype=dtype)
        a = norm(x)
        b = _rms_norm(x, norm.weight, norm.variance_epsilon)
        assert torch.equal(a, b), f"_rms_norm mismatch in {dtype}"
    print("3. _rms_norm == Qwen3RMSNorm (fp32, bf16): OK", flush=True)

    # ---- 4. gradient routing ------------------------------------------------
    set_aux(model, layers=[], chunk=1024)
    _, g0 = run_pass(model, batch)
    set_aux(model, layers=[AUX_K], weight=0.5, detach=False, chunk=1024)
    _, g1 = run_pass(model, batch)
    set_aux(model, layers=[AUX_K], weight=0.5, detach=True, chunk=1024)
    _, g2 = run_pass(model, batch)

    assert rel_diff(g1["upper.q"], g0["upper.q"]) < 1e-5, "aux leaked above layer k"
    assert rel_diff(g1["lower.q"], g0["lower.q"]) > 1e-3, "aux did not reach layers <= k"
    assert rel_diff(g1["norm.w"], g0["norm.w"]) > 1e-3, "shared-norm grad unchanged (no detach)"
    assert rel_diff(g2["upper.q"], g0["upper.q"]) < 1e-5, "detach: aux leaked above layer k"
    assert rel_diff(g2["norm.w"], g0["norm.w"]) < 1e-5, "detach: aux still flowed into norm"
    assert rel_diff(g2["lower.q"], g0["lower.q"]) > 1e-3, "detach: aux did not reach layers <= k"
    print("4. gradient routing (incl. detach fuse): OK", flush=True)

    # ---- 5. gradient checkpointing ------------------------------------------
    set_aux(model, layers=[AUX_K], weight=0.5, detach=False, chunk=1024)
    loss_nockpt, g_nockpt = run_pass(model, batch)
    model.gradient_checkpointing_enable()  # transformers 5.x: use_reentrant=False
    loss_ckpt, g_ckpt = run_pass(model, batch)
    model.gradient_checkpointing_disable()
    assert abs(loss_nockpt - loss_ckpt) < 1e-5, (loss_nockpt, loss_ckpt)
    r = rel_diff(g_nockpt["lower.q"], g_ckpt["lower.q"])
    assert r < 1e-4, f"aux grads under checkpointing diverge: rel={r}"
    print(f"5. gradient checkpointing: loss {loss_ckpt:.8f} == {loss_nockpt:.8f} OK", flush=True)

    # ---- 6. config validation ------------------------------------------------
    from vlm.train.train import validate_aux_exit_config  # noqa: E402

    validate_aux_exit_config([6], num_hidden_layers=n_layers, loss_chunk_size=1024)  # ok
    for bad_layers, chunk in (([0], 1024), ([n_layers], 1024), ([99], 1024), ([6], 0)):
        try:
            validate_aux_exit_config(bad_layers, num_hidden_layers=n_layers, loss_chunk_size=chunk)
        except ValueError:
            pass
        else:
            raise AssertionError(f"validation accepted layers={bad_layers}, chunk={chunk}")
    print("6. validate_aux_exit_config rejects bad configs: OK", flush=True)

    print("ALL AUX-EXIT TESTS PASSED", flush=True)


if __name__ == "__main__":
    main()
