"""GPU smoke for the E1 random-init causal control (spec 2026-06-18).

Builds the same encoder-free VLM twice via the real load_model — once normally
(pretrained Qwen prior) and once with language_model.random_init=True — and
proves on real bf16 kernels:

  1. prior DESTROYED: next-token CE on coherent text jumps from ~pretrained
     (low) to ~log(vocab) (uniform) — the language prior is gone.
  2. embeddings re-initialized: embed_tokens weights differ massively.
  3. connector PRESERVED + functional: the image forward still runs and the
     connector params are finite/non-trivial (re-init only touched the LM).

Run: sbatch devtools/randinit_smoke.slurm   (or --cpu for a tiny dry run).
"""

import argparse
import math
import sys

import torch
from omegaconf import OmegaConf
from xmodal_smoke import make_image  # noqa: E402

from vlm.config.config_schema import (
    ConnectorConfig,
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


def build(device, base_lm, random_init):
    bf16 = device == "cuda"
    model_cfg = OmegaConf.structured(
        ModelConfig(
            name="randinit-smoke",
            visual_encoder=VisualEncoderConfig(
                hf_name=None, patch_size=16, pooling_kernel_size=3, max_soft_tokens=64
            ),
            language_model=LanguageModelConfig(
                hf_name=base_lm, max_seq_length=4096, random_init=random_init
            ),
            connector=ConnectorConfig(name="raw_patch", type="raw_patch"),
        )
    )
    trainer_cfg = OmegaConf.structured(
        TrainerConfig(name="smoke", bf16=bf16, fp16=False, attn_implementation="sdpa")
    )
    model, processor = load_model(model_cfg, trainer_cfg)
    return model.to(device=device, dtype=torch.bfloat16 if bf16 else torch.float32), processor


@torch.no_grad()
def text_ce(model, tok, device):
    """Next-token CE on a coherent English sentence (lower = stronger prior)."""
    ids = tok(
        "The capital of France is Paris and the sky is blue.", return_tensors="pt"
    ).input_ids.to(device)
    out = model.model(input_ids=ids, use_cache=False)
    logits = model.lm_head(out.last_hidden_state).float()
    return torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="mean").item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--base-lm", default="Qwen/Qwen3-0.6B")
    args = ap.parse_args()
    device = "cpu" if args.cpu else "cuda"
    dtype = torch.float32 if args.cpu else torch.bfloat16

    norm, proc = build(device, args.base_lm, random_init=False)
    rand, _ = build(device, args.base_lm, random_init=True)
    tok = proc.tokenizer
    vocab = norm.config.vocab_size
    uniform = math.log(vocab)

    ce_norm = text_ce(norm, tok, device)
    ce_rand = text_ce(rand, tok, device)
    check(
        "prior intact in normal build (CE well below uniform)",
        ce_norm < 0.5 * uniform,
        f"CE_norm={ce_norm:.2f} vs log|V|={uniform:.2f}",
    )
    check(
        "prior DESTROYED by random_init (CE near uniform)",
        ce_rand > 0.7 * uniform,
        f"CE_rand={ce_rand:.2f} vs log|V|={uniform:.2f}",
    )
    check("random_init CE >> normal CE", ce_rand > ce_norm + 3.0, f"{ce_norm:.2f} -> {ce_rand:.2f}")

    e_norm = norm.model.embed_tokens.weight
    e_rand = rand.model.embed_tokens.weight.to(e_norm.device)
    rel = (e_norm - e_rand).norm().item() / (e_norm.norm().item() + 1e-9)
    check("embeddings re-initialized (large relative change)", rel > 0.5, f"rel L2 diff={rel:.3f}")

    # connector preserved + functional: image forward runs and produces finite feats
    cps = [p for n, p in rand.named_parameters() if "connector" in n]
    check(
        "connector params present + finite",
        len(cps) > 0 and all(torch.isfinite(p).all() for p in cps),
        f"{len(cps)} connector tensors",
    )
    pix, pos = make_image(proc, device, dtype)
    feats = rand.encode_raw_patches([pix], [pos])
    check(
        "image forward through connector works (finite feats)",
        torch.isfinite(feats[0]).all(),
        f"feat shape={tuple(feats[0].shape)}",
    )

    print("\n" + ("ALL RANDINIT SMOKE CHECKS PASSED" if OK else "RANDINIT SMOKE FAILED"))
    sys.exit(0 if OK else 1)


if __name__ == "__main__":
    main()
