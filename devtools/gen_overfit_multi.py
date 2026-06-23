"""N-sample overfit smoke for the text->image generation pathway.

Stronger correctness test than the single-sample version: can the model MEMORIZE
N distinct caption->image pairs (drive v-loss -> ~0) and then reconstruct each
sharply from its caption? If yes, the batched forward_generation + flow-matching
loss + left-padded multi-caption assembly + Euler sampler are all wired right
(a single-sample overfit can hide batch/padding/per-sample-mask bugs).

Trains full-batch (all N every step) for memorization, then samples all N at
cfg=1 and writes a target|generated comparison grid per sample.

Run (1 GPU):
  srun -p gpu-l40s -A krishna --gpus=l40s:1 -c 8 --mem=80G --time=1:30:00 \
      .venv/bin/python devtools/gen_overfit_multi.py --n 100 --steps 3000 --lr 1e-4 --bs 16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from PIL import Image

REPO = Path("/mmfs1/gscratch/krishna/leoym/small-vlm")
CONFIG_DIR = str(REPO / "src" / "vlm" / "config")
SAMPLE_DIR = Path("/mmfs1/gscratch/krishna/leoym/gen_overfit_data_100")
OUT_DIR = REPO / "outputs" / "gen-overfit-multi"


def load_samples(n: int):
    recs = [json.loads(l) for l in (SAMPLE_DIR / "samples.jsonl").read_text().splitlines() if l.strip()]
    return recs[:n]


def image_to_target_patches(img, resolution, patch_size, pixels_to_patches):
    img = img.resize((resolution, resolution), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    chw = torch.from_numpy(arr).permute(2, 0, 1).contiguous() * 2.0 - 1.0  # (3,H,W) [-1,1]
    return pixels_to_patches(chw, patch_size)  # (N, patch_dim)


def patches_to_uint8(patches, grid, patch_size, patches_to_pixels):
    chw = patches_to_pixels(patches.float().cpu(), grid, grid, patch_size)
    chw = ((chw.clamp(-1, 1) + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return chw.permute(1, 2, 0).numpy()  # (H,W,3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--sample_steps", type=int, default=100)
    ap.add_argument("--n_show", type=int, default=16, help="how many to render in the grid")
    ap.add_argument("--config", default="gen-overfit", help="hydra config name (gen-overfit | gen-overfit-16)")
    ap.add_argument("--tag", default="", help="output subdir suffix, e.g. -16")
    ap.add_argument("--render_at", default="", help="space-separated step counts to render at, e.g. '500 1000'")
    args = ap.parse_args()

    out_dir = Path(str(OUT_DIR) + args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[overfit-multi] device={device} n={args.n} steps={args.steps} bs={args.bs}", flush=True)

    from vlm.config.config_schema import register_configs
    from vlm.models.gen_image import make_position_ids, patches_to_pixels, pixels_to_patches
    from vlm.vlm import load_model

    register_configs()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name=args.config)

    t0 = time.time()
    model, processor = load_model(cfg.model, cfg.trainer)
    model = model.to(device)
    model.train()
    model.requires_grad_(True)
    model.config.use_cache = False
    print(f"[overfit-multi] model built in {time.time()-t0:.1f}s", flush=True)
    # self-report whether the shared connector / gen embedder built a bottleneck
    import torch.nn as _nn
    _emb = getattr(model, "gen_patch_embed", None)
    _conn = getattr(getattr(model, "model", None), "connector", None)
    _pl = getattr(_conn, "projection_layer", None)
    _which = _emb if _emb is not None else _pl
    _pd = getattr(_which, "patch_dense", None)
    if _pd is not None:
        kind = "Sequential(BOTTLENECK)" if isinstance(_pd, _nn.Sequential) else "Linear(no-bottleneck)"
        dims = [getattr(m, "out_features", "?") for m in _pd] if isinstance(_pd, _nn.Sequential) else getattr(_pd, "out_features", "?")
        print(f"[overfit-multi] patch_dense({'gen_embed' if _emb is not None else 'connector'}) = {kind} dims={dims}", flush=True)

    res = int(cfg.model.generation.resolution)
    independent = bool(cfg.model.generation.independent_embed)
    psz = int(cfg.model.generation.embed_patch_size) if independent else int(cfg.model.generation.patch_size)
    grid = res // psz
    n_patch = grid * grid
    print(f"[overfit-multi] independent_embed={independent} res={res} patch={psz} "
          f"grid={grid}x{grid} n_patch={n_patch} patch_dim={psz*psz*3}", flush=True)

    recs = load_samples(args.n)
    captions = [r["caption"] for r in recs]
    targets = torch.stack([
        image_to_target_patches(Image.open(SAMPLE_DIR / r["image"]).convert("RGB"), res, psz, pixels_to_patches)
        for r in recs
    ]).to(device)  # (N, n_patch, patch_dim)
    pos_one = make_position_ids(grid, grid).to(device)  # (n_patch, 2)
    n = len(recs)
    print(f"[overfit-multi] loaded {n} samples  targets={tuple(targets.shape)} grid={grid}x{grid}", flush=True)

    # Left-padded caption batch (matches training assembly / collator).
    processor.tokenizer.padding_side = "left"

    def tok_batch(idx):
        enc = processor.tokenizer([captions[i] for i in idx], return_tensors="pt",
                                  padding=True, truncation=True, max_length=128)
        return enc.input_ids.to(device), enc.attention_mask.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    torch.manual_seed(0)
    g = torch.Generator(device="cpu").manual_seed(0)
    show = min(args.n_show, n)
    render_at = {int(s) for s in args.render_at.split()} if args.render_at else set()

    def render(tag: str) -> None:
        """Sample the first `show` captions and save a target|generated grid."""
        was_training = model.training
        model.eval()
        ids_, attn_ = tok_batch(list(range(show)))
        pos_ = pos_one.unsqueeze(0).expand(show, -1, -1)
        t0 = time.time()
        gen = model.sample_images(input_ids=ids_, attention_mask=attn_, image_position_ids=pos_,
                                  num_patches=n_patch, steps=args.sample_steps, cfg_scale=1.0)
        pad, cell = 4, res
        rowH = cell + pad
        canvas = np.full((rowH * show, cell * 2 + pad, 3), 255, dtype=np.uint8)
        for i in range(show):
            tgt = patches_to_uint8(targets[i], grid, psz, patches_to_pixels)
            out_img = patches_to_uint8(gen[i], grid, psz, patches_to_pixels)
            y = i * rowH
            canvas[y:y + cell, :cell] = tgt
            canvas[y:y + cell, cell + pad:cell + pad + cell] = out_img
        path = out_dir / f"compare_grid_{tag}.png"
        Image.fromarray(canvas).save(path)
        print(f"[overfit-multi] [{tag}] sampled {show} in {time.time()-t0:.1f}s -> {path}", flush=True)
        if was_training:
            model.train()

    losses = []
    torch.cuda.synchronize() if device.type == "cuda" else None
    t_train = time.time()
    for step in range(args.steps):
        perm = torch.randperm(n, generator=g).tolist()
        idx = perm[: args.bs]
        input_ids, attn = tok_batch(idx)
        pos = pos_one.unsqueeze(0).expand(len(idx), -1, -1)
        out = model(input_ids=input_ids, attention_mask=attn,
                    target_patches=targets[idx], image_position_ids=pos)
        loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if step % args.log_every == 0 or step == args.steps - 1:
            recent = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            comps = getattr(model, "_last_ce_components", {}) or {}
            extra = "  ".join(f"{k} {float(v):.4f}" for k, v in comps.items())
            print(f"[overfit-multi] step {step:4d}  loss {losses[-1]:.5f}  avg{args.log_every} {recent:.5f}  {extra}", flush=True)
        if (step + 1) in render_at:
            render(f"step{step+1:04d}")

    torch.cuda.synchronize() if device.type == "cuda" else None
    step_sec = (time.time() - t_train) / args.steps
    print(f"[overfit-multi] first={losses[0]:.5f} last={losses[-1]:.5f} min={min(losses):.5f} "
          f"step_sec={step_sec:.3f} ({args.steps} steps)", flush=True)

    render("final")
    (out_dir / "loss.json").write_text(json.dumps(
        {"losses": losses, "captions": captions[:show], "step_sec": step_sec,
         "independent_embed": independent, "patch": psz, "n_patch": n_patch}))
    print(f"[overfit-multi] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
