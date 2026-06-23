"""Single-sample overfit smoke for the text->image generation pathway.

Validates the whole pipeline end-to-end on ONE GPIC sample: build the model
(encoder-free Qwen3-1.7B + generation head/t-token), drive the flow-matching
v-loss -> ~0 with a manual loop, then run the Euler sampler and save the
generated image next to the target. If loss falls to near-zero and the sample
visually matches the target, the generation forward + loss + sampler + data
handling are wired correctly.

Run (1 GPU):
  srun -p gpu-l40s -A krishna --gpus=l40s:1 -c 8 --mem=64G --time=0:40:00 \
      .venv/bin/python devtools/gen_overfit.py --steps 800 --lr 2e-4
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
SAMPLE_DIR = Path("/mmfs1/gscratch/krishna/leoym/gen_overfit_data")  # shared (not node-local /scr)
OUT_DIR = REPO / "outputs" / "gen-overfit"


def load_one_sample():
    rec = json.loads((SAMPLE_DIR / "samples.jsonl").read_text().splitlines()[0])
    img = Image.open(SAMPLE_DIR / rec["image"]).convert("RGB")
    return img, rec["caption"], rec


def image_to_target_patches(img: Image.Image, resolution: int, patch_size: int, pixels_to_patches):
    img = img.resize((resolution, resolution), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3) in [0,1]
    chw = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,H,W)
    chw = chw * 2.0 - 1.0  # [-1,1]
    return pixels_to_patches(chw, patch_size)  # (N, patch_dim)


def save_image_from_patches(patches, grid, patch_size, path, patches_to_pixels):
    chw = patches_to_pixels(patches.float().cpu(), grid, grid, patch_size)  # (3,H,W) [-1,1]
    chw = ((chw.clamp(-1, 1) + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    arr = chw.permute(1, 2, 0).numpy()  # (H,W,3)
    Image.fromarray(arr).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--sample_steps", type=int, default=100)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[overfit] device={device}", flush=True)

    from vlm.config.config_schema import register_configs
    from vlm.models.gen_image import make_position_ids, patches_to_pixels, pixels_to_patches
    from vlm.vlm import load_model

    register_configs()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="gen-overfit")

    t0 = time.time()
    model, processor = load_model(cfg.model, cfg.trainer)
    model = model.to(device)
    model.train()
    model.requires_grad_(True)
    model.config.use_cache = False
    print(f"[overfit] model built in {time.time()-t0:.1f}s", flush=True)

    res = int(cfg.model.generation.resolution)
    psz = int(cfg.model.generation.patch_size)
    grid = res // psz
    n_patch = grid * grid
    print(f"[overfit] resolution={res} patch={psz} grid={grid}x{grid} n_patch={n_patch} "
          f"patch_dim={model.config.vision_config.hidden_size}", flush=True)

    # ---- single sample ----
    img, caption, rec = load_one_sample()
    print(f"[overfit] caption: {caption[:120]}", flush=True)
    target_patches = image_to_target_patches(img, res, psz, pixels_to_patches).unsqueeze(0).to(device)
    pos = make_position_ids(grid, grid).unsqueeze(0).to(device)  # (1,N,2)
    tok = processor.tokenizer(caption, return_tensors="pt")
    input_ids = tok.input_ids.to(device)
    attn = tok.attention_mask.to(device)
    print(f"[overfit] target_patches={tuple(target_patches.shape)} input_ids={tuple(input_ids.shape)}",
          flush=True)
    # reference: round-trip the target through unpatchify so we can eyeball the
    # best achievable reconstruction at this resolution/patch size.
    save_image_from_patches(target_patches[0], grid, psz, OUT_DIR / "target.png", patches_to_pixels)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    torch.manual_seed(0)
    losses = []
    for step in range(args.steps):
        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            target_patches=target_patches,
            image_position_ids=pos,
        )
        loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"[overfit] step {step:4d}  loss {losses[-1]:.5f}", flush=True)

    print(f"[overfit] first={losses[0]:.5f} last={losses[-1]:.5f} "
          f"min={min(losses):.5f}", flush=True)

    # ---- sample (reconstruct) ----
    model.eval()
    t0 = time.time()
    patches = model.sample_images(
        input_ids=input_ids,
        attention_mask=attn,
        image_position_ids=pos,
        num_patches=n_patch,
        steps=args.sample_steps,
        cfg_scale=1.0,
    )
    print(f"[overfit] sampled in {time.time()-t0:.1f}s", flush=True)
    save_image_from_patches(patches[0], grid, psz, OUT_DIR / "generated.png", patches_to_pixels)
    (OUT_DIR / "loss.json").write_text(json.dumps({"losses": losses, "caption": caption}))
    print(f"[overfit] saved target/generated to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
