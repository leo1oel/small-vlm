"""Text->image inference from a trained generation checkpoint.

Loads a checkpoint (reload path rebuilds the generation head/t-token from the
serialized config and loads their trained weights), runs the Euler
flow-matching sampler for each caption, and saves a PNG grid.

Run (1 GPU):
  srun -p gpu-l40s -A krishna --gpus=l40s:1 -c 8 --mem=64G --time=0:30:00 \
      .venv/bin/python devtools/gen_infer.py \
      --ckpt outputs/gen-gpic/checkpoint-XXXX \
      --captions "a red bicycle leaning on a brick wall" "a bowl of ramen on a wooden table" \
      --cfg 6.0 --steps 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--captions", nargs="+", required=True)
    ap.add_argument("--cfg", type=float, default=None, help="override generation_cfg_scale")
    ap.add_argument("--steps", type=int, default=None, help="override sample_steps")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from vlm.inference.eval import load_model
    from vlm.models.gen_image import make_position_ids, patches_to_pixels

    model, processor, _ = load_model(args.ckpt)
    model = model.to(device).eval()
    cfg = model.config
    assert bool(getattr(cfg, "generation", False)), "checkpoint has no generation pathway"

    res = int(cfg.generation_resolution)
    psz = int(cfg.generation_patch_size)
    grid = res // psz
    n_patch = grid * grid
    cfg_scale = float(args.cfg if args.cfg is not None else cfg.generation_cfg_scale)
    steps = int(args.steps if args.steps is not None else cfg.generation_sample_steps)
    out_dir = Path(args.out) if args.out else Path(args.ckpt) / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[infer] res={res} grid={grid}x{grid} n_patch={n_patch} cfg={cfg_scale} steps={steps}",
        flush=True,
    )

    # Left-pad the batch of captions (matches training assembly).
    processor.tokenizer.padding_side = "left"
    enc = processor.tokenizer(
        list(args.captions), return_tensors="pt", padding=True, truncation=True, max_length=128
    )
    input_ids = enc.input_ids.to(device)
    attn = enc.attention_mask.to(device)
    bsz = input_ids.shape[0]
    pos = make_position_ids(grid, grid).unsqueeze(0).expand(bsz, -1, -1).to(device)

    patches = model.sample_images(
        input_ids=input_ids,
        attention_mask=attn,
        image_position_ids=pos,
        num_patches=n_patch,
        steps=steps,
        cfg_scale=cfg_scale,
    )  # (B, N, patch_dim)

    for i, caption in enumerate(args.captions):
        chw = patches_to_pixels(patches[i].float().cpu(), grid, grid, psz)  # (3,H,W) ~[-1,1]
        chw = ((chw.clamp(-1, 1) + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
        img = Image.fromarray(chw.permute(1, 2, 0).numpy())
        path = out_dir / f"sample_{i:02d}.png"
        img.save(path)
        print(f"[infer] {path}  <- {caption[:80]}", flush=True)


if __name__ == "__main__":
    main()
