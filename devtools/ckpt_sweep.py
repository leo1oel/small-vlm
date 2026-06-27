"""Sweep a fixed prompt battery over multiple checkpoints (one GPU job).

Usage: python devtools/ckpt_sweep.py <ckpt_dir> [<ckpt_dir> ...]

Per checkpoint: load -> run all cases -> free GPU memory. The battery mixes
generic VQA prompts with real training-jsonl samples (ground truth shown for
comparison). sft-clip checkpoints trained before image_aspect_ratio was
recorded in the config get aspect="square" (their training value) explicitly.
"""

import gc
import json
import sys
import time

import torch

from vlm.inference.eval import generate_response, load_model

CKPTS = sys.argv[1:]

# (query, images, ground_truth | None)
CASES: list[tuple[str, list[str] | None, str | None]] = [
    ("Describe this image in detail.", ["example/image.jpg"], None),
    ("How many people are in this image? Answer with a number.", ["example/image1.jpg"], None),
    ("<image>\nWhat colors dominate this picture?", ["example/image1.jpg"], None),
]
for s in json.load(open("outputs/infer_demo/samples.json")):
    convs = s["record"]["messages"]
    # first user turn text (image placeholder will be auto-prepended)
    first_user = next(m for m in convs if m["role"] == "user")
    if isinstance(first_user["content"], list):
        text = "\n".join(it["text"] for it in first_user["content"] if it.get("type") == "text")
    else:
        text = first_user["content"]
    first_gt = next(m for m in convs if m["role"] == "assistant")
    gt = first_gt["content"] if isinstance(first_gt["content"], str) else ""
    CASES.append((text, [s["image_local"]], gt))

for ckpt in CKPTS:
    print("#" * 100, flush=True)
    print(f"CHECKPOINT: {ckpt}", flush=True)
    t0 = time.time()
    model, processor, info = load_model(ckpt, bf16=True, attn_implementation="sdpa")
    # sft-clip trained with dataset.image_aspect_ratio=square before the
    # config began recording it; explicit override keeps train/infer parity.
    aspect = None
    if not info["encoder_free"] and getattr(model.config, "image_aspect_ratio", None) is None:
        aspect = "square"
    print(
        f"loaded in {time.time() - t0:.0f}s | encoder_free={info['encoder_free']} "
        f"| conv={info['conversation_version']} | aspect={aspect or 'from-config/pad'}",
        flush=True,
    )

    for qi, (query, images, gt) in enumerate(CASES, 1):
        t0 = time.time()
        try:
            answer = generate_response(
                model,
                processor,
                query=query,
                images=images,
                max_new_tokens=96,
                image_aspect_ratio=aspect,
            )
        except Exception as e:  # keep sweeping; report the failure
            answer = f"<ERROR {type(e).__name__}: {e}>"
        print(f"--- case {qi} ({time.time() - t0:.1f}s) Q: {query[:90]!r}", flush=True)
        print(f"    A: {answer}", flush=True)
        if gt:
            print(f"    GT: {gt[:220]}", flush=True)

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

print("SWEEP COMPLETED", flush=True)
