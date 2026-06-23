"""GPU smoke test: real inference on a trained encoder-free unified checkpoint.

Usage: python devtools/infer_trial.py <checkpoint_dir>

Loads the checkpoint through the production inference path
(vlm.inference.load_model / generate_response) and runs a battery of
single-image, multi-image, interleaved-placeholder and text-only generations.
"""

import sys
import time

import torch

from vlm.inference.eval import generate_response, load_model

CKPT = sys.argv[1]

print(f"torch {torch.__version__} | cuda: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)

t0 = time.time()
model, processor, info = load_model(CKPT, bf16=True, attn_implementation="sdpa")
print(f"loaded in {time.time() - t0:.1f}s", flush=True)
print(f"info: {info}", flush=True)
print(f"model dtype: {next(model.parameters()).dtype} | device: {model.device}", flush=True)

CASES: list[tuple[str, list[str] | None]] = [
    # plain single-image VQA (placeholder auto-prepended)
    ("Describe this image in detail.", ["example/image.jpg"]),
    ("What is unusual about this image?", ["example/image.jpg"]),
    ("How many people are in this image? Answer with a number.", ["example/image1.jpg"]),
    # explicit placeholder position
    ("<image>\nWhat colors dominate this picture?", ["example/image1.jpg"]),
    # multi-image, interleaved placeholders
    (
        "<image>\nThis is the first image. <image>\nThis is the second image. "
        "What is the main difference between them?",
        ["example/image.jpg", "example/image1.jpg"],
    ),
    # text-only (no media -> plain embedding path)
    ("Hello! Please introduce yourself in one sentence.", None),
]

for i, (query, images) in enumerate(CASES, 1):
    t0 = time.time()
    out = generate_response(
        model,
        processor,
        query=query,
        images=images,
        max_new_tokens=128,
    )
    print("=" * 80, flush=True)
    print(f"[case {i}] query={query!r}", flush=True)
    print(f"[case {i}] images={images}", flush=True)
    print(f"[case {i}] response ({time.time() - t0:.1f}s):\n{out}", flush=True)

print("=" * 80, flush=True)
print("ALL INFERENCE CASES COMPLETED", flush=True)
