"""Parity tests for chunked_ce_forward vs the inherited full-logits loss.

Run on a GPU node:
    srun -p ckpt-all -A cse-ckpt --gpus=l40:1 --mem=48G --time=0:20:00 \
        bash -c 'source .venv/bin/activate && HF_HUB_OFFLINE=1 python devtools/test_chunked_ce.py'

Checks (each on a real multimodal batch with images, padding and -100s):
  1. fp32+sdpa strict loss parity (|d| < 1e-5) and gradient parity on the
     tied lm_head/embedding, a decoder layer, and the image connector.
  2. chunk-size invariance (256 vs 1024 identical within fp32 tolerance).
  3. bf16+FA2 loose loss parity (rel < 1e-2) + grad cosine > 0.999.
  4. peak-memory comparison at a long-sequence batch (chunked << full).
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hydra import compose, initialize_config_dir  # noqa: E402

from vlm.config import register_configs  # noqa: E402
from vlm.data.data_arguments import DataArguments  # noqa: E402
from vlm.data.dataset import (  # noqa: E402
    DataCollatorForSupervisedDataset,
    preprocess_qwen,
    process_raw_image,
)
from vlm.utils import conversation as conversation_lib  # noqa: E402
from vlm.vlm import load_model  # noqa: E402


def build_batch(processor, data_args, seq_pad: int = 0):
    """Real multimodal batch: 3 samples with different lengths (=> padding),
    one text-only (dummy image), one 1-image, one 2-image conversation."""
    rng = np.random.default_rng(0)

    def img():
        return Image.fromarray(
            rng.integers(0, 255, (224 + int(rng.integers(0, 200)), 320, 3), dtype=np.uint8)
        )

    convs = [
        [
            {
                "from": "human",
                "value": "<image>\nDescribe the image." + " More detail please." * seq_pad,
            },
            {"from": "gpt", "value": "A colorful noise pattern." + " It is random." * seq_pad},
        ],
        [
            {"from": "human", "value": "What is the capital of France?"},
            {"from": "gpt", "value": "Paris."},
        ],
        [
            {"from": "human", "value": "<image>\nCompare with this: <image>"},
            {"from": "gpt", "value": "Both are noise; the second is wider." + " Indeed." * seq_pad},
        ],
    ]
    n_images = [1, 0, 2]
    samples = []
    for conv, n in zip(convs, n_images, strict=True):
        out = preprocess_qwen([conv], processor.tokenizer, data_args, has_image=n > 0)
        d = {"input_ids": out["input_ids"][0], "labels": out["labels"][0], "id": "t"}
        if n > 0:
            d["image"] = [process_raw_image(img(), processor.image_processor) for _ in range(n)]
        else:
            from vlm.data.dataset import make_dummy_image_entry

            d["image"] = [make_dummy_image_entry(processor.image_processor)]
        samples.append(d)
    collator = DataCollatorForSupervisedDataset(
        tokenizer=processor.tokenizer, ignore_index=data_args.ignore_index
    )
    return collator(samples)


def to_device(batch, device, dtype):
    out = {}
    for k, v in batch.items():
        if k == "images":
            out[k] = [
                tuple(x.to(device=device, dtype=dtype) if i == 0 else x for i, x in enumerate(e))
                if isinstance(e, tuple)
                else e
                for e in v
            ]
        elif torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def run_pass(model, batch, chunk: int):
    model.config.loss_chunk_size = chunk
    model.zero_grad(set_to_none=True)
    out = model(**batch)
    out.loss.backward()
    grads = {
        "lm_head": model.lm_head.weight.grad.detach().clone(),
        "layer0.q": model.model.layers[0].self_attn.q_proj.weight.grad.detach().clone(),
        "connector": next(
            p.grad.detach().clone()
            for p in model.model.connector.parameters()
            if p.grad is not None
        ),
    }
    return float(out.loss.detach()), grads


def compare(name, a, b, rtol, cos_min=None):
    """Relative-to-magnitude comparison: fp32 GEMMs of different shapes
    (one full matmul vs chunked matmuls) legitimately differ by accumulation
    order; the criterion is max|d| / max|ref|, not absolute."""
    d = (a - b).abs().max().item()
    ref = a.abs().max().item()
    rel = d / max(ref, 1e-12)
    cos = torch.nn.functional.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0
    ).item()
    ok = rel < rtol if cos_min is None else cos > cos_min
    print(
        f"  {name}: max|d|={d:.3e} max|ref|={ref:.3e} rel={rel:.3e} cos={cos:.7f}"
        f" {'OK' if ok else 'FAIL'}",
        flush=True,
    )
    assert ok, f"{name} parity failed: rel={rel}, cos={cos}"
    return d


def main():
    assert torch.cuda.is_available(), "needs a GPU node"
    register_configs()
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

    # ---- fp32 + sdpa: strict parity --------------------------------------
    model, processor = load_model(cfg.model, cfg.trainer)  # fp32 (bf16=false, fp16 default false)
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

    # TF32 would add ~1e-3 matmul noise and mask real bugs — keep the strict
    # phase in true fp32.
    torch.backends.cuda.matmul.allow_tf32 = False
    print("fp32+sdpa strict parity:", flush=True)
    loss_full, g_full = run_pass(model, batch, chunk=0)
    loss_1024, g_1024 = run_pass(model, batch, chunk=1024)
    loss_256, g_256 = run_pass(model, batch, chunk=256)
    print(
        f"  loss full={loss_full:.8f} chunk1024={loss_1024:.8f} chunk256={loss_256:.8f}", flush=True
    )
    assert abs(loss_full - loss_1024) < 1e-5, "fp32 loss parity failed (1024)"
    assert abs(loss_full - loss_256) < 1e-5, "fp32 loss parity failed (256)"
    for k in g_full:
        # Noise floor: same algorithm, different chunking = pure accumulation
        # reorder. full-vs-chunked must sit at the same scale, not above it.
        d_floor = compare(f"grad[{k}] 1024-vs-256 (noise floor)", g_1024[k], g_256[k], rtol=1e-4)
        d_full = compare(f"grad[{k}] full-vs-1024", g_full[k], g_1024[k], rtol=1e-4)
        assert d_full < max(d_floor * 20, 1e-6), (
            f"grad[{k}]: full-vs-chunked diff {d_full:.3e} far above the "
            f"chunking-noise floor {d_floor:.3e} — suspicious, not just rounding"
        )
    # tied embedding check: lm_head grad must equal embed_tokens grad object-wise
    tied = model.model.embed_tokens.weight is model.lm_head.weight
    print(f"  embeddings tied: {tied}", flush=True)
    del model
    torch.cuda.empty_cache()

    # ---- bf16 + FA2: loose parity + memory -------------------------------
    with initialize_config_dir(config_dir=str(REPO / "src/vlm/config"), version_base=None):
        cfg2 = compose(
            config_name="sft-unified", overrides=["model=qwen3-0.6b-unified", "trainer.bf16=true"]
        )
    model, processor = load_model(cfg2.model, cfg2.trainer)  # bf16 + flash_attention_2
    # load_model builds the connector AFTER from_pretrained (init_other_components)
    # so it is fp32 regardless of the requested dtype; in production deepspeed
    # homogenizes the engine to bf16 — mirror that here for a bare model.
    model = model.to(torch.bfloat16).cuda().train()
    long_batch = to_device(build_batch(processor, data_args, seq_pad=60), "cuda", torch.bfloat16)

    print("bf16+FA2 loose parity (long batch):", flush=True)
    torch.cuda.reset_peak_memory_stats()
    loss_full, g_full = run_pass(model, long_batch, chunk=0)
    mem_full = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    loss_chunk, g_chunk = run_pass(model, long_batch, chunk=1024)
    mem_chunk = torch.cuda.max_memory_allocated() / 2**30
    rel = abs(loss_full - loss_chunk) / max(abs(loss_full), 1e-8)
    print(f"  loss full={loss_full:.6f} chunked={loss_chunk:.6f} rel={rel:.2e}", flush=True)
    assert rel < 1e-2, "bf16 loss parity failed"
    for k in g_full:
        compare(f"grad[{k}]", g_full[k], g_chunk[k], rtol=float("inf"), cos_min=0.999)
    print(f"  peak mem: full={mem_full:.2f}GiB chunked={mem_chunk:.2f}GiB", flush=True)
    assert mem_chunk < mem_full, "chunked CE did not reduce peak memory"

    print("ALL CHUNKED-CE PARITY CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
