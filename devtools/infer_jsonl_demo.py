"""Train-vs-inference demo on REAL samples from the training jsonl.

For each sample in outputs/infer_demo/samples.json (picked from the energon
training mix, images pre-downloaded from Azure):

  A. TRAINING VIEW — runs the sample through the exact training pipeline
     (messages_to_conversations -> inject_missing_media_tokens ->
     preprocess_qwen, the same functions the energon TaskEncoder calls) and
     prints a token-level view: which spans are loss-masked vs supervised,
     where the <image> sentinel (-200) sits, and how many patch tokens the
     real image expands to at the splice.

  B. INFERENCE — loads the trained checkpoint and answers the sample's first
     user turn with the actual image, next to the dataset's ground-truth
     answer.

Usage: python devtools/infer_jsonl_demo.py <checkpoint_dir>
"""

import json
import sys

import torch
from PIL import Image

from vlm.data.data_arguments import DataArguments
from vlm.data.dataset import preprocess_qwen
from vlm.data.energon_dataset import inject_missing_media_tokens, messages_to_conversations
from vlm.inference.eval import generate_response, load_model

CKPT = sys.argv[1]
SAMPLES = json.load(open("outputs/infer_demo/samples.json"))
data_args = DataArguments()  # vision-only run: defaults match training


def render_token_view(tokenizer, input_ids: list[int], labels: list[int]) -> str:
    """Compact view: contiguous runs of (masked|supervised) tokens, with the
    media sentinel called out explicitly."""
    out, run_ids, run_state = [], [], None

    def flush():
        if not run_ids:
            return
        text = tokenizer.decode(run_ids)
        tag = "TRAIN " if run_state else "masked"
        out.append(f"  [{tag}] {text!r}")

    for tid, lab in zip(input_ids, labels, strict=True):
        if tid == data_args.image_token_index:
            flush()
            run_ids.clear()
            out.append("  [masked] <IMAGE SENTINEL -200 -> spliced to N patch embeddings>")
            run_state = None
            continue
        state = lab != data_args.ignore_index
        if state != run_state and run_ids:
            flush()
            run_ids.clear()
        run_state = state
        run_ids.append(tid)
    flush()
    return "\n".join(out)


print(f"loading {CKPT} ...", flush=True)
model, processor, info = load_model(CKPT, bf16=True, attn_implementation="sdpa")
tokenizer = processor.tokenizer
print(f"loaded; conversation_version={info['conversation_version']}\n", flush=True)

for i, s in enumerate(SAMPLES):
    rec, image_path = s["record"], s["image_local"]
    print("#" * 100, flush=True)
    print(f"SAMPLE {i}  id={rec['id']}  source={rec.get('source')}  image={image_path}", flush=True)

    # ---------- A. TRAINING VIEW (the exact functions the energon path runs)
    conversations = messages_to_conversations(rec["messages"], data_args)
    inject_missing_media_tokens(conversations, n_images=1, n_audios=0, data_args=data_args)
    print("\n--- conversations (post messages_to_conversations) ---", flush=True)
    for turn in conversations:
        v = turn["value"] if len(turn["value"]) < 300 else turn["value"][:300] + " ...[truncated]"
        print(f"  {turn['from']}: {v!r}", flush=True)

    out = preprocess_qwen([conversations], tokenizer, data_args, has_image=True)
    ids = out["input_ids"][0].tolist()
    labs = out["labels"][0].tolist()
    n_sup = sum(1 for l in labs if l != data_args.ignore_index)
    with Image.open(image_path) as img:
        w, h = img.size
    n_patches = processor.image_processor.get_num_patches(h, w)
    print(
        f"\n--- training tokenization: {len(ids)} text tokens "
        f"({n_sup} supervised, {len(ids) - n_sup} masked); image {w}x{h} -> "
        f"{n_patches} patch tokens at the splice (budget {processor.image_processor.max_soft_tokens}) ---",
        flush=True,
    )
    view = render_token_view(tokenizer, ids, labs)
    if len(view) > 4000:
        view = view[:4000] + "\n  ...[truncated]"
    print(view, flush=True)

    # ---------- B. INFERENCE on the first user turn
    first_user = next(t for t in conversations if t["from"] == "human")
    first_gt = next(t for t in conversations if t["from"] == "gpt")
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    answer = generate_response(
        model, processor, query=first_user["value"], images=image_path, max_new_tokens=256
    )
    t1.record()
    torch.cuda.synchronize()
    print(f"\n--- inference (first user turn, {t0.elapsed_time(t1) / 1000:.1f}s) ---", flush=True)
    print(f"  [question ] {first_user['value']!r}", flush=True)
    print(f"  [model    ] {answer!r}", flush=True)
    print(f"  [groundtru] {first_gt['value']!r}", flush=True)
    print(flush=True)

print("DEMO COMPLETED", flush=True)
