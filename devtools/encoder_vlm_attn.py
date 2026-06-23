"""Image-attention allocation on ENCODER-BASED VLMs (LLaVA-1.5, Qwen2.5-VL),
measured with FastV's exact definition so it is directly comparable to our
encoder-free numbers (NEO / Gemma / bee-mix):

  per layer:  mean_over_heads( attn[last_query, :] )  summed over image-token
  keys  ->  fraction of the answer-position attention that lands on image
  patches. (FastV fastv_forward.py: mean over heads -> [-1] last token ->
  [SYS:SYS+IMG] image columns.)

Also records last->{non-image-prefix} as the sink share. VMCBench dev.
Runs in MAIN .venv (transformers 5.10), eager attention.

Usage: python devtools/encoder_vlm_attn.py <model_id> <kind> <out.json> [n]
  kind in {llava, qwen}
"""

import json
import sys
from pathlib import Path

import torch

DEV = "cuda"
POST = "Answer with the option's letter from the given choices directly.\n"


def load_vmcbench():
    import datasets
    return datasets.load_dataset("suyc21/VMCBench", split="dev")


def doc_to_prompt(doc):
    op = "Options:\n" + "".join(f"{k}. {doc[k]}\n" for k in "ABCD")
    return f"Question: {doc['question']}\n{op}{POST}"


@torch.no_grad()
def main():
    model_id, kind, out_path = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(model_id)
    if kind == "llava":
        from transformers import LlavaForConditionalGeneration as M
    elif kind == "qwen":
        from transformers import Qwen2_5_VLForConditionalGeneration as M
    elif kind == "gemma":
        from transformers import Gemma4UnifiedForConditionalGeneration as M
    else:
        raise ValueError(kind)
    model = M.from_pretrained(model_id, dtype=torch.bfloat16,
                             attn_implementation="eager").to(DEV).eval()
    img_id = getattr(model.config, "image_token_index", None) or \
        getattr(model.config, "image_token_id", None)
    ds = load_vmcbench()
    n = min(n, len(ds))

    # locate the LLM decoder layers count via a probe forward later
    per_layer_img, per_layer_sink, cnt = None, None, 0
    print(f"[encattn] {model_id} kind={kind} img_id={img_id} n={n}", flush=True)
    for i in range(n):
        doc = ds[i]
        try:
            img = doc["image"].convert("RGB")
            q = doc_to_prompt(doc)
            if kind == "gemma":
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": img}, {"type": "text", "text": q}]}]
                inp = proc.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt")
                inp = {k: (v.to(DEV) if isinstance(v, torch.Tensor) else v)
                       for k, v in inp.items()}
            else:
                messages = [{"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": q}]}]
                prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
                inp = proc(images=[img], text=[prompt], return_tensors="pt").to(DEV)
            out = model(**inp, output_attentions=True)
            ids = inp["input_ids"][0]
            vis = (ids == img_id)
            if not bool(vis.any()):
                continue
            attns = out.attentions  # tuple(L) of (1,H,S,S)
            L = len(attns)
            if per_layer_img is None:
                per_layer_img = [0.0] * L
                per_layer_sink = [0.0] * L
            first_vis = int(vis.nonzero()[0])
            # sink class = non-image tokens BEFORE the image (system/template)
            pre = torch.zeros_like(vis); pre[:first_vis] = True
            for li, a in enumerate(attns):
                am = a[0].float().mean(0)  # mean over heads -> (S,S)
                last = am[-1]              # last-token query
                per_layer_img[li] += last[vis].sum().item()
                per_layer_sink[li] += last[pre].sum().item()
            cnt += 1
            del out, attns
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            if i < 3:
                print(f"  skip {i}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 25 == 0:
            torch.cuda.empty_cache()
            print(f"[encattn] {i + 1}/{n} (scored {cnt})", flush=True)

    img = [x / max(cnt, 1) for x in per_layer_img]
    sink = [x / max(cnt, 1) for x in per_layer_sink]
    rec = dict(model=model_id, kind=kind, n=cnt, n_layers=len(img),
               last2img=img, last2sink=sink)
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    out[model_id] = rec
    out_path.write_text(json.dumps(out, indent=1))
    import statistics as st
    print(f"[encattn] {model_id}: last->image mean={st.mean(img):.3f} "
          f"peak={max(img):.3f}@L{img.index(max(img))} | last->sink mean={st.mean(sink):.3f}",
          flush=True)


if __name__ == "__main__":
    main()
