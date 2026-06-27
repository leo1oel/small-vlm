"""Isolate the FA2 generate failure: forward vs generate, sdpa vs FA2."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
print("tokenizer:", type(tok).__name__, "len:", len(tok))
enc = tok("Flash attention smoke test:", return_tensors="pt")
print("input_ids:", tuple(enc.input_ids.shape), enc.input_ids.tolist())
assert enc.input_ids.numel() > 0, "tokenizer produced an EMPTY encoding"

for impl in ["sdpa", "flash_attention_2"]:
    m = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.bfloat16, attn_implementation=impl
    ).cuda()
    ids, mask = enc.input_ids.cuda(), enc.attention_mask.cuda()
    with torch.no_grad():
        out = m(input_ids=ids, attention_mask=mask, labels=ids)
    print(f"{impl}: forward OK, loss={float(out.loss):.4f}")
    try:
        gen = m.generate(input_ids=ids, attention_mask=mask, max_new_tokens=8, do_sample=False)
        print(f"{impl}: generate OK -> {tok.decode(gen[0][ids.shape[1] :])!r}")
    except Exception as e:  # noqa: BLE001
        print(f"{impl}: generate FAILED: {type(e).__name__}: {str(e)[:160]}")
    del m
    torch.cuda.empty_cache()
