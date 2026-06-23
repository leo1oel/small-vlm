"""GPU smoke for the flash-attn install: raw kernel, transformers FA2
generation on the Qwen3 backbone, and the repo's real load_model call site
(sft-unified composition). Run on a GPU node:

    srun -p ckpt-all -A cse-ckpt --gpus=l40:1 --mem=32G --time=0:15:00 \
        .venv/bin/python devtools/fa2_gpu_check.py
"""

import torch

print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)

# 1) raw flash-attn kernel
from flash_attn import flash_attn_func  # noqa: E402

q = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
out = flash_attn_func(q, torch.randn_like(q), torch.randn_like(q), causal=True)
assert out.isfinite().all()
print("1/3 flash_attn_func OK", tuple(out.shape))

# 2) transformers + FA2 end-to-end generation on the Qwen3 architecture
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
lm = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", dtype=torch.bfloat16, attn_implementation="flash_attention_2"
).cuda()
# Pass the attention_mask explicitly: Qwen3's pad==eos, and the FA2 path
# cannot infer a mask, which mangles the prefill into a 0-length sequence.
enc = tok("Flash attention smoke test:", return_tensors="pt").to("cuda")
gen = lm.generate(**enc, max_new_tokens=8, do_sample=False)
print("2/3 transformers FA2 generate OK:", tok.decode(gen[0][enc.input_ids.shape[1] :]))
del lm
torch.cuda.empty_cache()

# 3) the repo's actual load_model call site, composed from sft-unified
from hydra import compose, initialize_config_dir  # noqa: E402

from vlm.config import register_configs  # noqa: E402
from vlm.vlm import CONFIG_PATH, load_model  # noqa: E402

register_configs()
with initialize_config_dir(config_dir=str(CONFIG_PATH), version_base=None):
    cfg = compose(
        config_name="sft-unified",
        overrides=["model=qwen3-0.6b-unified", "trainer.bf16=true"],
    )
model, processor = load_model(cfg.model, cfg.trainer)
model = model.cuda()
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"3/3 load_model(sft-unified, FA2) OK — {n_train / 1e6:.1f}M params, audio off")
print("ALL CHECKS PASSED")
