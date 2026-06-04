"""CPU smoke test: build a tiny VLM end-to-end, run forward/backward/optimizer step,
save + reload, verify weights survive round-trip (catches v5 _init_weights regressions),
then generate.

Run: uv run python devtools/smoke_test.py
Requires network (downloads Qwen2.5-0.5B-Instruct + a small SigLIP) or a warm HF cache.

Adaptations vs. the original scaffolding (each explained):
  * load_model returns (model, processor); on the non-`from_pretrained` path it builds
    the VLM from the base LLM + a fresh connector/vision tower and resizes embeddings.
    We use it as-is.
  * load_model calls OmegaConf.to_container(...) on the model sub-configs, so the configs
    must be OmegaConf objects, not bare dataclasses. We build them with
    OmegaConf.structured(...) (which validates against the dataclass schema) instead of
    instantiating the dataclasses directly.
  * The reload uses get_dynamic_vlm(<checkpoint dir>). get_dynamic_vlm_class first tries
    AutoConfig.from_pretrained(dir); for our saved model_type="vlm" (unregistered) that
    raises, and the fallback reads config.json["hf_name"] (the base LLM) to rebuild the
    dynamic classes. So passing the checkpoint dir is correct.
  * image_token_index is a negative sentinel (-200), never a real vocab id; the model's
    multimodal merge code splits on it. We keep one image token at position 0.
  * generate() takes `inputs` (raw input_ids) + `images`; it internally builds
    inputs_embeds and calls the parent generate. This exercises the v5
    prepare_inputs_for_generation / cache_position path.
"""

import tempfile

import torch
from omegaconf import OmegaConf

from vlm.config.config_schema import (
    ConnectorConfig,
    LanguageModelConfig,
    ModelConfig,
    TrainerConfig,
    VisualEncoderConfig,
)
from vlm.vlm import load_model


def main() -> None:
    model_cfg = OmegaConf.structured(
        ModelConfig(
            name="smoke",
            visual_encoder=VisualEncoderConfig(
                hf_name="google/siglip-base-patch16-224",
                output_layer=-2,
                use_all_tokens=True,
            ),
            language_model=LanguageModelConfig(
                hf_name="Qwen/Qwen2.5-0.5B-Instruct",
                max_seq_length=512,
            ),
            connector=ConnectorConfig(name="mlp_2_gelu", type="mlp"),
        )
    )
    trainer_cfg = OmegaConf.structured(
        TrainerConfig(name="smoke", bf16=False, fp16=False, attn_implementation="sdpa")
    )
    model, processor = load_model(model_cfg, trainer_cfg)
    model.train()

    image_token_index = model.config.image_token_index
    input_ids = torch.tensor([[image_token_index, 1, 2, 3, 4]])
    labels = torch.tensor([[-100, 1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    images = torch.randn(1, 3, 224, 224)

    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, images=images)
    loss = out.loss if hasattr(out, "loss") else out[0]
    assert loss is not None and loss.requires_grad, "no differentiable loss"
    loss.backward()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    opt.step()
    print(f"forward/backward/step OK, loss={loss.item():.4f}")

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        processor.save_pretrained(tmp)
        from vlm.models import get_dynamic_vlm

        VLMForCausalLM, _ = get_dynamic_vlm(tmp)
        reloaded = VLMForCausalLM.from_pretrained(tmp, dtype=torch.float32)
        for (n1, p1), (_, p2) in zip(
            model.named_parameters(), reloaded.named_parameters(), strict=True
        ):
            assert torch.allclose(p1.detach(), p2.detach(), atol=1e-6), f"weight mismatch: {n1}"
    print("save/reload round-trip OK")

    model.eval()
    with torch.no_grad():
        gen = model.generate(inputs=input_ids, images=images, max_new_tokens=4)
    print("generate OK:", tuple(gen.shape))


if __name__ == "__main__":
    main()
