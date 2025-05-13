import re
from types import SimpleNamespace

import torch
from PIL import Image

from ..models import VLMProcessor, get_dynamic_vlm
from ..utils import conv_templates
from .generator import process_images, tokenizer_image_token


def eval_model(
    pretrained: str,
    query: str,
    image_path: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    num_beams: int = 1,
    max_new_tokens: int = 100,
    bf16: bool = True,
    fp16: bool = False,
    attn_implementation: str = "triton",
):
    processor = VLMProcessor.from_pretrained(
        pretrained,
    )
    VLMForCasualLM, _ = get_dynamic_vlm(pretrained)
    model: VLMForCasualLM = VLMForCasualLM.from_pretrained(
        pretrained,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32,
        attn_implementation=attn_implementation,
    )
    model.cuda()
    model.eval()
    tokenizer = processor.tokenizer
    image_processor = processor.image_processor

    image_token_index = model.config.image_token_index
    image_start_token = model.config.image_start_token
    image_end_token = model.config.image_end_token
    image_token = model.config.image_token
    image_placeholder = "<image-placeholder>"
    image_token_se = image_start_token + image_token + image_end_token
    if image_placeholder in query:
        if model.config.use_start_end_tokens:
            query = re.sub(image_placeholder, image_token_se, query)
        else:
            query = re.sub(image_placeholder, image_token, query)
    else:
        if model.config.use_start_end_tokens:
            query = image_token_se + "\n" + query
        else:
            query = image_token + "\n" + query

    conv_mode = "v1"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    images = Image.open(image_path).convert("RGB")
    image_sizes = [images.size]
    images_tensor = process_images(
        [images], image_processor, SimpleNamespace(image_aspect_ratio="pad")
    ).to(model.device, dtype=model.config.torch_dtype)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, image_token_index, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)
