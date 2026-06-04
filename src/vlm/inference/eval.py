import re
import warnings
from types import SimpleNamespace

import torch
from PIL import Image

from ..models import VLMProcessor, get_dynamic_vlm
from ..utils import conv_templates
from .generator import process_images, tokenizer_image_token


def _auto_detect_conv_mode(model_path: str):
    """Auto-detect conversation mode based on model path"""
    model_path_lower = model_path.lower()

    # Check for specific model types in order of specificity
    if "llama-3" in model_path_lower or "llama3" in model_path_lower:
        return "llava_llama_3"
    elif "llama-2" in model_path_lower or "llama2" in model_path_lower:
        return "llava_llama_2"
    elif "llama" in model_path_lower:
        return "llava_llama_2"  # Default for llama models
    elif "qwen2.5" in model_path_lower or "qwen-2.5" in model_path_lower:
        return "qwen_2_5"
    elif "qwen2" in model_path_lower or "qwen-2" in model_path_lower:
        return "qwen_2"
    elif "qwen1.5" in model_path_lower or "qwen-1.5" in model_path_lower:
        return "qwen_1_5"
    elif "qwen" in model_path_lower:
        return "qwen_1_5"  # Default for qwen models
    elif "mistral" in model_path_lower and "instruct" in model_path_lower:
        return "mistral_instruct"
    elif "mistral" in model_path_lower and "orca" in model_path_lower:
        return "mistral_orca"
    elif "mistral" in model_path_lower and "zephyr" in model_path_lower:
        return "mistral_zephyr"
    elif "mistral" in model_path_lower:
        return "mistral_instruct"  # Default for mistral models
    elif "gemma" in model_path_lower and "instruct" in model_path_lower:
        return "gemma_instruct"
    elif "vicuna" in model_path_lower:
        return "vicuna_v1"
    elif "mpt" in model_path_lower:
        return "mpt"
    elif "llava" in model_path_lower and "v0" in model_path_lower:
        return "llava_v0"
    elif "llava" in model_path_lower:
        return "llava_v1"  # Default for llava models
    else:
        # Default fallback
        warnings.warn(
            f"Could not auto-detect conv_mode for {model_path}, using 'v1' as default",
            stacklevel=2,
        )
        return "v1"


def load_model(
    pretrained: str,
    bf16: bool = True,
    fp16: bool = False,
    attn_implementation: str = "eager",
):
    """
    Load VLM model and processor for inference.

    Args:
        pretrained: Path to pretrained model
        bf16: Use bfloat16 precision
        fp16: Use float16 precision
        attn_implementation: Attention implementation type

    Returns:
        Tuple of (model, processor, config_dict)
    """
    processor = VLMProcessor.from_pretrained(pretrained)
    VLMForCausalLM, _ = get_dynamic_vlm(pretrained)
    model: VLMForCausalLM = VLMForCausalLM.from_pretrained(
        pretrained,
        dtype=torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32,
        attn_implementation=attn_implementation,
    )
    model.cuda()
    model.eval()

    config_dict = {
        "image_token_index": model.config.image_token_index,
        "image_start_token": model.config.image_start_token,
        "image_end_token": model.config.image_end_token,
        "image_token": getattr(model.config, "image_token", "<image>"),
        "use_start_end_tokens": model.config.use_start_end_tokens,
    }

    return model, processor, config_dict


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
    attn_implementation: str = "eager",
):
    model, processor, config = load_model(
        pretrained, bf16=bf16, fp16=fp16, attn_implementation=attn_implementation
    )

    tokenizer = processor.tokenizer
    image_processor = processor.image_processor

    image_token_index = config["image_token_index"]
    image_start_token = config["image_start_token"]
    image_end_token = config["image_end_token"]
    image_token = config["image_token"]
    image_placeholder = "<image-placeholder>"
    image_token_se = image_start_token + image_token + image_end_token
    if image_placeholder in query:
        if config["use_start_end_tokens"]:
            query = re.sub(image_placeholder, image_token_se, query)
        else:
            query = re.sub(image_placeholder, image_token, query)
    else:
        if config["use_start_end_tokens"]:
            query = image_token_se + "\n" + query
        else:
            query = image_token + "\n" + query

    conv_mode = _auto_detect_conv_mode(pretrained)

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    images = Image.open(image_path).convert("RGB")
    image_sizes = [images.size]
    images_tensor = process_images(
        [images], image_processor, SimpleNamespace(image_aspect_ratio="pad")
    )
    # Ensure it's a tensor
    if isinstance(images_tensor, list):
        images_tensor = torch.stack(images_tensor, dim=0)
    images_tensor = images_tensor.to(
        model.device, dtype=getattr(model.config, "dtype", None) or next(model.parameters()).dtype
    )

    input_ids = tokenizer_image_token(prompt, tokenizer, image_token_index, return_tensors="pt")
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_ids = input_ids.unsqueeze(0).cuda()

    # Create attention mask and set pad_token_id
    attention_mask = torch.ones_like(input_ids)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id,
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
