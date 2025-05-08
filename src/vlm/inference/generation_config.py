from transformers import GenerationConfig


def get_generation_config(generation_args: GenerationConfig) -> GenerationConfig:
    return GenerationConfig(
        max_length=generation_args.max_length,
        max_new_tokens=generation_args.max_new_tokens,
        min_length=generation_args.min_length,
        do_sample=generation_args.do_sample,
        num_beams=generation_args.num_beams,
        temperature=generation_args.temperature,
        top_k=generation_args.top_k,
        top_p=generation_args.top_p,
    )
