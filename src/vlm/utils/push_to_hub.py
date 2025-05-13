import torch

from ..models import VLMProcessor, get_dynamic_vlm


def push_to_hub(pretrained: str, repo_name: str):
    processor = VLMProcessor.from_pretrained(
        pretrained,
    )
    VLMForCasualLM, _ = get_dynamic_vlm(pretrained)
    model: VLMForCasualLM = VLMForCasualLM.from_pretrained(
        pretrained, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
    )
    # AutoConfig.register("vlm", VLMConfig)
    # AutoProcessor.register(VLMConfig, VLMProcessor)
    # AutoModel.register(VLMConfig, VLMForCasualLM)

    processor.push_to_hub(repo_name)
    model.push_to_hub(repo_name)
