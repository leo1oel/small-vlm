from typing import Any

from transformers import AutoImageProcessor, AutoTokenizer, ProcessorMixin
from transformers.image_processing_utils import ImageProcessingMixin


class OpenCLIPImageProcessorWrapper(ImageProcessingMixin):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, images, **kwargs):
        if isinstance(images, list):
            return [self.transform(img) for img in images]
        return self.transform(images)

    def preprocess(self, images, return_tensors="pt", **kwargs):
        import torch
        if isinstance(images, list):
            processed = [self.transform(img) for img in images]
            if return_tensors == "pt":
                return {"pixel_values": torch.stack(processed)}
        else:
            processed = self.transform(images)
            if return_tensors == "pt":
                return {"pixel_values": processed.unsqueeze(0)}
        return {"pixel_values": processed}


class VLMProcessor(ProcessorMixin):
    attributes: list[str] = ["image_processor", "tokenizer"]
    image_processor_class: str = "AutoImageProcessor"
    tokenizer_class: str = "AutoTokenizer"

    def __init__(
        self,
        image_processor: AutoImageProcessor = None,
        tokenizer: AutoTokenizer = None,
        open_clip_model: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(image_processor, tokenizer, **kwargs)

    @classmethod
    def from_names(
        cls,
        image_processor_name: str,
        tokenizer_name: str,
        open_clip_model: str | None = None,
        **kwargs: Any,
    ):
        image_processor_args = {
            k: v for k, v in kwargs.items() if k in ["trust_remote_code", "use_fast"]
        }
        tokenizer_args = {
            k: v
            for k, v in kwargs.items()
            if k in ["trust_remote_code", "use_fast", "model_max_length", "padding_side"]
        }

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_args)
        if open_clip_model:
            import open_clip

            _, _, transform = open_clip.create_model_and_transforms(open_clip_model)
            image_processor = OpenCLIPImageProcessorWrapper(transform)
        else:
            image_processor = AutoImageProcessor.from_pretrained(
                image_processor_name, **image_processor_args
            )
        return cls(image_processor=image_processor, tokenizer=tokenizer)
