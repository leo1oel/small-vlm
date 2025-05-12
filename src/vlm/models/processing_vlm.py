from typing import Any

from transformers import AutoImageProcessor, AutoTokenizer, ProcessorMixin


class VLMProcessor(ProcessorMixin):
    attributes: list[str] = ["image_processor", "tokenizer"]
    image_processor_class: str = "AutoImageProcessor"
    tokenizer_class: str = "AutoTokenizer"

    def __init__(
        self,
        image_processor: AutoImageProcessor = None,
        tokenizer: AutoTokenizer = None,
        **kwargs: Any,
    ):
        super().__init__(image_processor, tokenizer, **kwargs)

    @classmethod
    def from_names(cls, image_processor_name: str, tokenizer_name: str, **kwargs: Any):
        image_processor_args = {
            k: v for k, v in kwargs.items() if k in ["trust_remote_code", "use_fast"]
        }
        tokenizer_args = {
            k: v
            for k, v in kwargs.items()
            if k in ["trust_remote_code", "use_fast", "model_max_length", "padding_side"]
        }

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_args)
        image_processor = AutoImageProcessor.from_pretrained(
            image_processor_name, **image_processor_args
        )
        return cls(image_processor=image_processor, tokenizer=tokenizer)
