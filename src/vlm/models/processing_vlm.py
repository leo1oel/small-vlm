import json
from pathlib import Path
from typing import Any, override

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
    @override
    def from_pretrained(cls, pretrained_model_name_or_path: Any, **kwargs: Any):
        # RawImageProcessor (encoder-free path) is repo-local and not in
        # transformers' AutoImageProcessor registry, so the generic
        # ProcessorMixin.from_pretrained cannot resolve it from
        # preprocessor_config.json. Detect that case and rebuild manually;
        # everything else falls through to the stock implementation.
        preprocessor_config = Path(str(pretrained_model_name_or_path)) / "preprocessor_config.json"
        if preprocessor_config.is_file():
            config = json.loads(preprocessor_config.read_text())
            if config.get("image_processor_type") == "RawImageProcessor":
                from .image_processing_raw import RawImageProcessor

                image_processor = RawImageProcessor.from_pretrained(
                    pretrained_model_name_or_path, **kwargs
                )
                tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
                return cls(image_processor=image_processor, tokenizer=tokenizer)
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    @classmethod
    def from_names(
        cls,
        image_processor_name: str,
        tokenizer_name: str,
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
        image_processor = AutoImageProcessor.from_pretrained(
            image_processor_name, **image_processor_args
        )
        return cls(image_processor=image_processor, tokenizer=tokenizer)
