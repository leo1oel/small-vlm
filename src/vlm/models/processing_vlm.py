import json
from pathlib import Path
from typing import Any, override

from transformers import AutoImageProcessor, AutoTokenizer, ProcessorMixin
from transformers.utils import cached_file


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

    @override
    def save_pretrained(self, save_directory: Any, push_to_hub: bool = False, **kwargs: Any):
        # transformers v5 no longer saves non-tokenizer attributes to their own
        # files: the image processor is serialized into processor_config.json
        # and its CLASS is resolved through the AutoImageProcessor registry on
        # reload — which fails for the repo-local RawImageProcessor. Write the
        # classic standalone preprocessor_config.json (with
        # image_processor_type) alongside, which our from_pretrained override
        # detects and rebuilds manually. push_to_hub also relies on this file
        # to inject the remote-code auto_map for exported checkpoints.
        outputs = super().save_pretrained(save_directory, push_to_hub=push_to_hub, **kwargs)
        from .image_processing_raw import RawImageProcessor

        if isinstance(self.image_processor, RawImageProcessor):
            self.image_processor.save_pretrained(save_directory)
        return outputs

    @classmethod
    @override
    def from_pretrained(cls, pretrained_model_name_or_path: Any, **kwargs: Any):
        # RawImageProcessor (encoder-free path) is repo-local and not in
        # transformers' AutoImageProcessor registry, so the generic
        # ProcessorMixin.from_pretrained cannot resolve it from
        # preprocessor_config.json. Detect that case and rebuild manually;
        # everything else falls through to the stock implementation.
        # cached_file resolves both local checkpoint dirs and Hub repo ids
        # (returns None when the file is absent; raises on e.g. network
        # failure for a Hub id — then fall through and let the stock path
        # produce its usual error).
        try:
            resolved = cached_file(
                str(pretrained_model_name_or_path),
                "preprocessor_config.json",
                _raise_exceptions_for_missing_entries=False,
            )
        except Exception:
            resolved = None
        if resolved is not None:
            config = json.loads(Path(resolved).read_text())
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
