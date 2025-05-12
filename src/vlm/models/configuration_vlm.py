import logging
from typing import Any

from transformers import AutoConfig, PretrainedConfig

log: logging.Logger = logging.getLogger(name=__name__)


class VisionConfig(PretrainedConfig):
    model_type: str = "vision_model"

    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class ConnectorConfig(PretrainedConfig):
    model_type: str = "connector"

    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


def create_dynamic_vlm_config_class(
    base_language_model_name_or_path: str,
) -> type[PretrainedConfig]:
    BaseLMConfigClass = AutoConfig.from_pretrained(
        base_language_model_name_or_path, trust_remote_code=True
    ).__class__

    if not issubclass(BaseLMConfigClass, PretrainedConfig):
        raise TypeError(
            f"The base config class {BaseLMConfigClass.__name__} for "
            f"{base_language_model_name_or_path} does not inherit from PretrainedConfig."
        )

    class DynamicVLMConfig(BaseLMConfigClass):
        model_type: str = "vlm"

        _sub_config_classes: dict[str, type[PretrainedConfig]] = {
            "vision_config": VisionConfig,
            "connector_config": ConnectorConfig,
        }

        def __init__(
            self,
            vision_config_args: dict[str, Any] = None,
            connector_config_args: dict[str, Any] = None,
            lazy_load: bool = False,
            **kwargs: Any,
        ):
            final_vision_args = kwargs.pop("vision_config", vision_config_args)
            final_connector_args = kwargs.pop("connector_config", connector_config_args)

            self.vision_config: VisionConfig = VisionConfig(**(final_vision_args or {}))
            self.connector_config: ConnectorConfig = ConnectorConfig(**(final_connector_args or {}))
            self.lazy_load: bool = lazy_load

            # Initialize the base language model configuration part
            super().__init__(**kwargs)

    base_model_type_str = getattr(
        BaseLMConfigClass, "model_type", BaseLMConfigClass.__name__.replace("Config", "")
    )
    dynamic_class_name = f"VLMWith{base_model_type_str.capitalize()}BaseConfig"
    DynamicVLMConfig.__name__ = dynamic_class_name
    DynamicVLMConfig.__qualname__ = dynamic_class_name  # For better representation

    return DynamicVLMConfig
