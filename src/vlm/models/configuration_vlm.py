import logging
from typing import Any, ClassVar

from transformers import AutoConfig, PretrainedConfig

log: logging.Logger = logging.getLogger(name=__name__)


class VisionConfig(PretrainedConfig):
    # transformers v5 wraps every PretrainedConfig subclass in
    # @dataclass(kw_only=True); a plainly-annotated class attribute would become a
    # dataclass field. The base declares `model_type: ClassVar[str]`
    # (configuration_utils.py:227), and ClassVar is excluded from dataclass fields,
    # so this override is safe.
    model_type: ClassVar[str] = "vision_model"

    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class ConnectorConfig(PretrainedConfig):
    model_type: ClassVar[str] = "connector"

    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class AudioConfig(PretrainedConfig):
    """Audio-pathway config (encoder-free, gemma4_unified-style).

    Carries the yaml dials for the audio connector: enabled, name, type
    (connector_map key, e.g. "raw_waveform"), samples_per_token (frame size,
    640 = 40ms @ 16kHz), plus anything future connectors need. Kwargs
    passthrough like VisionConfig/ConnectorConfig.
    """

    model_type: ClassVar[str] = "audio_connector"

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
        # ClassVar: override the base value without creating a dataclass field
        # (transformers v5 wraps configs in @dataclass(kw_only=True)).
        model_type: ClassVar[str] = "vlm"

        def __init__(
            self,
            vision_config_args: dict[str, Any] = None,
            connector_config_args: dict[str, Any] = None,
            audio_config_args: dict[str, Any] = None,
            lazy_load: bool = False,
            **kwargs: Any,
        ):
            final_vision_args = kwargs.pop("vision_config", vision_config_args)
            final_connector_args = kwargs.pop("connector_config", connector_config_args)
            final_audio_args = kwargs.pop("audio_config", audio_config_args)

            self.vision_config: VisionConfig = VisionConfig(**(final_vision_args or {}))
            self.connector_config: ConnectorConfig = ConnectorConfig(**(final_connector_args or {}))
            # None (not an empty AudioConfig) when absent: old checkpoints and
            # vision-only configs simply have no audio pathway.
            self.audio_config: AudioConfig | None = (
                AudioConfig(**final_audio_args) if final_audio_args else None
            )
            self.lazy_load: bool = lazy_load
            # Initialize the base language model configuration part
            super().__init__(**kwargs)

    return DynamicVLMConfig
