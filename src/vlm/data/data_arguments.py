from dataclasses import dataclass, field

from transformers import AutoConfig

from ..config import DatasetConfig, ModelConfig


@dataclass
class DataArguments:
    data_path: str | None = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = True
    is_multimodal: bool = True
    early_mix_text: bool = False
    image_folder: str | None = field(default=None)
    use_start_end_tokens: bool = False
    use_image_patch_token: bool = False
    image_token: str = "<image>"
    image_start_token: str = "<im_start>"
    image_end_token: str = "<im_end>"
    image_patch_token: str = "<im_patch>"
    ignore_index: int = -100
    image_token_index: int = -200
    image_aspect_ratio: str = "square"
    # Soft tokens each image splices into (encoder path only: a fixed
    # (image_size/patch_size)^2 per tower; None on the encoder-free path,
    # where the per-image patch count is variable and read off the entry).
    # Used by length bucketing's effective_sample_length.
    image_soft_tokens: int | None = None
    # --- audio (encoder-free raw-waveform path; mirrors the image fields) ---
    audio_token: str = "<audio>"
    audio_token_index: int = -201
    audio_folder: str | None = None
    audio_enabled: bool = False
    audio_sampling_rate: int = 16000
    audio_samples_per_token: int = 640
    max_audio_tokens: int | None = 750


def _image_soft_tokens(model_config: ModelConfig) -> int | None:
    """Fixed per-image splice width for encoder towers (CLIP/SigLIP/DINO).
    Reads the tower's HF config (already in the local cache by the time data
    args are built — load_model runs first). None for the encoder-free path."""
    hf_name = model_config.visual_encoder.hf_name
    if hf_name is None:
        return None
    config = AutoConfig.from_pretrained(hf_name)
    config = getattr(config, "vision_config", None) or config
    tokens = (config.image_size // config.patch_size) ** 2
    if model_config.visual_encoder.use_cls_token:
        tokens += 1
    return tokens


def get_data_args(data_config: DatasetConfig, trainer_config: ModelConfig) -> DataArguments:
    return DataArguments(
        image_soft_tokens=_image_soft_tokens(trainer_config),
        data_path=data_config.path,
        lazy_preprocess=data_config.lazy_preprocess,
        is_multimodal=data_config.is_multimodal,
        image_folder=data_config.image_folder,
        image_token=trainer_config.language_model.image_token,
        early_mix_text=data_config.early_mix_text,
        use_start_end_tokens=trainer_config.language_model.use_start_end_tokens,
        use_image_patch_token=trainer_config.language_model.use_image_patch_token,
        image_start_token=trainer_config.language_model.image_start_token,
        image_end_token=trainer_config.language_model.image_end_token,
        image_patch_token=trainer_config.language_model.image_patch_token,
        ignore_index=trainer_config.language_model.ignore_index,
        image_token_index=trainer_config.language_model.image_token_index,
        image_aspect_ratio=data_config.image_aspect_ratio,
        audio_token=trainer_config.language_model.audio_token,
        audio_token_index=trainer_config.language_model.audio_token_index,
        audio_folder=data_config.audio_folder,
        audio_enabled=trainer_config.audio.enabled,
        audio_sampling_rate=trainer_config.audio.sampling_rate,
        audio_samples_per_token=trainer_config.audio.samples_per_token,
        max_audio_tokens=trainer_config.audio.max_audio_tokens,
    )
