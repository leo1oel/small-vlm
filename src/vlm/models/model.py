import ast
import logging
from typing import Any, override

import torch
from omegaconf import OmegaConf
from torch import FloatTensor, LongTensor, Tensor, device, dtype
from transformers import (
    AutoConfig,
    GenerationMixin,
    PreTrainedModel,
)

from ..config import (
    ConnectorConfig,
    LLMConfig,
    ModelConfig,
    TrainerConfig,
    VisualEncoderConfig,
)
from .connectors import Connector, connector_map
from .language_models import LanguageModel
from .visual_encoders import VisualEncoder

log: logging.Logger = logging.getLogger(name=__name__)

PRECISION_MAP: dict[str, dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "default": torch.float32,
}


def select_best_resolution(
    original_size: tuple[int, int], possible_resolutions: list[tuple[int, int]]
) -> tuple[int, int]:
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = (
            int(original_width * scale),
            int(original_height * scale),
        )
        effective_resolution = min(
            downscaled_width * downscaled_height, original_width * original_height
        )
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution
            and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit  # pyright: ignore


def get_anyres_image_grid_shape(
    image_size: tuple[int, int], grid_pinpoints: list[tuple[int, int]], patch_size: int
) -> tuple[int, int]:
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


class VLM(PreTrainedModel, GenerationMixin):
    def __init__(
        self,
        model_config: ModelConfig,
        trainer_config: TrainerConfig,
        torch_device: device,
    ) -> None:
        super().__init__(config=AutoConfig.from_pretrained(model_config.llm.hf_name))
        # process config
        self.model_config: ModelConfig = self._process_config(model_config)
        self.trainer_config: TrainerConfig = self._process_config(trainer_config)

        if trainer_config.fp16:
            self.torch_dtype: dtype = PRECISION_MAP["fp16"]
        elif trainer_config.bf16:
            self.torch_dtype = PRECISION_MAP["bf16"]
        else:
            self.torch_dtype = PRECISION_MAP["default"]

        self.torch_device: torch.device = torch_device

        # initialize components
        self.initialize_components()
        self.set_trainable_params(self.trainer_config.unfreeze)

    def _process_config(self, config: Any) -> Any:
        if isinstance(config, dict):
            return OmegaConf.create(config)  # pyright: ignore
        return config

    # initialize all components
    def initialize_components(self) -> None:
        self._visual_encoder: VisualEncoder = self._build_visual_encoder()
        self._language_model: LanguageModel = self._build_language_model()
        self._connector: Connector = self._build_connector()

    @property
    def visual_encoder(self) -> VisualEncoder:
        return self._visual_encoder

    @property
    def language_model(self) -> LanguageModel:
        return self._language_model

    @property
    def connector(self) -> Connector:
        return self._connector

    @property
    @override
    def supports_gradient_checkpointing(self):
        return self.language_model.language_model.supports_gradient_checkpointing

    def _build_visual_encoder(self) -> VisualEncoder:
        encoder_config: VisualEncoderConfig = self.model_config.visual_encoder
        if encoder_config.type == "hf_visual_encoder":
            from .visual_encoders import HFVisualEncoder

            return HFVisualEncoder(encoder_config, self.torch_dtype, self.torch_device)
        else:
            error_msg = f"Unknown visual encoder type: {encoder_config.type}"
            log.error(error_msg)
            raise ValueError(error_msg)

    def _build_language_model(self) -> LanguageModel:
        llm_config: LLMConfig = self.model_config.llm
        if llm_config.type == "hf_llm":
            from .language_models import HFLLMLanguageModel

            return HFLLMLanguageModel(llm_config, self.torch_dtype, self.torch_device)
        else:
            error_msg = f"Unknown language model type: {llm_config.type}"
            log.error(error_msg)
            raise ValueError(error_msg)

    def _build_connector(self) -> Connector:
        connector_config: ConnectorConfig = self.model_config.connector
        connector_class = connector_map.get(connector_config.type)
        if not connector_class:
            raise ValueError(f"Unsupported connector type: {connector_config.type}")
        return connector_class(
            connector_config,
            self.visual_encoder.hidden_size,
            self.language_model.hidden_size,
            self.torch_dtype,
            self.torch_device,
        )

    @override
    def forward(
        self,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: LongTensor | None = None,
        past_key_values: list[FloatTensor] | None = None,
        labels: LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        images: FloatTensor | None = None,
        image_sizes: list[list[int]] | None = None,
        return_dict: bool | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = (
                self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                )
            )
        return self.language_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    # @override
    # def generate(
    #     self,
    #     inputs: Tensor | None = None,
    #     generation_config: GenerationConfig | None = None,
    #     logits_processor: LogitsProcessorList | None = None,
    #     stopping_criteria: StoppingCriteriaList | None = None,
    #     prefix_allowed_tokens_fn: Callable | None = None,
    #     synced_gpus: bool | None = None,
    #     assistant_model: PreTrainedModel | None = None,
    #     streamer: Streamer | None = None,
    #     negative_prompt_ids: Tensor | None = None,
    #     negative_prompt_attention_mask: Tensor | None = None,
    #     use_model_defaults: bool | None = None,
    #     images: FloatTensor | None = None,
    #     image_sizes: list[list[int]] | None = None,
    #     **kwargs,
    # ) -> Any:
    #     position_ids = kwargs.pop("position_ids", None)
    #     attention_mask = kwargs.pop("attention_mask", None)
    #     if "inputs_embeds" in kwargs:
    #         raise NotImplementedError("`inputs_embeds` is not supported")

    #     if images is not None:
    #         (inputs, position_ids, attention_mask, _, inputs_embeds, _) = (
    #             self.prepare_inputs_labels_for_multimodal(
    #                 inputs,
    #                 position_ids,
    #                 attention_mask,
    #                 None,
    #                 None,
    #                 images,
    #                 image_sizes=image_sizes,
    #             )
    #         )
    #     else:
    #         inputs_embeds = self.language_model.embeddings(inputs)

    #     return self.language_model.generate(
    #         position_ids=position_ids,
    #         attention_mask=attention_mask,
    #         inputs_embeds=inputs_embeds,
    #         **kwargs,
    #     )

    def freeze_visual_encoder(self, freeze: bool = True) -> None:
        for param in self.visual_encoder.visual_encoder.parameters():
            param.requires_grad = not freeze

    def freeze_language_model(self, freeze: bool = True, except_layer_norm: bool = False) -> None:
        for name, param in self.language_model.language_model.named_parameters():
            if except_layer_norm and (
                "layernorm" in name.lower() or "layer_norm" in name.lower() or "ln_" in name.lower()
            ):
                param.requires_grad = True
            else:
                param.requires_grad = not freeze
        for param in self.language_model.embeddings.parameters():
            param.requires_grad = False
        if self.model_config.llm.use_start_end_tokens:
            for param in self.language_model.embeddings.parameters():
                param.requires_grad = True

    def freeze_connector(self, freeze: bool = True) -> None:
        for param in self.connector.parameters():
            param.requires_grad = not freeze

    def set_trainable_params(self, config: dict[str, bool]) -> None:
        if "train_visual_encoder" in config:
            self.freeze_visual_encoder(not config["train_visual_encoder"])

        if "train_language_model" in config:
            self.freeze_language_model(not config["train_language_model"])

        if "train_connector" in config:
            self.freeze_connector(not config["train_connector"])

        self._log_trainable_params()

    def _log_trainable_params(self) -> None:
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        log.info(
            f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%} of total)"
        )
        for module_name, module in [
            ("visual_encoder", self.visual_encoder),
            ("language_model", self.language_model),
            ("connector", self.connector),
        ]:
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total = sum(p.numel() for p in module.parameters())
            if total > 0:
                log.info(f"  - {module_name}: {trainable:,} ({trainable / total:.2%} of {total:,})")

    def encode_images(self, images: Tensor) -> tuple[Tensor, ...]:
        image_features = self.visual_encoder(images)
        image_features = self.connector(image_features)
        return image_features

    def unpad_image(self, tensor: Tensor, original_size: tuple[int, int]) -> Tensor:
        """
        Unpads a PyTorch tensor of a padded and resized image.

        Args:
        tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
        original_size (tuple): The original size of PIL image (width, height).

        Returns:
        torch.Tensor: The unpadded image tensor.
        """
        original_width, original_height = original_size
        current_height, current_width = tensor.shape[1:]

        original_aspect_ratio = original_width / original_height
        current_aspect_ratio = current_width / current_height

        if original_aspect_ratio > current_aspect_ratio:
            scale_factor = current_width / original_width
            new_height = int(original_height * scale_factor)
            padding = (current_height - new_height) // 2
            unpadded_tensor = tensor[:, padding : current_height - padding, :]
        else:
            scale_factor = current_height / original_height
            new_width = int(original_width * scale_factor)
            padding = (current_width - new_width) // 2
            unpadded_tensor = tensor[:, :, padding : current_width - padding]

        return unpadded_tensor

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: Tensor | None = None,
        position_ids: LongTensor | None = None,
        attention_mask: Tensor | None = None,
        past_key_values: list[FloatTensor] | None = None,
        labels: LongTensor | None = None,
        images: FloatTensor | None = None,
    ) -> tuple[
        Tensor | None,
        LongTensor | None,
        Tensor | None,
        list[FloatTensor] | None,
        Tensor | None,
        LongTensor | None,
    ]:
        visual_encoder = self.visual_encoder
        if visual_encoder is None or images is None or input_ids.shape[1] == 1:  # pyright: ignore
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if isinstance(images, list) or images.ndim == 5:
            if isinstance(images, list):
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]  # pyright: ignore
            concat_images = torch.cat([image for image in images], dim=0)  # pyright: ignore
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]  # pyright: ignore
            image_features: tuple[Tensor, ...] = torch.split(image_features, split_sizes, dim=0)  # pyright: ignore
            image_features = [x.flatten(0, 1) for x in image_features]  # pyright: ignore
        else:
            image_features = self.encode_images(images)

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0,
                input_ids.shape[1],  # pyright: ignore
                dtype=torch.long,
                device=input_ids.device,  # pyright: ignore
            )
        if labels is None:
            labels = torch.full_like(input_ids, self.model_config.llm.ignore_index)  # pyright: ignore

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask, strict=False)
        ]  # pyright: ignore
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask, strict=False)
        ]  # pyright: ignore

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == self.model_config.llm.image_token_index).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.language_model.embeddings(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])  # pyright: ignore
                cur_image_idx += 1
                continue

            image_token_indices = (
                [-1]
                + torch.where(cur_input_ids == self.model_config.llm.image_token_index)[0].tolist()
                + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]  # pyright: ignore
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(
                    cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]]
                )
                cur_labels_noim.append(
                    cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]]
                )
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.language_model.embeddings(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            self.model_config.llm.ignore_index,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = self.language_model.tokenizer.model_max_length
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            self.model_config.llm.ignore_index,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device
        )
        position_ids = torch.zeros(
            (batch_size, max_len),
            dtype=position_ids.dtype,  # pyright: ignore
            device=position_ids.device,  # pyright: ignore
        )  # pyright: ignore

        for i, (cur_new_embed, cur_new_labels) in enumerate(
            zip(new_input_embeds, new_labels, strict=False)
        ):
            cur_len = cur_new_embed.shape[0]
            if self.language_model.tokenizer.padding_side == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(  # pyright: ignore
                        0,
                        cur_len,
                        dtype=position_ids.dtype,  # pyright: ignore
                        device=position_ids.device,  # pyright: ignore
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(  # pyright: ignore
                        0,
                        cur_len,
                        dtype=position_ids.dtype,  # pyright: ignore
                        device=position_ids.device,  # pyright: ignore
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels  # pyright: ignore
