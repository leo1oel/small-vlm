import json
import logging
from pathlib import Path
from typing import Any, override

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch import FloatTensor, LongTensor, Tensor
from transformers import (
    AutoConfig,
)
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_MAPPING,
)

from ..config import (
    ConnectorConfig,
)
from .configuration_vlm import create_dynamic_vlm_config_class
from .connectors import Connector, connector_map
from .visual_encoders import HFVisualEncoder

log: logging.Logger = logging.getLogger(name=__name__)


def get_dynamic_vlm_class(
    base_language_model_name_or_path: str,  # e.g., "google/gemma-3-4b-it"
):
    # Determine the base LLM's actual config class and model class
    try:
        base_language_model_actual_config = AutoConfig.from_pretrained(
            base_language_model_name_or_path, trust_remote_code=True
        )
        base_language_model_config_class = base_language_model_actual_config.__class__
    except Exception as _:
        config_file = Path(base_language_model_name_or_path) / "config.json"
        try:
            data = json.loads(config_file.read_text())
            base_language_model_name_or_path = data["hf_name"]
            base_language_model_actual_config = AutoConfig.from_pretrained(
                base_language_model_name_or_path, trust_remote_code=True
            )
            base_language_model_config_class = base_language_model_actual_config.__class__
        except Exception as e:
            raise e

    if base_language_model_config_class not in MODEL_FOR_CAUSAL_LM_MAPPING:
        raise ValueError(
            f"Config class {base_language_model_config_class.__name__} for LLM {base_language_model_name_or_path} "
            f"not in MODEL_FOR_CAUSAL_LM_MAPPING. Cannot determine parent CausalLM class."
        )
    ParentLLMClass = MODEL_MAPPING[base_language_model_config_class]
    ParentCasualLLMClass = MODEL_FOR_CAUSAL_LM_MAPPING[base_language_model_config_class]
    log.info(
        f"Determined ParentLLMClass and ParentCasualLLMClass for VLM: {ParentLLMClass.__name__} and {ParentCasualLLMClass.__name__} (from {base_language_model_name_or_path})"
    )
    return ParentLLMClass, ParentCasualLLMClass, base_language_model_name_or_path


def create_dynamic_vlm_class(
    base_language_model_name_or_path: str,  # e.g., "google/gemma-3-4b-it"
    config_class: Any,
    ParentLLMClass: Any,
):
    @override
    def __init__(self: Any, config):  # pyright: ignore
        super(self.__class__, self).__init__(config)
        if not config.lazy_load:
            self.vision_model = HFVisualEncoder(self._process_config(config.vision_config))
            self.connector = self._build_connector(config)
        log.info(f"DynamicVLM class {self.__class__.__name__} initialized.")

    def init_other_components(self: Any) -> None:
        self.vision_model = HFVisualEncoder(self._process_config(self.config.vision_config))
        self.connector = self._build_connector(self.config)

    # Placeholder for _process_config (you need to implement this)
    def _process_config(self: Any, config: Any) -> Any:
        if isinstance(config, dict):
            return OmegaConf.create(config.to_dict())
        return config

    # Placeholder for _build_connector (you need to implement this)
    def _build_connector(self: Any, config: Any) -> Connector:
        connector_config: ConnectorConfig = self._process_config(config.connector_config)
        connector_class = connector_map.get(connector_config.type)
        if not connector_class:
            raise ValueError(f"Unsupported connector type: {connector_config.type}")
        return connector_class(
            connector_config,
            self.vision_model.hidden_size,
            config.hidden_size,
        )

    DynamicVLMClass = type(
        "VLM",
        (ParentLLMClass,),  # Inherit from the specific LLM class
        {
            "config_class": config_class,
            "__init__": __init__,
            "init_other_components": init_other_components,
            "_process_config": _process_config,
            "_build_connector": _build_connector,
        },
    )
    return DynamicVLMClass


def create_dynamic_casual_vlm_class(
    base_language_model_name_or_path: str,  # e.g., "google/gemma-3-4b-it"
    pretrain_class: Any,
    config_class: Any,
    ParentCasualLLMClass: Any,
):
    @override
    def __init__(self: Any, config):  # pyright: ignore
        super(self.__class__, self).__init__(config)
        self.model = pretrain_class(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        log.info(f"DynamicCasualVLM class {self.__class__.__name__} initialized.")

    @override
    def forward(
        self: Any,
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
        return super(self.__class__, self).forward(
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

    @override
    def generate(
        self: Any,
        inputs: Tensor | None = None,
        images: FloatTensor | None = None,
        image_sizes: list[list[int]] | None = None,
        **kwargs: Any,
    ) -> Any:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (_, position_ids, attention_mask, _, inputs_embeds, _) = (
                self.prepare_inputs_labels_for_multimodal(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    images,
                )
            )
        else:
            inputs_embeds = self.get_input_embeddings()(inputs)

        return super(self.__class__, self).generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )

    @override
    def prepare_inputs_for_generation(
        self: Any,
        input_ids: Tensor,
        past_key_values: list[FloatTensor] | None = None,
        inputs_embeds: Tensor | None = None,
        **kwargs: Any,
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super(self.__class__, self).prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        inputs.pop("cache_position")
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs

    def encode_images(self: Any, images: Tensor) -> tuple[Tensor, ...]:
        image_features = self.model.vision_model(images)
        image_features = self.model.connector(image_features)

        return image_features

    def unpad_image(self: Any, tensor: Tensor, original_size: tuple[int, int]) -> Tensor:
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
        self: Any,
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
        vision_model = self.model.vision_model
        if vision_model is None or images is None or input_ids.shape[1] == 1:  # pyright: ignore
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
            labels = torch.full_like(input_ids, self.config.ignore_index)  # pyright: ignore

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
            num_images = (cur_input_ids == self.config.image_token_index).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_input_embeddings()(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])  # pyright: ignore
                cur_image_idx += 1
                continue

            image_token_indices = (
                [-1]
                + torch.where(cur_input_ids == self.config.image_token_index)[0].tolist()
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
            cur_input_embeds = self.get_input_embeddings()(torch.cat(cur_input_ids_noim))
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
                            self.config.ignore_index,
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
        tokenizer_model_max_length = self.config.max_seq_length
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            self.config.ignore_index,
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
            if self.config.padding_side == "left":
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

    DynamicCasualVLMClass = type(
        "VLMForCausalLM",
        (ParentCasualLLMClass,),  # Inherit from the specific LLM class
        {
            "config_class": config_class,
            "__init__": __init__,
            "forward": forward,
            "generate": generate,
            "prepare_inputs_for_generation": prepare_inputs_for_generation,
            "encode_images": encode_images,
            "unpad_image": unpad_image,
            "prepare_inputs_labels_for_multimodal": prepare_inputs_labels_for_multimodal,
        },
    )

    return DynamicCasualVLMClass


def get_dynamic_vlm(
    base_language_model_name_or_path: str,
):
    ParentLLMClass, ParentCasualLLMClass, base_language_model_name_or_path = get_dynamic_vlm_class(
        base_language_model_name_or_path
    )
    VLMConfig = create_dynamic_vlm_config_class(base_language_model_name_or_path)
    VLM = create_dynamic_vlm_class(base_language_model_name_or_path, VLMConfig, ParentLLMClass)
    VLMForCausalLM = create_dynamic_casual_vlm_class(
        base_language_model_name_or_path,
        VLM,
        VLMConfig,
        ParentCasualLLMClass,
    )
    return VLMForCausalLM, VLMConfig
