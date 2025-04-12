import logging
from abc import ABC, abstractmethod
from typing import NamedTuple, cast, override

import torch
import torch.nn as nn

from ...config.config_schema import ConnectorConfig

log: logging.Logger = logging.getLogger(name=__name__)


class ProcessedVisualFeatures(NamedTuple):
    features: list[torch.Tensor]


class ForwardOutput(NamedTuple):
    embeddings: torch.Tensor
    attention_mask: torch.Tensor


class Connector(nn.Module, ABC):
    def __init__(
        self, config: ConnectorConfig, image_hidden_size: int, text_hidden_size: int
    ) -> None:
        super().__init__()
        self.config: ConnectorConfig = config
        self.name: str = self.config.name

        self.image_hidden_size: int = image_hidden_size
        self.text_hidden_size: int = text_hidden_size
        self.projection_layer: nn.Module = self.build_projection_layer()
        self.initialize_layers()

    @abstractmethod
    def _build_projection_layer(self) -> nn.Module:
        pass

    def build_projection_layer(self) -> nn.Module:
        return self._build_projection_layer()

    @abstractmethod
    def _initialize_layers(self) -> None:
        pass

    def initialize_layers(self) -> None:
        self._initialize_layers()

    @abstractmethod
    def projection(self, visual_features: torch.Tensor) -> torch.Tensor:
        pass

    def _process_visual_features(
        self, visual_features: tuple[torch.Tensor, ...], batch_size: int
    ) -> ProcessedVisualFeatures:
        # Handle batch case
        if (
            len(visual_features) == 1 and visual_features[0].size(0) == batch_size
        ):  # visual features: ([batch_size, image_tokens_num, image_feature_dim])
            batch_visual = visual_features[0]
            batch_flat = batch_visual.view(
                -1, batch_visual.size(-1)
            )  # [batch_size * image_tokens_num, image_feature_dim]
            batch_projected = self.projection(
                batch_flat
            )  # [batch_size * image_tokens_num, text_feature_dim]
            batch_projected = batch_projected.view(
                batch_size, -1, batch_projected.size(-1)
            )  # [batch_size, image_tokens_num, text_feature_dim]
            return ProcessedVisualFeatures([batch_projected[i] for i in range(batch_size)])

        # ([image_num1, image_tokens_num, image_feature_dim], [image_num2, image_tokens_num, image_feature_dim], ...)
        # Handle individual case
        return ProcessedVisualFeatures(
            [self.projection(vf.view(-1, vf.size(-1))) for vf in visual_features]
        )  # [[image_num1 * image_tokens_num, text_feature_dim], [image_num2 * image_tokens_num, text_feature_dim], ...]

    def _create_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Create a causal (lower triangular) attention mask."""
        return torch.tril(torch.ones(size, size, device=device))

    def _process_no_image_case(
        self, valid_embeddings: torch.Tensor, need_complex_mask: bool, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        """Process a sequence with no image tokens."""
        valid_length = valid_embeddings.size(0)
        mask = self._create_causal_mask(valid_length, device) if need_complex_mask else None
        return valid_embeddings, mask, valid_length

    def _create_fusion_mask(
        self, chunks: list[torch.Tensor], is_visual: list[bool], device: torch.device
    ) -> torch.Tensor:
        """
        Create attention mask for fused chunks.

        Creates a mask where:
        - Text chunks have causal attention
        - Visual chunks have full self-attention
        - Later chunks can attend to earlier chunks
        """
        chunk_sizes = [chunk.size(0) for chunk in chunks]
        chunk_starts = [sum(chunk_sizes[:i]) for i in range(len(chunks))]
        total_length = sum(chunk_sizes)

        mask = torch.zeros(total_length, total_length, device=device)

        # Set attention patterns for each chunk
        for i, (size, start, is_vis) in enumerate(
            zip(chunk_sizes, chunk_starts, is_visual, strict=False)
        ):
            # Set diagonal blocks
            if is_vis:
                # Visual chunks have full self-attention
                mask[start : start + size, start : start + size] = 1.0
            else:
                # Text chunks have causal attention
                mask[start : start + size, start : start + size] = self._create_causal_mask(
                    size, device
                )

            # Allow attention to previous chunks
            if i > 0:
                mask[start : start + size, :start] = 1.0

        return mask

    def _fuse_with_images(
        self,
        valid_embeddings: torch.Tensor,
        visual_features: torch.Tensor,
        img_positions: torch.Tensor,
        need_complex_mask: bool,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        """
        Fuse text embeddings with visual features at image token positions.
        Creates an interleaved sequence of text and visual embeddings.
        """
        valid_length = valid_embeddings.size(0)
        num_img_tokens = len(img_positions)

        # 计算每个图像标记对应的视觉特征数量
        visual_per_token = visual_features.size(0) // num_img_tokens

        # 创建融合序列
        fused_chunks: list[torch.Tensor] = []
        is_visual: list[bool] = []

        # 追踪当前处理位置
        current_pos = 0
        visual_idx = 0

        # 按原始顺序重建序列
        for img_pos in img_positions:
            img_pos = int(img_pos.item())

            # 添加图像位置之前的文本（如果有）
            if img_pos > current_pos:
                text_chunk = valid_embeddings[current_pos:img_pos]
                fused_chunks.append(text_chunk)
                is_visual.append(False)

            # 添加图像特征
            visual_chunk = visual_features[
                visual_idx * visual_per_token : (visual_idx + 1) * visual_per_token
            ]
            fused_chunks.append(visual_chunk)
            is_visual.append(True)
            visual_idx += 1

            # 更新当前位置
            current_pos = img_pos + 1

        # 添加最后剩余的文本（如果有）
        if current_pos < valid_length:
            text_chunk = valid_embeddings[current_pos:valid_length]
            fused_chunks.append(text_chunk)
            is_visual.append(False)

        # 连接所有块
        fused_embeddings = torch.cat(fused_chunks, dim=0)
        fused_length = fused_embeddings.size(0)

        # 创建注意力掩码（如果需要）
        mask: torch.Tensor | None = None
        if need_complex_mask:
            mask = self._create_fusion_mask(fused_chunks, is_visual, device)

        return fused_embeddings, mask, fused_length

    def _process_batch_item(
        self,
        text: torch.Tensor,
        mask: torch.Tensor,
        embeddings: torch.Tensor,
        visual_features: torch.Tensor,
        image_token_id: int,
        need_complex_mask: bool,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        # Extract valid portions
        valid_length = int(mask.sum().item())
        valid_text = text[:valid_length]
        valid_embeddings = embeddings[:valid_length]

        # Find image token positions
        img_positions = (valid_text == image_token_id).nonzero(as_tuple=True)[0]

        # Process based on whether image tokens exist
        if len(img_positions) == 0:
            return self._process_no_image_case(valid_embeddings, need_complex_mask, device)

        return self._fuse_with_images(
            valid_embeddings, visual_features, img_positions, need_complex_mask, device
        )

    @override
    def forward(
        self,
        visual_features: tuple[torch.Tensor, ...] | None,
        texts: torch.Tensor | None,
        embeddings: nn.Module,
        image_token_id: int,
        pad_token_id: int,
        mask_format: str = "2d",
    ) -> ForwardOutput:
        if texts is None:
            batch_size, _ = cast(tuple[torch.Tensor, ...], visual_features)[0].shape
            device: torch.device = cast(tuple[torch.Tensor, ...], visual_features)[0].device
        else:
            batch_size, _ = texts.shape
            device = texts.device

        # Create padding mask and determine mask complexity
        padding_mask = (texts != pad_token_id).bool()  # [batch_size, seq_len]
        need_complex_mask = mask_format in ["4d", "3d"]

        # Process visual features
        if texts is None:
            projected_visual = self._process_visual_features(
                cast(tuple[torch.Tensor, ...], visual_features), batch_size
            ).features

            return ForwardOutput(
                torch.cat(projected_visual, dim=0),
                torch.zeros(batch_size, 0, device=device, dtype=torch.bool),
            )

        projected_visual = self._process_visual_features(
            cast(tuple[torch.Tensor, ...], visual_features), batch_size
        ).features

        # Get text embeddings
        text_embeddings = embeddings(texts)  # [batch_size, seq_len, text_dim]

        # Process each item in batch
        fused_embeddings_list: list[torch.Tensor] = []
        attention_mask_list: list[torch.Tensor | None] = []
        valid_lengths_list: list[int] = []

        for batch_idx in range(batch_size):
            result: tuple[torch.Tensor, torch.Tensor | None, int] = self._process_batch_item(
                texts[batch_idx],
                padding_mask[batch_idx],
                text_embeddings[batch_idx],
                projected_visual[batch_idx],
                image_token_id,
                need_complex_mask,
                device,
            )
            embedding: torch.Tensor = result[0]
            mask: torch.Tensor | None = result[1]
            length: int = result[2]

            fused_embeddings_list.append(embedding)
            if need_complex_mask:
                attention_mask_list.append(mask)
            valid_lengths_list.append(length)

        # Pad to maximum length
        max_length: int = max(valid_lengths_list)

        padded_embeddings = torch.zeros(
            batch_size,
            max_length,
            text_embeddings.size(-1),
            device=device,
            dtype=text_embeddings.dtype,
        )

        # Add embeddings with proper padding
        for i, (embed, length) in enumerate(
            zip(fused_embeddings_list, valid_lengths_list, strict=False)
        ):
            padded_embeddings[i, :length] = embed

        # Create final attention mask based on format
        if mask_format == "2d":
            padded_mask = torch.zeros(batch_size, max_length, device=device, dtype=torch.bool)
            for i, length in enumerate(valid_lengths_list):
                padded_mask[i, :length] = True
            return ForwardOutput(padded_embeddings, padded_mask)

        # Create complex attention mask
        padded_attention_mask: torch.Tensor = torch.zeros(
            batch_size,
            max_length,
            max_length,
            device=device,
            dtype=torch.float32,
        )

        for i, (mask, length) in enumerate(
            zip(attention_mask_list, valid_lengths_list, strict=False)
        ):
            padded_attention_mask[i, :length, :length] = cast(torch.Tensor, mask)

        log.debug(f"padded_embeddings: {padded_embeddings.shape}")
        log.debug(f"padded_attention_mask: {padded_attention_mask.shape}")

        # Return in requested format
        if mask_format == "4d":
            return ForwardOutput(padded_embeddings, padded_attention_mask.unsqueeze(1))

        return ForwardOutput(padded_embeddings, padded_attention_mask)
