import logging
from abc import ABC, abstractmethod
from typing import override

import torch
import torch.nn as nn

from ...config.config_schema import ConnectorConfig

log: logging.Logger = logging.getLogger(name=__name__)


class Connector(nn.Module, ABC):
    def __init__(self, config: ConnectorConfig) -> None:
        super().__init__()
        self.config: ConnectorConfig = config
        self.name: str = self.config.name
        self.projection_layer: nn.Module = self.build_projection_layer()
        self.initialize_layers()

    @abstractmethod
    def _build_projection_layer(self) -> nn.Module:
        pass

    def build_projection_layer(self) -> nn.Module:
        log.info(
            f"[bold green]Building projection layer for[/bold green] [bold blue] {self.name} [/bold blue] [bold green]connector[/bold green]"
        )
        return self._build_projection_layer()

    @abstractmethod
    def _initialize_layers(self) -> None:
        pass

    def initialize_layers(self) -> None:
        log.info(
            f"[bold green]Initializing layers for[/bold green] [bold blue] {self.name} [/bold blue] [bold green]connector[/bold green]"
        )
        self._initialize_layers()

    @override
    def forward(
        self,
        visual_features: tuple[torch.Tensor, ...],
        texts: torch.Tensor,
        embeddings: nn.Embedding,
        image_token_id: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = texts.shape

        projected_visual_features: list[torch.Tensor] = []
        for _batch_idx, visual_feature in enumerate(visual_features):
            flattened_features: torch.Tensor = visual_feature.view(
                -1, visual_feature.size(-1)
            )  # [image_num*feature, feature_dimension]
            projected: torch.Tensor = self.projection(
                flattened_features
            )  # [image_num*feature, text_dim]
            projected_visual_features.append(projected)

        text_embeddings: torch.Tensor = embeddings(texts)  # [batch_size, seq_len, text_dim]

        fused_embeddings_list: list[torch.Tensor] = []
        attention_mask_list: list[torch.Tensor] = []

        for batch_idx in range(batch_size):
            current_text: torch.Tensor = texts[batch_idx]  # [seq_len]
            current_text_embeddings: torch.Tensor = text_embeddings[
                batch_idx
            ]  # [seq_len, text_dim]
            image_token_positions: torch.Tensor = (current_text == image_token_id).nonzero(
                as_tuple=True
            )[0]
            num_image_tokens: int = len(image_token_positions)

            if num_image_tokens == 0:
                mask: torch.Tensor = torch.tril(torch.ones(seq_len, seq_len, device=texts.device))
                fused_embeddings_list.append(current_text_embeddings)
                attention_mask_list.append(mask)
                continue

            current_visual_features: torch.Tensor = projected_visual_features[
                batch_idx
            ]  # [image_num*feature, text_dim]

            text_embedding_chunks: list[torch.Tensor] = []
            start_idx: int = 0

            for img_pos_tensor in image_token_positions:
                img_pos = int(img_pos_tensor.item())
                if img_pos > start_idx:
                    text_embedding_chunks.append(current_text_embeddings[start_idx:img_pos])
                    start_idx = img_pos + 1

            if start_idx < seq_len:
                text_embedding_chunks.append(current_text_embeddings[start_idx:seq_len])

            num_visual_chunks: int = len(image_token_positions)
            visual_features_per_chunk: int = current_visual_features.size(0) // num_visual_chunks

            fused_chunks: list[torch.Tensor] = []
            for i in range(num_visual_chunks):
                if i < len(text_embedding_chunks):
                    fused_chunks.append(text_embedding_chunks[i])

                start: int = i * visual_features_per_chunk
                end: int = (i + 1) * visual_features_per_chunk
                fused_chunks.append(current_visual_features[start:end])

            if num_visual_chunks < len(text_embedding_chunks):
                fused_chunks.append(text_embedding_chunks[-1])

            fused_embeddings: torch.Tensor = torch.cat(fused_chunks, dim=0)

            fused_length: int = fused_embeddings.size(0)
            attention_mask: torch.Tensor = torch.zeros(
                fused_length, fused_length, device=texts.device
            )

            current_pos: int = 0

            for i, chunk in enumerate(fused_chunks):
                chunk_size: int = chunk.size(0)

                if i % 2 == 1 and i < 2 * num_visual_chunks:
                    attention_mask[
                        current_pos : current_pos + chunk_size,
                        current_pos : current_pos + chunk_size,
                    ] = 1.0
                else:
                    attention_mask[
                        current_pos : current_pos + chunk_size,
                        current_pos : current_pos + chunk_size,
                    ] = torch.tril(torch.ones(chunk_size, chunk_size, device=texts.device))

                if i > 0:
                    for j in range(i):
                        prev_chunk_size: int = fused_chunks[j].size(0)
                        prev_start: int = sum(fused_chunks[k].size(0) for k in range(j))

                        attention_mask[
                            current_pos : current_pos + chunk_size,
                            prev_start : prev_start + prev_chunk_size,
                        ] = 1.0

                current_pos += chunk_size

            fused_embeddings_list.append(fused_embeddings)
            attention_mask_list.append(attention_mask)

        max_length: int = max(embed.size(0) for embed in fused_embeddings_list)

        padded_embeddings: torch.Tensor = torch.zeros(
            batch_size,
            max_length,
            text_embeddings.size(-1),
            device=texts.device,
            dtype=text_embeddings.dtype,
        )
        padded_attention_mask: torch.Tensor = torch.zeros(
            batch_size,
            max_length,
            max_length,
            device=texts.device,
            dtype=attention_mask_list[0].dtype,
        )

        log.debug(f"[bold yellow]padded_embeddings: {padded_embeddings.shape}[/bold yellow]")
        log.debug(
            f"[bold yellow]padded_attention_mask: {padded_attention_mask.shape}[/bold yellow]"
        )

        for i, (embed, mask) in enumerate(
            zip(fused_embeddings_list, attention_mask_list, strict=False)
        ):
            length: int = embed.size(0)
            padded_embeddings[i, :length] = embed
            padded_attention_mask[i, :length, :length] = mask

        return padded_embeddings, padded_attention_mask.unsqueeze(1)

    @abstractmethod
    def projection(self, visual_features: torch.Tensor) -> torch.Tensor:
        pass
