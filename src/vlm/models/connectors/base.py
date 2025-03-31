import logging
from abc import ABC, abstractmethod
from typing import override

import torch
import torch.nn as nn

from ...config.config_schema import ConnectorConfig

log: logging.Logger = logging.getLogger(name=__name__)


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
        pad_token_id: int,
        mask_format: str = "2d",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _ = texts.shape
        padding_mask = (texts != pad_token_id).bool()  # [batch_size, seq_len]
        need_complex_mask = mask_format in ["4d", "3d"]

        if len(visual_features) == 1 and visual_features[0].size(0) == batch_size:
            batch_visual_features: torch.Tensor = visual_features[0]
            projected_visual_features: list[torch.Tensor] = []
            batch_projected = self.projection(
                batch_visual_features.view(-1, batch_visual_features.size(-1))
            )
            batch_projected = batch_projected.view(batch_size, -1, batch_projected.size(-1))
            projected_visual_features = [batch_projected[i] for i in range(batch_size)]
        else:
            projected_visual_features = []
            for _batch_idx, visual_feature in enumerate(visual_features):
                flattened_features: torch.Tensor = visual_feature.view(-1, visual_feature.size(-1))
                projected: torch.Tensor = self.projection(flattened_features)
                projected_visual_features.append(projected)

        text_embeddings: torch.Tensor = embeddings(texts)  # [batch_size, seq_len, text_dim]

        fused_embeddings_list: list[torch.Tensor] = []
        attention_mask_list: list[torch.Tensor] = []
        valid_lengths_list: list[int] = []

        for batch_idx in range(batch_size):
            current_text: torch.Tensor = texts[batch_idx]
            current_padding_mask: torch.Tensor = padding_mask[batch_idx]
            current_text_embeddings: torch.Tensor = text_embeddings[batch_idx]

            valid_length: int = int(current_padding_mask.sum().item())

            valid_text: torch.Tensor = current_text[:valid_length]
            valid_text_embeddings: torch.Tensor = current_text_embeddings[:valid_length]

            image_token_positions: torch.Tensor = (valid_text == image_token_id).nonzero(
                as_tuple=True
            )[0]
            num_image_tokens: int = len(image_token_positions)

            if num_image_tokens == 0:
                fused_embeddings_list.append(valid_text_embeddings)
                if need_complex_mask:
                    valid_mask: torch.Tensor = torch.tril(
                        torch.ones(valid_length, valid_length, device=texts.device)
                    )
                    attention_mask_list.append(valid_mask)
                valid_lengths_list.append(valid_length)
                continue

            current_visual_features: torch.Tensor = projected_visual_features[batch_idx]

            text_embedding_chunks: list[torch.Tensor] = []
            start_idx: int = 0

            for img_pos_tensor in image_token_positions:
                img_pos = int(img_pos_tensor.item())
                if img_pos > start_idx:
                    text_embedding_chunks.append(valid_text_embeddings[start_idx:img_pos])
                    start_idx = img_pos + 1

            if start_idx < valid_length:
                text_embedding_chunks.append(valid_text_embeddings[start_idx:valid_length])

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

            fused_embeddings_list.append(fused_embeddings)
            valid_lengths_list.append(fused_length)
            if need_complex_mask:
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

                attention_mask_list.append(attention_mask)

        max_length: int = max(valid_lengths_list)

        padded_embeddings: torch.Tensor = torch.zeros(
            batch_size,
            max_length,
            text_embeddings.size(-1),
            device=texts.device,
            dtype=text_embeddings.dtype,
        )

        for i, embed in enumerate(fused_embeddings_list):
            length = embed.size(0)
            padded_embeddings[i, :length] = embed

        if mask_format == "2d":
            padded_mask = torch.zeros(batch_size, max_length, device=texts.device, dtype=torch.bool)
            for i, length in enumerate(valid_lengths_list):
                padded_mask[i, :length] = True
            return padded_embeddings, padded_mask

        else:
            padded_attention_mask: torch.Tensor = torch.zeros(
                batch_size,
                max_length,
                max_length,
                device=texts.device,
                dtype=torch.float32,
            )

            for i, mask in enumerate(attention_mask_list):
                length: int = mask.size(0)
                padded_attention_mask[i, :length, :length] = mask

            log.debug(f"[bold yellow]padded_embeddings: {padded_embeddings.shape}[/bold yellow]")
            log.debug(
                f"[bold yellow]padded_attention_mask: {padded_attention_mask.shape}[/bold yellow]"
            )

            if mask_format == "4d":
                return padded_embeddings, padded_attention_mask.unsqueeze(1)
            else:
                return padded_embeddings, padded_attention_mask

    @abstractmethod
    def projection(self, visual_features: torch.Tensor) -> torch.Tensor:
        pass
