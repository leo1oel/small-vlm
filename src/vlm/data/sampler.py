from typing import override

import torch
from torch.utils.data import Sampler


def split_to_even_chunks(
    indices: list[int], lengths: list[int], num_chunks: int
) -> list[list[int]]:
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(
    lengths: list[int], batch_size: int, world_size: int, generator: torch.Generator | None = None
) -> list[int]:
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(length != 0 for length in lengths), "Should not have zero length."
    if all(length > 0 for length in lengths) or all(length < 0 for length in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(
        *[(i, length) for i, length in enumerate(lengths) if length > 0], strict=False
    )
    lang_indices, lang_lengths = zip(
        *[(i, -length) for i, length in enumerate(lengths) if length < 0], strict=False
    )

    mm_shuffle = [
        mm_indices[i]
        for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)
    ]
    lang_shuffle = [
        lang_indices[i]
        for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)
    ]
    megabatch_size = world_size * batch_size
    mm_megabatches = [
        mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)
    ]
    lang_megabatches = [
        lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)
    ]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(
    lengths: list[int], batch_size: int, world_size: int, generator: torch.Generator | None = None
) -> list[int]:
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [
        indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)
    ]
    megabatches = [
        sorted(megabatch, key=lambda i: lengths[i], reverse=True)  # pyright: ignore
        for megabatch in megabatches
    ]
    megabatches = [
        split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches
    ]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class MultiModalLengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: list[int] | None = None,
        generator: torch.Generator | None = None,
        group_by_modality: bool = False,
    ):
        super().__init__()
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size: int = batch_size
        self.world_size: int = world_size
        self.lengths: list[int] = lengths
        self.generator: torch.Generator | None = generator
        self.group_by_modality: bool = group_by_modality

    def __len__(self):
        return len(self.lengths)

    @override
    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        else:
            indices = get_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        return iter(indices)
