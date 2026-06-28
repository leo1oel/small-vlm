"""Prepared-WebDataset (energon `CrudeWebdataset`) streaming for the unified
VLM — the in-tar image layout, alongside the loose-file jsonl layout in
``energon_dataset.py``.

Two layouts exist on the blob:

* **jsonl-loose** (``dataset.folders``): one ``train.jsonl`` per folder + loose
  media files; the cooker (``cook_mm_chat``) fetches each image with one Azure
  GET via ``media_root.get(path)``.
* **prepared CrudeWebdataset** (``dataset.wds_path``, this module): the output of
  ``energon prepare`` — ``{00000..NNNNN}.tar`` shards (image bytes bundled IN the
  tar) + a ``.nv-meta/`` dir (``dataset.yaml`` with ``__class__:
  CrudeWebdataset``, ``split.yaml``, ``index.sqlite`` …). One sequential GET
  streams ~10k samples, so there is no per-image round-trip, no ~90 s cold-start
  fill, and far fewer fat-tail stragglers (data/datapipe-rootcause-m6).

The only structural difference is the cooker: a prepared CrudeWebdataset hands
the cooker a raw sample dict whose fields ARE the in-tar members
(``{__key__, __shard__, '<hash>_img0.jpg': <bytes>, 'json': <bytes>, …}``) — no
``media_root``, no ``crude_type`` subflavor. ``cook_mm_chat_wds`` reads the image
bytes out of those fields; everything downstream (``encode_sample``, the
collator, bucketing, BREEN query injection, savable resume, the model contract)
is shared verbatim with the jsonl path by subclassing the ``energon_dataset``
task encoders and overriding only ``cookers``.

Wired by ``build_energon_train_loader`` when ``dataset.wds_path`` is set.
"""

import json
import logging
from typing import Any

from .data_arguments import DataArguments

# Reuse the guarded energon imports + the shared task encoders / helpers from the
# jsonl module. energon_dataset imports THIS module lazily (inside
# build_energon_train_loader), so importing it here at module load is acyclic.
from .energon_dataset import (
    Cooker,
    MMChatRawSample,
    VLMBucketedChatTaskEncoder,
    VLMChatTaskEncoder,
    VLMGenTaskEncoder,
    WorkerConfig,
    _check_distributed_consistency,
    _require_credentials,
    _streaming_import_error,
    basic_sample_keys,
    get_savable_loader,
    get_train_dataset,
    remote_url,
    stateless,
)

log: logging.Logger = logging.getLogger(name=__name__)

#: Tar-member extensions energon hands the cooker as raw image bytes.
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")


@stateless  # pyright: ignore[reportUntypedFunctionDecorator]
def cook_mm_chat_wds(sample: dict) -> MMChatRawSample:
    """Cook a prepared CrudeWebdataset sample into the SAME ``MMChatRawSample``
    the jsonl cooker emits — but read image bytes from the in-tar sample fields
    instead of ``media_root.get(path)`` (which a prepared WDS has no metadataset
    aux for). Stateless (energon asserts cookers are stateless)."""
    rec = sample["json"]
    if isinstance(rec, bytes | bytearray):
        rec = json.loads(bytes(rec))
    elif isinstance(rec, str):
        rec = json.loads(rec)

    # In-tar media fields (everything energon decoded as raw image bytes), in a
    # deterministic order so multi-image positional fallback is stable.
    media_fields = sorted(
        k for k in sample if isinstance(k, str) and k.lower().endswith(_IMG_EXTS)
    )
    media_set = set(media_fields)

    # Collect image items in placeholder order; fail loud on audio. A prepared
    # WDS sample carries NO in-tar audio bytes, so an audio content item would
    # still emit an <audio> placeholder downstream with no backing feature and
    # silently mis-splice the batch — surface it instead of dropping it.
    image_bases: list[str] = []
    for msg in rec.get("messages", ()):
        content = msg.get("content")
        if not isinstance(content, list):
            continue  # plain-string content => text-only message
        for item in content:
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            if itype == "audio":
                raise ValueError(
                    f"WDS sample {sample.get('__key__')!r} carries an 'audio' "
                    "content item, but prepared-WebDataset audio is unsupported "
                    "(no in-tar audio bytes) — it would emit an <audio> "
                    "placeholder with no backing feature and mis-splice the "
                    "batch. Use the jsonl-loose layout for audio data."
                )
            if itype == "image":
                image_bases.append(str(item.get("path", "")).split("/")[-1])

    # Two-pass field assignment: bind every explicit basename match FIRST
    # (marking those fields used), then positional-fallback the still-unmatched
    # items over the still-unused fields. A single pass can let a positional
    # fallback consume a field that a later item names by basename, duplicating
    # one image's bytes and dropping another in mixed-naming multi-image samples.
    assigned: list[str | None] = [None] * len(image_bases)
    used: set[str] = set()
    for i, base in enumerate(image_bases):
        if base in media_set and base not in used:  # tar field == path basename
            assigned[i] = base
            used.add(base)
    fallback = [f for f in media_fields if f not in used]
    fi = 0
    for i, base in enumerate(image_bases):
        if assigned[i] is not None:
            continue
        if fi >= len(fallback):
            raise ValueError(
                f"WDS sample {sample.get('__key__')!r} declares an image item "
                f"({base!r}) with no matching in-tar field "
                f"(available: {media_fields})"
            )
        assigned[i] = fallback[fi]
        used.add(fallback[fi])
        fi += 1

    image_bytes: list[bytes] = [sample[f] for f in assigned]

    return MMChatRawSample(
        **basic_sample_keys(sample),
        messages=rec.get("messages", []),
        image_bytes=image_bytes,
        audio_bytes=[],  # prepared WDS bee_stage2 carries no in-tar audio
        source=rec.get("source"),
    )


# The prepared shards carry NO `crude_type` subflavor, so register the cooker
# with no subflavor gate (unlike the jsonl `has_subflavors=...` gate). Only the
# `cookers` attribute differs from the jsonl encoders — encode_sample / batch /
# bucketing / query injection are inherited unchanged.
class WDSChatTaskEncoder(VLMChatTaskEncoder):  # pyright: ignore[reportUntypedBaseClass]
    cookers = [Cooker(cook_mm_chat_wds)]


class WDSBucketedChatTaskEncoder(VLMBucketedChatTaskEncoder):  # pyright: ignore[reportUntypedBaseClass]
    cookers = [Cooker(cook_mm_chat_wds)]


class WDSGenTaskEncoder(VLMGenTaskEncoder):  # pyright: ignore[reportUntypedBaseClass]
    cookers = [Cooker(cook_mm_chat_wds)]


def resolve_wds_path(wds_path: str) -> str:
    """A full ``msc://…`` / ``s3://…`` / local-fs URL is used verbatim; a bare
    container-relative path is resolved through the same MSC profile/container as
    ``dataset.folders`` (so ``yiming/bee_stage2/train-wds`` ->
    ``msc://azure/data/yiming/bee_stage2/train-wds``)."""
    p = str(wds_path).strip()
    if "://" in p:
        return p
    return remote_url(p)


def select_wds_task_encoder(
    dataset_config: Any, processor: Any, data_args: DataArguments
) -> Any:
    """Pick the WDS task encoder mirroring the jsonl branch: generation ->
    bucketed -> plain."""
    if str(getattr(dataset_config, "task", "understanding")) == "generation":
        gen_psz = getattr(dataset_config, "gen_patch_size", None)
        return WDSGenTaskEncoder(
            processor,
            data_args,
            resolution=int(getattr(dataset_config, "gen_resolution", 384)),
            caption_max_len=int(getattr(dataset_config, "gen_caption_max_len", 128)),
            patch_size=int(gen_psz) if gen_psz else None,
        )
    length_buckets = getattr(dataset_config, "length_buckets", None)
    if length_buckets:
        return WDSBucketedChatTaskEncoder(
            processor,
            data_args,
            list(length_buckets),
            batch_token_budget=getattr(dataset_config, "batch_token_budget", None),
        )
    return WDSChatTaskEncoder(processor, data_args)


def build_wds_train_loader(
    dataset_config: Any,
    processor: Any,
    data_args: DataArguments,
    batch_size: int | None,
    *,
    task_encoder: Any = None,
    worker_config: Any = None,
    **savable_loader_kwargs: Any,
) -> Any:
    """Stream a prepared energon CrudeWebdataset (``dataset.wds_path``) into the
    same trainer-ready batch dicts the jsonl loader yields.

    No jsonl download / index / metadataset generation — energon reads the
    ``.nv-meta`` dir directly. Multi-GPU rank sharding, savable resume, the
    watchdog and the model contract are identical to the jsonl path."""
    if _streaming_import_error is not None:
        raise ImportError(
            "dataset.wds_path requires the streaming dependencies — install "
            "megatron-energon (with the [azure-storage-blob] extra) and "
            f"multi-storage-client. Original error: {_streaming_import_error}"
        ) from _streaming_import_error

    path = resolve_wds_path(dataset_config.wds_path)
    if path.startswith("msc://"):
        _require_credentials()  # local-fs paths (tests) need no Azure creds
    # Every rank must agree on the same prepared dataset (else disjoint slices
    # would silently overlap / drop). The path fully identifies the source.
    _check_distributed_consistency(("wds", path))

    wc = worker_config or WorkerConfig.default_worker_config(
        num_workers=dataset_config.num_workers
    )
    if task_encoder is None:
        task_encoder = select_wds_task_encoder(dataset_config, processor, data_args)
    # Token-budget bucketing sizes every batch itself; energon then requires the
    # loader-wide batch_size to be None ("one of the two should be None").
    if getattr(task_encoder, "batch_token_budget", None):
        batch_size = None

    log.info("building prepared-WDS loader on %s", path)
    dataset = get_train_dataset(
        path,
        batch_size=batch_size,
        shuffle_buffer_size=dataset_config.shuffle_buffer_size,
        max_samples_per_sequence=dataset_config.max_samples_per_sequence,
        worker_config=wc,
        task_encoder=task_encoder,
    )
    # Mirror the jsonl loader: lengthen the rolling-checkpoint interval so the
    # (caught, non-fatal) SSL-socket PicklingError noise doesn't balloon the
    # logs over a multi-day stream (durable resume uses the separate coordinated
    # get_checkpoint). watchdog_initial_timeout_seconds is threaded in by the
    # caller (build_energon_train_loader).
    savable_loader_kwargs.setdefault("checkpoint_every_sec", 900)
    return get_savable_loader(dataset, **savable_loader_kwargs)
